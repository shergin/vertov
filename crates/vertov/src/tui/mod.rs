//! The TUI shell: terminal lifecycle and the event/poll loop. All state
//! lives in [`app`], all drawing in [`view`] — this module only wires
//! events to them (deterministic core, effectful shell).

pub mod app;
pub mod theme;
pub mod view;

use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyEventKind};

use vertov_model::Project;

use crate::Args;
use app::{Action, App};

pub fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut project = Project::new(&args.logdir);
    if !args.no_cache {
        project.load_cache();
    }
    project.refresh()?;
    let mut app = App::new(project, args.interval, &args.tag);
    // Land with the cursor on the first run, not in a cursorless limbo.
    app.runs.cursor = app.project.runs.keys().next().cloned();
    // Probe before raw mode (§5.6): the capability query reads terminal
    // replies, which a raw-mode event loop would swallow as input. The
    // result is cached for the process. A forced protocol skips the probe.
    app.graphics = args
        .pixels
        .graphics(|| malevich::pixel::Capabilities::detect_for(&std::io::stdout()));

    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut app);
    ratatui::restore();

    if !args.no_cache {
        // Best-effort: the cache is an accelerator, never a requirement.
        let _ = app.project.save_cache();
    }
    // No alt-screen surprises: leave a plain rendering of the final view in
    // scrollback.
    print!("{}", final_frame(&app));
    result.map_err(Into::into)
}

fn event_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
) -> std::io::Result<()> {
    let mut last_refresh = Instant::now();
    // Repaint the image on data or state change, not on a timer (§5.6):
    // when nothing changed, the previous transmission stays on screen.
    let mut emit_needed = true;
    let mut image_on_screen = false;
    // The panel rects of the previous frame, to detect layout changes.
    let mut previous_areas: Vec<ratatui::layout::Rect> = Vec::new();
    loop {
        app.ensure_materialized()?;
        let mut panels = Vec::new();
        terminal.draw(|frame| panels = view::draw(frame, app, false))?;

        // Fresh ground for transparent images: cells under a skip-reserved
        // rect survive ratatui's diff untouched, so whatever was there —
        // a previous view's table, or the previous emission's own chrome
        // (tick labels, a longer title) — would show through the image's
        // alpha or linger beside shorter text. Before every emission, draw
        // a blanking frame: the panel rects painted with real spaces
        // instead of skip cells, physically clearing them. (Not
        // `Terminal::clear`: that round-trips a cursor-position query,
        // which hangs headless terminals and races the raw-mode event
        // reader. This becomes unnecessary once malevich's pixel block
        // pads its rows to own its full rectangle.)
        let areas: Vec<ratatui::layout::Rect> = panels.iter().map(view::PixelPanel::area).collect();
        if !areas.is_empty() && (areas != previous_areas || emit_needed) {
            if image_on_screen {
                delete_kitty_images(app)?;
                image_on_screen = false;
            }
            terminal.draw(|frame| panels = view::draw(frame, app, true))?;
            emit_needed = true;
        }
        previous_areas = areas;

        if panels.is_empty() {
            // Left the charts (view switch, help overlay): cell content
            // repaints the area, but kitty images live on their own layer
            // and need an explicit goodbye.
            if image_on_screen {
                delete_kitty_images(app)?;
                image_on_screen = false;
                emit_needed = true;
            }
        } else if let Some(graphics) = &app.graphics
            && emit_needed
        {
            // Panel count can shrink (compare grid resize): clear kitty
            // placements before re-emitting so orphans never linger.
            if image_on_screen {
                delete_kitty_images(app)?;
            }
            for panel in &panels {
                emit_pixels(panel, graphics)?;
            }
            emit_needed = false;
            image_on_screen = true;
        }

        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => {
                    if app.update(key) == Action::Quit {
                        return Ok(());
                    }
                    emit_needed = true;
                }
                Event::Resize(_, _) => emit_needed = true,
                _ => {}
            }
        }

        if app.force_refresh || (!app.paused && last_refresh.elapsed() >= app.interval) {
            let report = app.project.refresh()?;
            if report.new_points > 0 || report.new_files > 0 {
                app.last_change = Instant::now();
                emit_needed = true;
            }
            last_refresh = Instant::now();
            app.force_refresh = false;
        }
    }
}

/// Prints the chart panel as a real image: cursor to the panel's top row,
/// then malevich's absolute-column block — chrome as text, plot rectangle
/// as pixels (§5.6: emit after the frame).
fn emit_pixels(panel: &view::PixelPanel, graphics: &malevich::pixel::Graphics) -> std::io::Result<()> {
    use crossterm::{cursor::MoveTo, style::Print};
    use std::io::Write as _;
    let area = panel.area();
    let frame = malevich::Frame {
        width: area.width as usize,
        height: area.height as usize,
        charset: malevich::Charset::Quadrants,
        color: malevich::ColorMode::TrueColor,
        theme: malevich::Theme::DARK,
    };
    let block = panel
        .plot()
        .render_pixels_at(&frame, graphics, area.x as usize);
    let mut out = std::io::stdout();
    crossterm::queue!(out, MoveTo(area.x, area.y), Print(block))?;
    out.flush()
}

/// Deletes all visible kitty-protocol image placements. Other protocols
/// paint into cells, which ordinary redraws already replace.
fn delete_kitty_images(app: &App) -> std::io::Result<()> {
    use std::io::Write as _;
    if app.graphics.map(|graphics| graphics.protocol) == Some(malevich::pixel::Protocol::Kitty) {
        let mut out = std::io::stdout();
        out.write_all(b"\x1b_Ga=d,d=A\x1b\\")?;
        out.flush()?;
    }
    Ok(())
}

/// A plain-text rendering of the current view for scrollback: the runs
/// table as text, or the current chart through a detected frame.
fn final_frame(app: &App) -> String {
    use crate::chart::ChartData;
    use crate::table::{Cell, Format, Table, fmt_duration};
    match app.view {
        app::View::Runs | app::View::Hparams | app::View::Distributions => {
            let rows = app
                .run_rows()
                .into_iter()
                .map(|row| {
                    vec![
                        Cell::Text(row.name),
                        Cell::Text(row.status.to_owned()),
                        Cell::Int(row.series as i64),
                        Cell::Int(row.points as i64),
                        Cell::Int(row.restarts as i64),
                        row.step.map_or(Cell::Empty, Cell::Int),
                        row.duration
                            .map_or(Cell::Empty, |seconds| Cell::Text(fmt_duration(seconds))),
                    ]
                })
                .collect();
            Table {
                columns: ["run", "status", "series", "points", "restarts", "step", "duration"]
                    .map(String::from)
                    .to_vec(),
                rows,
            }
            .render(Format::Text)
        }
        app::View::Scalars | app::View::Compare => match app.current_tag() {
            Some(tag) => {
                let scope = app.scoped_runs();
                let data = ChartData::for_tag(&app.project, &scope, &tag, &app.chart_options());
                data.plot(&tag).render(&malevich::Frame::detect())
            }
            None => String::new(),
        },
    }
}
