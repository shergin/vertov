//! The TUI shell: terminal lifecycle and the event/poll loop. All state
//! lives in [`app`], all drawing in [`view`] — this module only wires
//! events to them (deterministic core, effectful shell).

pub mod app;
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
    loop {
        app.ensure_materialized()?;
        terminal.draw(|frame| view::draw(frame, app))?;

        if event::poll(Duration::from_millis(200))?
            && let Event::Key(key) = event::read()?
            && key.kind != KeyEventKind::Release
            && app.update(key) == Action::Quit
        {
            return Ok(());
        }

        if app.force_refresh || (!app.paused && last_refresh.elapsed() >= app.interval) {
            let report = app.project.refresh()?;
            if report.new_points > 0 || report.new_files > 0 {
                app.last_change = Instant::now();
            }
            last_refresh = Instant::now();
            app.force_refresh = false;
        }
    }
}

/// A plain-text rendering of the current view for scrollback: the runs
/// table as text, or the current chart through a detected frame.
fn final_frame(app: &App) -> String {
    use crate::chart::ChartData;
    use crate::table::{Cell, Format, Table, fmt_duration};
    match app.view {
        app::View::Runs => {
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
        app::View::Scalars => match app.current_tag() {
            Some(tag) => {
                let scope = app.scoped_runs();
                let data = ChartData::for_tag(&app.project, &scope, &tag, &app.chart_options());
                data.plot(&tag).render(&malevich::Frame::detect())
            }
            None => String::new(),
        },
    }
}
