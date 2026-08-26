//! Drawing: pure functions from [`App`] to a ratatui frame. No state, no
//! I/O — the payoff is the snapshot tests at the bottom, byte-for-byte over
//! a `TestBackend`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Row, Table};

use crate::chart::ChartData;
use crate::table::fmt_duration;
use crate::tui::app::{App, View};

pub fn draw(frame: &mut Frame, app: &App) {
    let [body, status] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(frame.area());
    match app.view {
        View::Runs => draw_runs(frame, app, body),
        View::Scalars => draw_scalars(frame, app, body),
    }
    draw_status(frame, app, status);
    if app.help {
        draw_help(frame, frame.area());
    }
}

fn draw_runs(frame: &mut Frame, app: &App, area: Rect) {
    let rows = app.run_rows();
    let cursor = app
        .runs
        .cursor
        .as_ref()
        .and_then(|cursor| rows.iter().position(|row| &row.name == cursor));
    let table_rows: Vec<Row> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let marker = if app.runs.selected.contains(&row.name) {
                "▸"
            } else {
                " "
            };
            let style = if Some(index) == cursor {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            Row::new(vec![
                format!("{marker}{}", row.name),
                row.status.to_owned(),
                row.series.to_string(),
                row.points.to_string(),
                row.restarts.to_string(),
                row.step.map(|step| step.to_string()).unwrap_or_default(),
                row.duration.map(fmt_duration).unwrap_or_default(),
            ])
            .style(style)
        })
        .collect();

    let sort_marker = |label: &str| {
        if label == app.runs.sort.label() {
            format!("{label}{}", if app.runs.descending { "▼" } else { "▲" })
        } else {
            label.to_owned()
        }
    };
    let header = Row::new(vec![
        sort_marker("run"),
        "status".to_owned(),
        "series".to_owned(),
        sort_marker("points"),
        sort_marker("restarts"),
        sort_marker("step"),
        sort_marker("duration"),
    ])
    .style(Style::default().add_modifier(Modifier::BOLD));

    let title = panel_title("runs", &app.runs.filter, app.runs.editing_filter);
    let table = Table::new(
        table_rows,
        [
            Constraint::Fill(2),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(table, area);
}

fn draw_scalars(frame: &mut Frame, app: &App, area: Rect) {
    let [left, right] =
        Layout::horizontal([Constraint::Length(28), Constraint::Fill(1)]).areas(area);

    let tags = app.visible_tags();
    let current = app.current_tag();
    let items: Vec<ListItem> = tags
        .iter()
        .map(|tag| {
            let style = if Some(tag) == current.as_ref() {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            ListItem::new(tag.clone()).style(style)
        })
        .collect();
    let title = panel_title("tags", &app.scalars.filter, app.scalars.editing_filter);
    frame.render_widget(
        List::new(items).block(Block::default().borders(Borders::ALL).title(title)),
        left,
    );

    match current {
        Some(tag) => {
            let scope = app.scoped_runs();
            let data = ChartData::for_tag(&app.project, &scope, &tag, &app.chart_options());
            let title = chart_title(app, &tag, &data);
            frame.render_widget(data.plot(&title).widget(), right);
        }
        None => {
            frame.render_widget(
                Paragraph::new("no scalar tags match")
                    .block(Block::default().borders(Borders::ALL)),
                right,
            );
        }
    }
}

fn chart_title(app: &App, tag: &str, data: &ChartData) -> String {
    use std::fmt::Write as _;
    let mut title = tag.to_owned();
    if data.run_count > 1 {
        let _ = write!(title, " · {} runs", data.run_count);
    }
    if data.cut > 0 {
        let _ = write!(title, " · +{} not shown", data.cut);
    }
    if app.scalars.smooth > 0.0 {
        let _ = write!(title, " · smooth {:.1}", app.scalars.smooth);
    }
    if app.scalars.log_y {
        title.push_str(" · log");
    }
    match app.scalars.x_axis {
        crate::chart::XAxis::Step => {}
        crate::chart::XAxis::Wall => title.push_str(" · wall"),
        crate::chart::XAxis::Relative => title.push_str(" · relative"),
    }
    title
}

fn panel_title(name: &str, filter: &str, editing: bool) -> String {
    if editing {
        format!("{name} · /{filter}▏")
    } else if !filter.is_empty() {
        format!("{name} · /{filter}")
    } else {
        name.to_owned()
    }
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let mut left = format!(
        "vertov · {} · {} runs",
        app.project.root().display(),
        app.project.runs.len()
    );
    if app.project.dropped_records > 0 {
        left.push_str(&format!(" · {} records lost", app.project.dropped_records));
    }
    if app.project.dead_files > 0 {
        left.push_str(&format!(" · {} dead files", app.project.dead_files));
    }
    let state = if app.paused {
        "paused".to_owned()
    } else {
        let quiet = app.last_change.elapsed();
        if quiet > 2 * app.interval {
            format!("stale {}s", quiet.as_secs())
        } else {
            "live".to_owned()
        }
    };
    let line = match &app.message {
        Some(message) => Line::from(vec![
            Span::raw(message.clone()),
            Span::raw("  ·  "),
            Span::styled(state, Style::default().fg(Color::DarkGray)),
        ]),
        None => Line::from(vec![
            Span::raw(left),
            Span::raw("  ·  "),
            Span::raw(state),
            Span::styled("  ·  ? help", Style::default().fg(Color::DarkGray)),
        ]),
    };
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let text = "\
  q          quit
  ?          this help (any key closes)
  Tab 1 2    switch view: runs / scalars
  /          filter (Esc clears, Enter keeps)
  e          export current view as CSV, next to your shell
  p          pause polling      r  refresh now

  runs:      j/k move · space select for overlay · Enter open scalars
             s sort column · S reverse
             / takes a predicate too: lr > 1e-3 and status == active
  scalars:   j/k move tag (fuzzy /) · s/S smoothing -/+ · L log-y
             x cycle x axis: step/wall/relative · Esc back to runs";
    let lines = text.lines().count() as u16 + 2;
    let width = 64.min(area.width);
    let popup = Rect {
        x: area.width.saturating_sub(width) / 2,
        y: area.height.saturating_sub(lines) / 2,
        width,
        height: lines.min(area.height),
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("keys")),
        popup,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::app::App;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::time::Duration;
    use tfevents::writer::{events_file, scalar_event};
    use vertov_model::Project;

    fn test_app(name: &str) -> (App, tempdir::Guard) {
        let guard = tempdir::Guard::new(name);
        let wall = |step: i64| 1.7e9 + step as f64 * 10.0;
        let adam: Vec<Vec<u8>> = (0..10)
            .flat_map(|step| {
                [
                    scalar_event(wall(step), step, "train/loss", 8.0 / (step + 1) as f32),
                    scalar_event(wall(step), step, "train/acc", 0.1 * step as f32),
                ]
            })
            .collect();
        guard.write("adam/events.out.tfevents.1000.host", &events_file(&adam));
        let sgd: Vec<Vec<u8>> = (0..8)
            .map(|step| scalar_event(wall(step), step, "train/loss", 9.0 - step as f32))
            .collect();
        guard.write("sgd/events.out.tfevents.1000.host", &events_file(&sgd));

        let mut project = Project::new(guard.path());
        project.refresh().unwrap();
        (App::new(project, Duration::from_secs(5), ""), guard)
    }

    mod tempdir {
        use std::path::{Path, PathBuf};

        pub struct Guard(PathBuf);

        impl Guard {
            pub fn new(name: &str) -> Guard {
                let dir = std::env::temp_dir()
                    .join(format!("vertov-tui-test-{}-{name}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir);
                std::fs::create_dir_all(&dir).unwrap();
                Guard(dir)
            }

            pub fn path(&self) -> &Path {
                &self.0
            }

            pub fn write(&self, relative: &str, bytes: &[u8]) {
                let path = self.0.join(relative);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, bytes).unwrap();
            }
        }

        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    /// Renders the app and returns rows as strings — the status bar (which
    /// contains the temp path) excluded.
    fn snapshot(app: &App, width: u16, height: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..height.saturating_sub(1))
            .map(|y| {
                let mut line: String = (0..width)
                    .map(|x| buffer.cell((x, y)).unwrap().symbol().to_owned())
                    .collect();
                while line.ends_with(' ') {
                    line.pop();
                }
                line
            })
            .collect()
    }

    #[test]
    fn runs_view_snapshot() {
        let (mut app, _guard) = test_app("runs-view");
        app.runs.cursor = Some("adam".to_owned());
        app.runs.selected.insert("sgd".to_owned());
        let lines = snapshot(&app, 72, 8);
        let expected = [
            "┌runs──────────────────────────────────────────────────────────────────┐",
            "│run▲                 status series points   restarts step     duration│",
            "│ adam                active 2      20       0        9        90s     │",
            "│▸sgd                 active 1      8        0        7        70s     │",
            "│                                                                      │",
            "│                                                                      │",
            "└──────────────────────────────────────────────────────────────────────┘",
        ];
        assert_eq!(lines, expected, "runs view drifted:\n{}", lines.join("\n"));
    }

    #[test]
    fn scalars_view_lists_tags_and_draws_chart() {
        let (mut app, _guard) = test_app("scalars-view");
        app.view = View::Scalars;
        // train/loss exists in both runs; train/acc in adam only.
        app.scalars.cursor = Some("train/loss".to_owned());
        app.ensure_materialized().unwrap();
        let lines = snapshot(&app, 72, 12);
        let text = lines.join("\n");
        // Tag list shows both tags, chart panel titles the current one with
        // its run count. Glyph-exact chart goldens live in malevich.
        assert!(text.contains("train/acc"), "tags: {text}");
        assert!(text.contains("train/loss"), "tags: {text}");
        assert!(text.contains("2 runs"), "chart title: {text}");
        assert!(text.contains("adam"), "legend: {text}");
        assert!(text.contains("sgd"), "legend: {text}");
    }

    #[test]
    fn refresh_preserves_cursor_selection_and_filter() {
        let (mut app, guard) = test_app("state-preserved");
        app.runs.cursor = Some("sgd".to_owned());
        app.runs.selected.insert("adam".to_owned());
        app.scalars.filter = "loss".to_owned();

        // New data and a brand-new run arrive.
        let more: Vec<Vec<u8>> = (0..3)
            .map(|step| scalar_event(2.0e9, step, "train/loss", 1.0))
            .collect();
        guard.write("zeta/events.out.tfevents.1000.host", &events_file(&more));
        app.project.refresh().unwrap();

        let rows = app.run_rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(app.runs.cursor.as_deref(), Some("sgd"));
        assert!(app.runs.selected.contains("adam"));
        assert_eq!(app.visible_tags(), vec!["train/loss".to_owned()]);
    }

    #[test]
    fn help_overlay_renders() {
        let (mut app, _guard) = test_app("help");
        app.help = true;
        let text = snapshot(&app, 72, 20).join("\n");
        assert!(text.contains("keys"), "{text}");
        assert!(text.contains("switch view"), "{text}");
    }
}
