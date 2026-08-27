//! Drawing: pure functions from [`App`] to a ratatui frame. No state, no
//! I/O — the payoff is the snapshot tests at the bottom, byte-for-byte over
//! a `TestBackend`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Clear, List, ListItem, ListState, Paragraph, Row, Table, TableState,
};

use crate::chart::{ChartData, DistData};
use crate::table::fmt_duration;
use crate::tui::app::{App, View};
use crate::tui::theme;

/// A chart panel reserved for pixel graphics this frame: the cell diff left
/// its rect untouched (skip cells), and the caller emits this after the
/// frame flushes.
pub enum PixelPanel {
    Chart {
        area: Rect,
        data: ChartData,
        title: String,
        /// Shared x domain, for compare panels.
        domain: Option<(f64, f64)>,
        /// Legend-free rendering (compare panels).
        bare: bool,
    },
    Dist {
        area: Rect,
        data: DistData,
        title: String,
    },
}

impl PixelPanel {
    pub fn area(&self) -> Rect {
        match self {
            PixelPanel::Chart { area, .. } | PixelPanel::Dist { area, .. } => *area,
        }
    }

    /// The panel's plot, ready for pixel rendering.
    pub fn plot(&self) -> malevich::Plot<'_> {
        match self {
            PixelPanel::Chart {
                data,
                title,
                domain,
                bare,
                ..
            } => {
                if *bare {
                    data.compare_plot(title, *domain)
                } else {
                    data.plot(title)
                }
            }
            PixelPanel::Dist { data, title, .. } => data.plot(title),
        }
    }
}

/// Draws one frame. `blank_panels` is the layout-change special: pixel
/// panel rects are painted with real spaces (cleared) instead of
/// skip-reserved, giving a following transparent image fresh ground.
pub fn draw(frame: &mut Frame, app: &App, blank_panels: bool) -> Vec<PixelPanel> {
    let [header, body, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Fill(1),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    draw_header(frame, app, header);
    let panels = match app.view {
        View::Runs => {
            draw_runs(frame, app, body);
            Vec::new()
        }
        View::Scalars => draw_scalars(frame, app, body, blank_panels),
        View::Compare => draw_compare(frame, app, body, blank_panels),
        View::Hparams => {
            draw_hparams(frame, app, body);
            Vec::new()
        }
        View::Distributions => draw_distributions(frame, app, body, blank_panels),
    };
    draw_status(frame, app, status);
    if app.help {
        draw_help(frame, frame.area());
    }
    panels
}

/// The header: brand mark, the view tabs (the accent shows where you are),
/// and the run count on the right.
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![
        Span::styled("▌ ", theme::accent()),
        Span::styled("vertov", theme::header()),
        Span::raw("   "),
    ];
    for (view, key, label) in [
        (View::Runs, "1", "runs"),
        (View::Scalars, "2", "scalars"),
        (View::Compare, "3", "compare"),
        (View::Hparams, "4", "hparams"),
        (View::Distributions, "5", "dists"),
    ] {
        let active = app.view == view;
        spans.push(Span::styled(
            format!("{key} "),
            if active { theme::accent() } else { theme::dim() },
        ));
        spans.push(Span::styled(
            format!("{label}  "),
            if active {
                theme::title_focus()
            } else {
                theme::dim()
            },
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{} runs ", app.project.runs.len()),
            theme::dim(),
        )))
        .alignment(ratatui::layout::Alignment::Right),
        area,
    );
}

fn draw_runs(frame: &mut Frame, app: &App, area: Rect) {
    // Narrow terminals keep the columns that matter: trend, then duration,
    // then step yield before the run name ever starves.
    let show_step = area.width >= 70;
    let show_duration = area.width >= 84;
    let show_trend = area.width >= 100;
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
            let selected = app.runs.selected.contains(&row.name);
            let marker = Span::styled(if selected { "▌" } else { " " }, theme::accent());
            let (dot, dot_style) = theme::status(row.status);
            let restarts = if row.restarts > 0 {
                Span::styled(row.restarts.to_string(), theme::accent())
            } else {
                Span::styled("0".to_owned(), theme::dim())
            };
            let mut cells = vec![
                Line::from(vec![marker, Span::raw(row.name.clone())]),
                Line::from(vec![
                    Span::styled(format!("{dot} "), dot_style),
                    Span::raw(row.status),
                ]),
                Line::raw(row.series.to_string()),
                Line::raw(row.points.to_string()),
                Line::from(restarts),
            ];
            if show_step {
                cells.push(Line::raw(
                    row.step.map(|step| step.to_string()).unwrap_or_default(),
                ));
            }
            if show_duration {
                cells.push(Line::from(Span::styled(
                    row.duration.map(fmt_duration).unwrap_or_default(),
                    theme::dim(),
                )));
            }
            if show_trend {
                cells.push(Line::from(Span::styled(
                    row.spark.clone().unwrap_or_default(),
                    theme::dim(),
                )));
            }
            let mut table_row = Row::new(cells);
            if Some(index) == cursor {
                table_row = table_row.style(theme::cursor_row());
            }
            table_row
        })
        .collect();

    let sort_cell = |label: &str| {
        if label == app.runs.sort.label() {
            Line::from(vec![
                Span::styled(label.to_owned(), theme::header()),
                Span::styled(
                    if app.runs.descending { "▼" } else { "▲" },
                    theme::accent(),
                ),
            ])
        } else {
            Line::from(Span::styled(label.to_owned(), theme::dim()))
        }
    };
    let mut header_cells = vec![
        sort_cell("run"),
        Line::from(Span::styled("status", theme::dim())),
        Line::from(Span::styled("series", theme::dim())),
        sort_cell("points"),
        sort_cell("restarts"),
    ];
    let mut constraints = vec![
        Constraint::Fill(2),
        Constraint::Length(9),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(8),
    ];
    if show_step {
        header_cells.push(sort_cell("step"));
        constraints.push(Constraint::Length(8));
    }
    if show_duration {
        header_cells.push(sort_cell("duration"));
        constraints.push(Constraint::Length(8));
    }
    if show_trend {
        header_cells.push(Line::from(Span::styled("trend", theme::dim())));
        constraints.push(Constraint::Length(12));
    }
    let header = Row::new(header_cells);

    let mut title = panel_title("runs", &app.runs.filter, app.runs.editing_filter);
    if let Some(kept) = &app.working_set {
        use std::fmt::Write as _;
        let _ = write!(title, " · keeping {} (U resets)", kept.len());
    }
    let table = Table::new(table_rows, constraints)
        .header(header)
        .block(panel_block(&title, app.runs.editing_filter));
    // Stateful render purely for scrolling: the state is rebuilt from the
    // cursor every frame, so the table follows it below the fold.
    let mut state = TableState::default().with_selected(cursor);
    frame.render_stateful_widget(table, area, &mut state);
}

/// A rounded panel frame: quiet by default, accent-titled while its filter
/// is being edited (the moment the panel has your keyboard).
fn panel_block(title: &str, focus: bool) -> Block<'static> {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(if focus {
            theme::border_focus()
        } else {
            theme::border()
        });
    if title.is_empty() {
        block
    } else {
        block.title(Span::styled(
            format!(" {title} "),
            if focus {
                theme::title_focus()
            } else {
                theme::title()
            },
        ))
    }
}

fn draw_scalars(frame: &mut Frame, app: &App, area: Rect, blank: bool) -> Vec<PixelPanel> {
    let [left, right] =
        Layout::horizontal([Constraint::Length(28), Constraint::Fill(1)]).areas(area);

    let tags = app.visible_tags();
    let current = app.current_tag();
    let items: Vec<ListItem> = tags
        .iter()
        .map(|tag| {
            let style = if Some(tag) == current.as_ref() {
                theme::cursor_row()
            } else {
                Style::default()
            };
            ListItem::new(tag.clone()).style(style)
        })
        .collect();
    let title = panel_title("tags", &app.scalars.filter, app.scalars.editing_filter);
    let mut state = ListState::default()
        .with_selected(tags.iter().position(|tag| Some(tag) == current.as_ref()));
    frame.render_stateful_widget(
        List::new(items).block(panel_block(&title, app.scalars.editing_filter)),
        left,
        &mut state,
    );

    match current {
        Some(tag) => {
            let scope = app.scoped_runs();
            let data = ChartData::for_tag(&app.project, &scope, &tag, &app.chart_options());
            let title = chart_title(app, &tag, &data);
            // Pixel path: reserve the rect with skip cells and hand the
            // panel back for post-frame emission — unless the help overlay
            // needs to paint over this area with ordinary cells.
            if app.graphics.is_some() && !app.help {
                reserve(right, frame.buffer_mut(), blank);
                return vec![PixelPanel::Chart {
                    area: right,
                    data,
                    title,
                    domain: None,
                    bare: false,
                }];
            }
            frame.render_widget(data.plot(&title).widget(), right);
        }
        None => {
            frame.render_widget(
                Paragraph::new(Span::styled(empty_hint(app), theme::dim()))
                    .block(panel_block("", false)),
                right,
            );
        }
    }
    Vec::new()
}

/// Small multiples: one panel per visible tag, scoped runs overlaid, x
/// domain shared across every panel.
fn draw_compare(frame: &mut Frame, app: &App, area: Rect, blank: bool) -> Vec<PixelPanel> {
    let (tags, runs) = app.compare_scope();
    if tags.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(empty_hint(app), theme::dim())).block(panel_block("", false)),
            area,
        );
        return Vec::new();
    }
    // Panels get at least ~24×8 cells; the grid shrinks to what fits.
    let columns = usize::from(area.width / 24).clamp(1, 3);
    let max_rows = usize::from(area.height / 8).max(1);
    let shown = tags.len().min(columns * max_rows);
    let rows = shown.div_ceil(columns);

    let options = app.chart_options();
    let panels: Vec<(String, ChartData)> = tags[..shown]
        .iter()
        .map(|tag| {
            let data = ChartData::for_tag(&app.project, &runs, tag, &options);
            (tag.clone(), data)
        })
        .collect();
    let domain = panels
        .iter()
        .filter_map(|(_, data)| data.x_extent())
        .reduce(|(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)));

    let row_areas = Layout::vertical(vec![Constraint::Fill(1); rows]).split(area);
    let mut cells = Vec::new();
    for row_area in row_areas.iter() {
        cells.extend(
            Layout::horizontal(vec![Constraint::Fill(1); columns])
                .split(*row_area)
                .iter()
                .copied(),
        );
    }

    let pixels = app.graphics.is_some() && !app.help;
    let mut out = Vec::new();
    for ((tag, data), cell) in panels.into_iter().zip(cells) {
        let mut title = tag;
        if out.is_empty()
            && let Some(cut) = tags.len().checked_sub(shown).filter(|&cut| cut > 0)
        {
            use std::fmt::Write as _;
            let _ = write!(title, " (+{cut} more)");
        }
        if pixels {
            reserve(cell, frame.buffer_mut(), blank);
            out.push(PixelPanel::Chart {
                area: cell,
                data,
                title,
                domain,
                bare: true,
            });
        } else {
            frame.render_widget(data.compare_plot(&title, domain).widget(), cell);
        }
    }
    out
}

/// The flat runs × (params + metrics) table with keep/exclude refinement.
fn draw_hparams(frame: &mut Frame, app: &App, area: Rect) {
    let (columns, rows) = app.hparam_table();
    let cursor = app.runs.cursor.as_ref().and_then(|cursor| {
        rows.iter()
            .position(|row| row.first().map(|cell| cell.text()) == Some(cursor.clone()))
    });
    let table_rows: Vec<Row> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            // On screen, floats truncate to significant digits (never
            // rounded); the CSV export keeps full precision.
            let mut cells: Vec<String> = row
                .iter()
                .map(|cell| match cell {
                    crate::tui::app::HparamCell::Number(value) => {
                        crate::table::fmt_sig(*value, 6)
                    }
                    other => other.text(),
                })
                .collect();
            if let Some(first) = cells.first_mut() {
                let marker = if app.runs.selected.contains(&first.clone()) {
                    "▸"
                } else {
                    " "
                };
                *first = format!("{marker}{first}");
            }
            let style = if Some(index) == cursor {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            Row::new(cells).style(style)
        })
        .collect();
    let header = Row::new(columns.clone()).style(Style::default().add_modifier(Modifier::BOLD));
    let mut constraints = vec![Constraint::Fill(2)];
    constraints.extend(
        columns
            .iter()
            .skip(1)
            .map(|column| Constraint::Length((column.len() as u16).clamp(8, 22))),
    );
    let mut title = panel_title("hparams", &app.runs.filter, app.runs.editing_filter);
    if let Some(kept) = &app.working_set {
        use std::fmt::Write as _;
        let _ = write!(title, " · keeping {} (U resets)", kept.len());
    }
    let mut state = TableState::default().with_selected(cursor);
    frame.render_stateful_widget(
        Table::new(table_rows, constraints)
            .header(header)
            .block(panel_block(&title, app.runs.editing_filter)),
        area,
        &mut state,
    );
}

/// Histogram series as a ridgeline over steps.
fn draw_distributions(frame: &mut Frame, app: &App, area: Rect, blank: bool) -> Vec<PixelPanel> {
    let [left, right] =
        Layout::horizontal([Constraint::Length(28), Constraint::Fill(1)]).areas(area);

    let tags = app.histogram_tags();
    let target = app.distribution_target();
    let current_tag = target.as_ref().map(|(_, tag)| tag.clone());
    let items: Vec<ListItem> = tags
        .iter()
        .map(|tag| {
            let style = if Some(tag) == current_tag.as_ref() {
                theme::cursor_row()
            } else {
                Style::default()
            };
            ListItem::new(tag.clone()).style(style)
        })
        .collect();
    let title = panel_title(
        "histograms",
        &app.distributions.filter,
        app.distributions.editing_filter,
    );
    let mut state = ListState::default()
        .with_selected(tags.iter().position(|tag| Some(tag) == current_tag.as_ref()));
    frame.render_stateful_widget(
        List::new(items).block(panel_block(&title, app.distributions.editing_filter)),
        left,
        &mut state,
    );

    let built = target.as_ref().and_then(|(run, tag)| {
        let series = app.project.histogram_series(run, tag)?;
        let data = DistData::build(series, 12)?;
        let mut title = format!("{tag} · {run} · steps {}–{}", data.step_range.0, data.step_range.1);
        if data.drawn < data.total {
            use std::fmt::Write as _;
            let _ = write!(title, " · {} of {} shown", data.drawn, data.total);
        }
        Some((data, title))
    });
    match built {
        Some((data, title)) => {
            if app.graphics.is_some() && !app.help {
                reserve(right, frame.buffer_mut(), blank);
                return vec![PixelPanel::Dist {
                    area: right,
                    data,
                    title,
                }];
            }
            frame.render_widget(data.plot(&title).widget(), right);
        }
        None => {
            frame.render_widget(
                Paragraph::new(Span::styled(empty_hint(app), theme::dim()))
                    .block(panel_block("", false)),
                right,
            );
        }
    }
    Vec::new()
}

/// Fills `area` with skip-flagged cells so the frame diff never paints over
/// the image region (§5.6: reserve, don't render). In `blank` mode the
/// cells are ordinary spaces instead — drawn for real, clearing whatever a
/// previous view left there before a transparent image lands on top.
fn reserve(area: Rect, buffer: &mut ratatui::buffer::Buffer, blank: bool) {
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.reset();
                cell.set_symbol(" ");
                if !blank {
                    cell.set_diff_option(ratatui::buffer::CellDiffOption::Skip);
                }
            }
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
    if app.scalars.show_ghosts {
        title.push_str(" · ghosts");
    }
    match app.scalars.x_axis {
        crate::chart::XAxis::Step => {}
        crate::chart::XAxis::Wall => title.push_str(" · wall"),
        crate::chart::XAxis::Relative => title.push_str(" · relative"),
        crate::chart::XAxis::Tokens => title.push_str(" · tokens"),
    }
    if data.runs_without_counter > 0 {
        let _ = write!(title, " · {} runs lack a counter", data.runs_without_counter);
    }
    if data.tokens_dropped > 0 {
        let _ = write!(title, " · {} pts outside counter", data.tokens_dropped);
    }
    if app.scalars.x_window.is_some() {
        title.push_str(" · zoom (0 fits)");
    }
    // The crosshair readout: exact values (truncated display, never
    // rounded) for every overlaid series at the nearest point.
    if let Some(x) = app.scalars.crosshair {
        let readings = data.values_at(x);
        if let Some((_, exact_x, _)) = readings.first() {
            let _ = write!(title, " ┊ @{}", crate::table::fmt_sig(*exact_x, 6));
            for (label, _, value) in readings.iter().take(3) {
                if readings.len() == 1 {
                    let _ = write!(title, " = {}", crate::table::fmt_sig(*value, 6));
                } else {
                    let _ = write!(title, "  {} {}", label, crate::table::fmt_sig(*value, 6));
                }
            }
            if readings.len() > 3 {
                let _ = write!(title, "  +{}", readings.len() - 3);
            }
        }
    }
    title
}

/// The empty-state line: says *why* nothing is drawn when filters, the
/// working set, or a selection narrow the scope, instead of a bare
/// "no match".
fn empty_hint(app: &App) -> String {
    let mut reasons = Vec::new();
    if !app.runs.filter.is_empty() {
        reasons.push("the runs filter");
    }
    if app.working_set.is_some() {
        reasons.push("the working set (U resets)");
    }
    if !app.runs.selected.is_empty() {
        reasons.push("the selection (Esc in runs clears)");
    }
    if reasons.is_empty() {
        "no matching series".to_owned()
    } else {
        format!(
            "no matching series — {} may be hiding runs",
            reasons.join(", ")
        )
    }
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

/// Per-view key hints for the footer: `(key, what it does)`.
fn hints(view: View) -> &'static [(&'static str, &'static str)] {
    match view {
        View::Runs => &[
            ("enter", "open"),
            ("space", "pick"),
            ("/", "filter"),
            ("s", "sort"),
            ("K/X", "keep/drop"),
            ("e", "export"),
            ("?", "help"),
        ],
        View::Scalars => &[
            ("j/k", "tag"),
            ("←→", "scan"),
            ("+/-", "zoom"),
            ("[]", "pan"),
            ("s/S", "smooth"),
            ("L", "log"),
            ("x", "axis"),
            ("v", "ghosts"),
            ("?", "help"),
        ],
        View::Compare => &[
            ("/", "tags"),
            ("s/S", "smooth"),
            ("x", "axis"),
            ("esc", "back"),
            ("?", "help"),
        ],
        View::Hparams => &[
            ("space", "pick"),
            ("K/X", "keep/drop"),
            ("U", "reset"),
            ("e", "export"),
            ("?", "help"),
        ],
        View::Distributions => &[
            ("j/k", "tag"),
            ("/", "filter"),
            ("e", "export"),
            ("?", "help"),
        ],
    }
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    // The state must always be visible: it renders right-aligned last over
    // the hints, so nothing may ever push "paused" off the screen.
    let (state, state_style) = if app.paused {
        ("‖ paused".to_owned(), theme::dim())
    } else {
        let quiet = app.last_change.elapsed();
        if quiet > 2 * app.interval {
            (
                format!("○ stale {}s", quiet.as_secs()),
                Style::default().fg(theme::STALE),
            )
        } else {
            ("● live".to_owned(), Style::default().fg(theme::LIVE))
        }
    };
    let mut right_spans = Vec::new();
    if app.project.dropped_records > 0 {
        right_spans.push(Span::styled(
            format!("{} lost  ", app.project.dropped_records),
            Style::default().fg(theme::STALE),
        ));
    }
    if app.project.dead_files > 0 {
        right_spans.push(Span::styled(
            format!("{} dead  ", app.project.dead_files),
            theme::accent(),
        ));
    }
    right_spans.push(Span::styled(state, state_style));
    right_spans.push(Span::raw(" "));

    let left = match &app.message {
        Some(message) => Line::from(Span::styled(message.clone(), theme::accent())),
        None => {
            let mut spans = vec![Span::raw(" ")];
            for (key, what) in hints(app.view) {
                spans.push(Span::styled(*key, theme::accent()));
                spans.push(Span::styled(format!(" {what}   "), theme::dim()));
            }
            Line::from(spans)
        }
    };
    frame.render_widget(Paragraph::new(left), area);
    frame.render_widget(
        Paragraph::new(Line::from(right_spans)).alignment(ratatui::layout::Alignment::Right),
        area,
    );
}

fn draw_help(frame: &mut Frame, area: Rect) {
    const KEYS: &[(&str, &str)] = &[
        ("q", "quit"),
        ("?", "this help — any key closes"),
        ("Tab · 1-5", "views: runs, scalars, compare, hparams, dists"),
        ("/", "filter · a predicate works: lr > 1e-3 and status == active"),
        ("Esc", "progressively: editor, selection, filter, back"),
        ("K / X / U", "keep / exclude selection as working set / reset"),
        ("e", "export this view as CSV, next to your shell"),
        ("p · r", "pause polling · refresh now"),
        ("", ""),
        ("j k g G", "move · space picks runs for overlay · Enter opens"),
        ("s / S", "runs: sort column, reverse · scalars: smoothing -/+"),
        ("L · x · v", "log-y · x axis: step, wall, relative, tokens · ghosts"),
        ("← →", "crosshair point by point, with exact values in the title"),
        ("+ - [ ] 0", "zoom around the crosshair · pan · fit everything"),
    ];
    let mut lines = vec![Line::raw("")];
    for (key, what) in KEYS {
        lines.push(Line::from(vec![
            Span::styled(format!("  {key:>10}  "), theme::accent()),
            Span::styled((*what).to_owned(), Style::default()),
        ]));
    }
    let height = lines.len() as u16 + 3;
    let width = 76.min(area.width);
    let popup = Rect {
        x: area.width.saturating_sub(width) / 2,
        y: area.height.saturating_sub(height) / 2,
        width,
        height: height.min(area.height),
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(panel_block("keys", true)), popup);
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
        let mut adam: Vec<Vec<u8>> = (0..10)
            .flat_map(|step| {
                [
                    scalar_event(wall(step), step, "train/loss", 8.0 / (step + 1) as f32),
                    scalar_event(wall(step), step, "train/acc", 0.1 * step as f32),
                ]
            })
            .collect();
        for step in 0..3 {
            adam.push(tfevents::writer::histogram_event(
                wall(step),
                step,
                "params/w",
                &[(-1.0, 0.0, 3.0 + step as f64), (0.0, 1.0, 5.0)],
            ));
        }
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
        terminal
            .draw(|frame| {
                draw(frame, app, false);
            })
            .unwrap();
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
            "▌ vertov   1 runs  2 scalars  3 compare  4 hparams  5 dists      2 runs",
            "╭ runs ────────────────────────────────────────────────────────────────╮",
            "│run▲                       status    series points   restarts step    │",
            "│ adam                      ● active  3      23       0        9       │",
            "│▌sgd                       ● active  1      8        0        7       │",
            "│                                                                      │",
            "╰──────────────────────────────────────────────────────────────────────╯",
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
    fn compare_view_draws_small_multiples() {
        let (mut app, _guard) = test_app("compare");
        app.view = View::Compare;
        app.ensure_materialized().unwrap();
        let text = snapshot(&app, 80, 20).join("\n");
        // One panel per scalar tag, titled by tag; histograms stay out.
        assert!(text.contains("train/acc"), "{text}");
        assert!(text.contains("train/loss"), "{text}");
        assert!(!text.contains("params/w"), "{text}");
    }

    #[test]
    fn hparams_view_lists_params_and_metrics() {
        let (mut app, _guard) = test_app("hparams-view");
        app.view = View::Hparams;
        app.project
            .runs
            .get_mut("adam")
            .unwrap()
            .hparams
            .insert("lr".to_owned(), vertov_model::HparamValue::F64(0.01));
        let (columns, rows) = app.hparam_table();
        assert_eq!(columns, vec!["run", "lr", "train/acc", "train/loss"]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][1], crate::tui::app::HparamCell::Number(0.01));
        assert_eq!(rows[1][1], crate::tui::app::HparamCell::Empty);
        let text = snapshot(&app, 72, 8).join("\n");
        assert!(text.contains("lr"), "{text}");
        assert!(text.contains("adam"), "{text}");
    }

    #[test]
    fn distributions_view_draws_ridgeline() {
        let (mut app, _guard) = test_app("distributions");
        app.view = View::Distributions;
        app.ensure_materialized().unwrap();
        let text = snapshot(&app, 78, 16).join("\n");
        assert!(text.contains("params/w"), "tag list: {text}");
        assert!(text.contains("steps 0–2"), "title: {text}");
        assert!(text.contains("adam"), "run in title: {text}");
    }

    #[test]
    fn enter_replaces_a_selection_the_cursor_left() {
        use crossterm::event::{KeyCode, KeyEvent};
        let (mut app, _guard) = test_app("enter-scope");
        app.runs.cursor = Some("adam".to_owned());
        app.update(KeyEvent::from(KeyCode::Enter));
        assert_eq!(app.scoped_runs(), vec!["adam"]);
        assert_eq!(app.view, View::Scalars);
        // Point somewhere new: the old scope is over.
        app.update(KeyEvent::from(KeyCode::Esc));
        app.runs.cursor = Some("sgd".to_owned());
        app.update(KeyEvent::from(KeyCode::Enter));
        assert_eq!(app.scoped_runs(), vec!["sgd"]);
    }

    #[test]
    fn escape_clears_selection_then_filter_then_leaves_scalars() {
        use crossterm::event::{KeyCode, KeyEvent};
        let (mut app, _guard) = test_app("esc-progressive");
        app.runs.selected.insert("adam".to_owned());
        app.runs.filter = "ad".to_owned();
        app.update(KeyEvent::from(KeyCode::Esc));
        assert!(app.runs.selected.is_empty());
        assert_eq!(app.runs.filter, "ad");
        app.update(KeyEvent::from(KeyCode::Esc));
        assert!(app.runs.filter.is_empty());

        app.view = View::Scalars;
        app.scalars.filter = "loss".to_owned();
        app.update(KeyEvent::from(KeyCode::Esc));
        assert!(app.scalars.filter.is_empty());
        assert_eq!(app.view, View::Scalars);
        app.update(KeyEvent::from(KeyCode::Esc));
        assert_eq!(app.view, View::Runs);
    }

    #[test]
    fn smoothing_snaps_to_exact_zero() {
        use crossterm::event::{KeyCode, KeyEvent};
        let (mut app, _guard) = test_app("smooth-snap");
        app.view = View::Scalars;
        for _ in 0..3 {
            app.update(KeyEvent::from(KeyCode::Char('S')));
        }
        assert!((app.scalars.smooth - 0.3).abs() < 1e-12);
        for _ in 0..3 {
            app.update(KeyEvent::from(KeyCode::Char('s')));
        }
        assert_eq!(app.scalars.smooth, 0.0, "must be exactly zero");
    }

    #[test]
    fn alt_modified_chars_do_not_enter_filters() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let (mut app, _guard) = test_app("alt-chars");
        app.update(KeyEvent::from(KeyCode::Char('/')));
        assert!(app.runs.editing_filter);
        app.update(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::ALT));
        assert_eq!(app.runs.filter, "", "Esc+g over ssh must not type a g");
        app.update(KeyEvent::from(KeyCode::Char('g')));
        assert_eq!(app.runs.filter, "g");
    }

    #[test]
    fn empty_hint_names_the_narrowing_state() {
        let (mut app, _guard) = test_app("empty-hint");
        assert_eq!(empty_hint(&app), "no matching series");
        app.runs.filter = "zzz".to_owned();
        app.working_set = Some(Default::default());
        let hint = empty_hint(&app);
        assert!(hint.contains("runs filter"), "{hint}");
        assert!(hint.contains("working set"), "{hint}");
    }

    #[test]
    fn crosshair_steps_points_and_reads_exact_values() {
        use crossterm::event::{KeyCode, KeyEvent};
        let (mut app, _guard) = test_app("crosshair");
        app.view = View::Scalars;
        app.scalars.cursor = Some("train/loss".to_owned());
        app.ensure_materialized().unwrap();

        // First → lands on the newest point of the first series (adam,
        // steps 0..=9); two ← steps walk back to 7.
        app.update(KeyEvent::from(KeyCode::Right));
        assert_eq!(app.scalars.crosshair, Some(9.0));
        app.update(KeyEvent::from(KeyCode::Left));
        app.update(KeyEvent::from(KeyCode::Left));
        assert_eq!(app.scalars.crosshair, Some(7.0));

        // The readout carries every overlaid series' exact value there.
        let chart = app.scalars_chart().unwrap();
        let readings = chart.values_at(7.0);
        assert_eq!(readings.len(), 2);
        assert_eq!(readings[0].2, 1.0, "adam: 8/(7+1)");
        assert_eq!(readings[1].2, 2.0, "sgd: 9-7");
        // And the title mentions the position.
        let title = chart_title(&app, "train/loss", &chart);
        assert!(title.contains("@7"), "{title}");
    }

    #[test]
    fn zoom_pan_and_reset() {
        use crossterm::event::{KeyCode, KeyEvent};
        let (mut app, _guard) = test_app("zoom");
        app.view = View::Scalars;
        app.scalars.cursor = Some("train/loss".to_owned());
        app.ensure_materialized().unwrap();

        // Zoom in around the center of the full 0..=9 extent.
        app.update(KeyEvent::from(KeyCode::Char('+')));
        let (from, to) = app.scalars.x_window.unwrap();
        assert!((to - from - 6.0).abs() < 1e-9, "width 9/1.5, got {from}..{to}");
        // Pan right, clamped to the extent.
        for _ in 0..10 {
            app.update(KeyEvent::from(KeyCode::Char(']')));
        }
        let (from, to) = app.scalars.x_window.unwrap();
        assert_eq!(to, 9.0);
        assert!((from - 3.0).abs() < 1e-9);
        // Zoom out far enough and the window clears to full extent.
        app.update(KeyEvent::from(KeyCode::Char('-')));
        app.update(KeyEvent::from(KeyCode::Char('-')));
        assert_eq!(app.scalars.x_window, None);

        // The window resets when the axis's units change.
        app.update(KeyEvent::from(KeyCode::Char('+')));
        assert!(app.scalars.x_window.is_some());
        app.update(KeyEvent::from(KeyCode::Char('x')));
        assert_eq!(app.scalars.x_window, None);
        assert_eq!(app.scalars.crosshair, None);
    }

    #[test]
    fn crosshair_walks_the_window_along() {
        use crossterm::event::{KeyCode, KeyEvent};
        let (mut app, _guard) = test_app("crosshair-pan");
        app.view = View::Scalars;
        app.scalars.cursor = Some("train/loss".to_owned());
        app.ensure_materialized().unwrap();
        // Crosshair at the newest point, then zoom in tight around it.
        app.update(KeyEvent::from(KeyCode::Right));
        app.update(KeyEvent::from(KeyCode::Char('+')));
        app.update(KeyEvent::from(KeyCode::Char('+')));
        let (from, _) = app.scalars.x_window.unwrap();
        assert!(from > 0.0);
        // Walking left past the edge drags the window with it.
        for _ in 0..9 {
            app.update(KeyEvent::from(KeyCode::Left));
        }
        assert_eq!(app.scalars.crosshair, Some(0.0));
        let (from, _) = app.scalars.x_window.unwrap();
        assert_eq!(from, 0.0);
    }

    #[test]
    fn keep_and_exclude_refine_the_working_set() {
        use crossterm::event::{KeyCode, KeyEvent};
        let (mut app, _guard) = test_app("working-set");
        // Keep the cursor run.
        app.runs.cursor = Some("sgd".to_owned());
        app.update(KeyEvent::from(KeyCode::Char('K')));
        assert_eq!(
            app.run_rows().iter().map(|row| row.name.clone()).collect::<Vec<_>>(),
            vec!["sgd"]
        );
        assert_eq!(app.scoped_runs(), vec!["sgd"]);
        // Reset, then exclude it instead.
        app.update(KeyEvent::from(KeyCode::Char('U')));
        assert_eq!(app.run_rows().len(), 2);
        app.update(KeyEvent::from(KeyCode::Char('X')));
        assert_eq!(
            app.run_rows().iter().map(|row| row.name.clone()).collect::<Vec<_>>(),
            vec!["adam"]
        );
    }

    #[test]
    fn pixel_mode_reserves_the_panel_and_block_carries_the_image() {
        use malevich::pixel::{Graphics, Protocol};
        let (mut app, _guard) = test_app("pixel-reserve");
        app.view = View::Scalars;
        app.scalars.cursor = Some("train/loss".to_owned());
        app.graphics = Some(Graphics::new(Protocol::Kitty));
        app.ensure_materialized().unwrap();

        let mut terminal = Terminal::new(TestBackend::new(72, 14)).unwrap();
        let mut panels = Vec::new();
        terminal.draw(|frame| panels = draw(frame, &app, false)).unwrap();
        assert_eq!(panels.len(), 1);
        let panel = &panels[0];
        let area = panel.area();
        assert_eq!(area.x, 28);

        // Reserve, don't render: the chart rect stays blank in the cell
        // buffer — the image is emitted after the frame, not through it.
        let buffer = terminal.backend().buffer();
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                assert_eq!(buffer.cell((x, y)).unwrap().symbol(), " ", "at {x},{y}");
            }
        }

        // The post-frame block really is a kitty graphics transmission.
        let frame = malevich::Frame {
            width: area.width as usize,
            height: area.height as usize,
            charset: malevich::Charset::Quadrants,
            color: malevich::ColorMode::TrueColor,
            theme: malevich::Theme::DARK,
        };
        let block = panel.plot().render_pixels_at(
            &frame,
            &Graphics::new(Protocol::Kitty),
            area.x as usize,
        );
        assert!(block.contains("\x1b_G"), "kitty APC missing");
        assert!(block.contains("train/loss"), "chrome title missing");

        // Help overlay wins over the image: no panels while it is open.
        app.help = true;
        let mut panels = vec![];
        terminal.draw(|frame| panels = draw(frame, &app, false)).unwrap();
        assert!(panels.is_empty());
    }

    #[test]
    fn help_overlay_renders() {
        let (mut app, _guard) = test_app("help");
        app.help = true;
        let text = snapshot(&app, 72, 20).join("\n");
        assert!(text.contains("keys"), "{text}");
        assert!(text.contains("views: runs, scalars"), "{text}");
    }
}
