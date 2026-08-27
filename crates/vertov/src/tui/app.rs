//! TUI state and its transitions — everything a keypress can change, kept
//! apart from drawing so both are testable without a terminal.
//!
//! The cardinal rule (guild.ai's canonical paper cut, inverted): a refresh
//! never loses state. Cursors are run/tag *names*, not indices; selection,
//! sort, filters, smoothing, and axes all survive any amount of new data.

use std::collections::BTreeSet;
use std::time::{Duration, Instant, SystemTime};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use vertov_model::{Project, SeriesClass};

use crate::chart::{ChartOptions, XAxis};

/// Which screen is showing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    Runs,
    Scalars,
    Compare,
    Hparams,
    Distributions,
}

impl View {
    pub fn next(self) -> View {
        match self {
            View::Runs => View::Scalars,
            View::Scalars => View::Compare,
            View::Compare => View::Hparams,
            View::Hparams => View::Distributions,
            View::Distributions => View::Runs,
        }
    }
}

/// Sort column of the runs table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunsSort {
    Name,
    Points,
    Restarts,
    Step,
    Duration,
}

impl RunsSort {
    pub fn next(self) -> RunsSort {
        match self {
            RunsSort::Name => RunsSort::Points,
            RunsSort::Points => RunsSort::Restarts,
            RunsSort::Restarts => RunsSort::Step,
            RunsSort::Step => RunsSort::Duration,
            RunsSort::Duration => RunsSort::Name,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RunsSort::Name => "run",
            RunsSort::Points => "points",
            RunsSort::Restarts => "restarts",
            RunsSort::Step => "step",
            RunsSort::Duration => "duration",
        }
    }
}

/// A cell of the hparams table — one small value enum so drawing and CSV
/// export share the same rows.
#[derive(Clone, PartialEq, Debug)]
pub enum HparamCell {
    Text(String),
    Number(f64),
    Empty,
}

impl HparamCell {
    pub fn text(&self) -> String {
        match self {
            HparamCell::Text(text) => text.clone(),
            HparamCell::Number(number) => number.to_string(),
            HparamCell::Empty => String::new(),
        }
    }
}

/// One row of the runs table, precomputed for sorting and drawing.
pub struct RunRow {
    pub name: String,
    pub status: &'static str,
    pub series: usize,
    pub points: u64,
    pub restarts: u64,
    pub step: Option<i64>,
    pub duration: Option<f64>,
}

pub struct RunsView {
    /// Run name under the cursor — a name, so new runs can't steal it.
    pub cursor: Option<String>,
    /// Runs picked (space) for overlay in Scalars.
    pub selected: BTreeSet<String>,
    pub filter: String,
    pub editing_filter: bool,
    pub sort: RunsSort,
    pub descending: bool,
}

pub struct ScalarsView {
    /// Tag under the cursor — a name, same reason.
    pub cursor: Option<String>,
    pub filter: String,
    pub editing_filter: bool,
    /// EWMA factor; 0 disables.
    pub smooth: f64,
    pub x_axis: XAxis,
    pub log_y: bool,
    /// Draw preempted ghost tails as faded lines.
    pub show_ghosts: bool,
}

pub struct DistributionsView {
    /// Histogram tag under the cursor.
    pub cursor: Option<String>,
    pub filter: String,
    pub editing_filter: bool,
}

/// What the event loop should do after a keypress.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Continue,
    Quit,
}

pub struct App {
    pub project: Project,
    pub view: View,
    pub runs: RunsView,
    pub scalars: ScalarsView,
    pub distributions: DistributionsView,
    /// HiPlot's best idea: a progressively refined working set of runs.
    /// `K` keeps the selection (or cursor), `X` excludes it, `U` resets.
    /// `None` means every run. Applied before the filter bar everywhere.
    pub working_set: Option<BTreeSet<String>>,
    pub paused: bool,
    pub help: bool,
    pub interval: Duration,
    pub last_change: Instant,
    /// Transient one-line notice (export path, errors), cleared on keypress.
    pub message: Option<String>,
    pub force_refresh: bool,
    /// Pixel graphics for chart panels, when the terminal offered a
    /// protocol (probed once, before raw mode). `None` means cell glyphs —
    /// the honest fallback rung, not an error.
    pub graphics: Option<malevich::pixel::Graphics>,
}

impl App {
    pub fn new(project: Project, interval: Duration, tag_filter: &str) -> App {
        App {
            project,
            view: View::Runs,
            runs: RunsView {
                cursor: None,
                selected: BTreeSet::new(),
                filter: String::new(),
                editing_filter: false,
                sort: RunsSort::Name,
                descending: false,
            },
            scalars: ScalarsView {
                cursor: None,
                filter: tag_filter.to_owned(),
                editing_filter: false,
                smooth: 0.0,
                x_axis: XAxis::Step,
                log_y: false,
                show_ghosts: false,
            },
            distributions: DistributionsView {
                cursor: None,
                filter: String::new(),
                editing_filter: false,
            },
            working_set: None,
            paused: false,
            help: false,
            interval,
            last_change: Instant::now(),
            message: None,
            force_refresh: false,
            graphics: None,
        }
    }

    /// The runs filter as a predicate when it parses as one; a plain
    /// substring match on the run name otherwise (so typing stays forgiving
    /// mid-expression).
    fn parsed_filter(&self) -> Option<vertov_model::Predicate> {
        if self.runs.filter.is_empty() {
            return None;
        }
        vertov_model::Predicate::parse(&self.runs.filter).ok()
    }

    fn filter_passes(
        &self,
        predicate: Option<&vertov_model::Predicate>,
        name: &str,
        status: &str,
        run: &vertov_model::Run,
    ) -> bool {
        if self.runs.filter.is_empty() {
            return true;
        }
        match predicate {
            Some(predicate) => predicate.matches(name, status, run),
            None => name.contains(&self.runs.filter),
        }
    }

    /// Whether the working set (keep/exclude refinement) admits a run.
    fn in_working_set(&self, name: &str) -> bool {
        self.working_set
            .as_ref()
            .is_none_or(|kept| kept.contains(name))
    }

    /// The runs table under working set, filter, and sort. Pure.
    pub fn run_rows(&self) -> Vec<RunRow> {
        let now = SystemTime::now();
        let window = Duration::from_secs(60).max(2 * self.interval);
        let predicate = self.parsed_filter();
        let mut rows: Vec<RunRow> = self
            .project
            .runs
            .iter()
            .filter(|(name, run)| {
                let status = crate::status_text(run, now, window);
                self.in_working_set(name)
                    && self.filter_passes(predicate.as_ref(), name, status, run)
            })
            .map(|(name, run)| RunRow {
                name: name.clone(),
                status: crate::status_text(run, now, window),
                series: run.series.len(),
                points: run.series.values().map(|series| series.summary.count()).sum(),
                restarts: run.preemptions,
                step: run
                    .series
                    .values()
                    .filter_map(|series| series.summary.last().map(|point| point.step))
                    .max(),
                duration: match (run.first_wall, run.last_wall) {
                    (Some(first), Some(last)) if last >= first => Some(last - first),
                    _ => None,
                },
            })
            .collect();
        rows.sort_by(|a, b| {
            let ordering = match self.runs.sort {
                RunsSort::Name => a.name.cmp(&b.name),
                RunsSort::Points => a.points.cmp(&b.points),
                RunsSort::Restarts => a.restarts.cmp(&b.restarts),
                RunsSort::Step => a.step.cmp(&b.step),
                RunsSort::Duration => a.duration.partial_cmp(&b.duration).unwrap_or(std::cmp::Ordering::Equal),
            };
            let ordering = ordering.then_with(|| a.name.cmp(&b.name));
            if self.runs.descending {
                ordering.reverse()
            } else {
                ordering
            }
        });
        rows
    }

    /// Columns (`run` + params + metrics) and rows of the Hparams view —
    /// the flat runs × (params + metrics) table, scoped and sorted like the
    /// runs table. Pure; the CSV export shares it.
    pub fn hparam_table(&self) -> (Vec<String>, Vec<Vec<HparamCell>>) {
        use vertov_model::HparamValue;
        let scope = self.run_rows();
        let mut param_keys = BTreeSet::new();
        let mut metric_tags = BTreeSet::new();
        for row in &scope {
            let Some(run) = self.project.runs.get(&row.name) else {
                continue;
            };
            param_keys.extend(run.hparams.keys().cloned());
            metric_tags.extend(
                run.series
                    .iter()
                    .filter(|(_, series)| series.class == SeriesClass::Scalar)
                    .map(|(tag, _)| tag.clone()),
            );
        }
        let mut columns = vec!["run".to_owned()];
        columns.extend(param_keys.iter().cloned());
        columns.extend(metric_tags.iter().cloned());

        let rows = scope
            .iter()
            .filter_map(|scope_row| {
                let run = self.project.runs.get(&scope_row.name)?;
                let mut row = vec![HparamCell::Text(scope_row.name.clone())];
                for key in &param_keys {
                    row.push(match run.hparams.get(key) {
                        Some(HparamValue::F64(value)) => HparamCell::Number(*value),
                        Some(HparamValue::String(value)) => HparamCell::Text(value.clone()),
                        Some(HparamValue::Bool(value)) => HparamCell::Text(value.to_string()),
                        None => HparamCell::Empty,
                    });
                }
                for tag in &metric_tags {
                    row.push(
                        run.series
                            .get(tag)
                            .and_then(|series| series.summary.last())
                            .map_or(HparamCell::Empty, |point| HparamCell::Number(point.value)),
                    );
                }
                Some(row)
            })
            .collect();
        (columns, rows)
    }

    /// Scalar tags visible in the Scalars view: union over scoped runs,
    /// filtered by fuzzy subsequence match. Pure.
    pub fn visible_tags(&self) -> Vec<String> {
        let scope = self.scoped_runs();
        let mut tags = BTreeSet::new();
        for name in &scope {
            let Some(run) = self.project.runs.get(name) else {
                continue;
            };
            for (tag, series) in &run.series {
                if series.class == SeriesClass::Scalar
                    && fuzzy_match(tag, &self.scalars.filter)
                {
                    tags.insert(tag.clone());
                }
            }
        }
        tags.into_iter().collect()
    }

    /// The runs the chart views draw: the selection when non-empty, else
    /// every run passing the working set and the runs filter.
    pub fn scoped_runs(&self) -> Vec<String> {
        if !self.runs.selected.is_empty() {
            return self
                .runs
                .selected
                .iter()
                .filter(|name| self.in_working_set(name))
                .cloned()
                .collect();
        }
        let now = SystemTime::now();
        let window = Duration::from_secs(60).max(2 * self.interval);
        let predicate = self.parsed_filter();
        self.project
            .runs
            .iter()
            .filter(|(name, run)| {
                let status = crate::status_text(run, now, window);
                self.in_working_set(name)
                    && self.filter_passes(predicate.as_ref(), name, status, run)
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Histogram-class tags over the scoped runs, fuzzy-filtered — the
    /// Distributions view's list. Pure.
    pub fn histogram_tags(&self) -> Vec<String> {
        let mut tags = BTreeSet::new();
        for name in self.scoped_runs() {
            let Some(run) = self.project.runs.get(&name) else {
                continue;
            };
            for (tag, series) in &run.series {
                if series.class == SeriesClass::Histogram
                    && fuzzy_match(tag, &self.distributions.filter)
                {
                    tags.insert(tag.clone());
                }
            }
        }
        tags.into_iter().collect()
    }

    /// The `(run, tag)` the Distributions view shows: the cursor tag when
    /// still visible (else the first), in the cursor run when it has the
    /// tag (else the first scoped run that does).
    pub fn distribution_target(&self) -> Option<(String, String)> {
        let tags = self.histogram_tags();
        let tag = match &self.distributions.cursor {
            Some(cursor) if tags.contains(cursor) => cursor.clone(),
            _ => tags.first().cloned()?,
        };
        let has_tag = |name: &String| {
            self.project
                .runs
                .get(name)
                .is_some_and(|run| run.series.contains_key(&tag))
        };
        let scope = self.scoped_runs();
        let run = self
            .runs
            .cursor
            .clone()
            .filter(|cursor| scope.contains(cursor) && has_tag(cursor))
            .or_else(|| scope.iter().find(|name| has_tag(name)).cloned())?;
        Some((run, tag))
    }

    /// The tag the chart draws: the cursor if it still exists, else the
    /// first visible tag.
    pub fn current_tag(&self) -> Option<String> {
        let tags = self.visible_tags();
        if let Some(cursor) = &self.scalars.cursor
            && tags.contains(cursor)
        {
            return Some(cursor.clone());
        }
        tags.first().cloned()
    }

    pub fn chart_options(&self) -> ChartOptions {
        ChartOptions {
            x_axis: self.scalars.x_axis,
            smooth: (self.scalars.smooth > 0.0).then_some(self.scalars.smooth),
            runs_filter: None,
            log_y: self.scalars.log_y,
            show_ghosts: self.scalars.show_ghosts,
            tokens_tag: None,
        }
    }

    /// Tags the Compare grid panels (visible scalar tags, capped) and the
    /// runs each panel overlays (scoped, capped so the whole grid stays
    /// inside the materialization budget).
    pub fn compare_scope(&self) -> (Vec<String>, Vec<String>) {
        let mut tags = self.visible_tags();
        tags.truncate(12);
        let mut runs = self.scoped_runs();
        runs.truncate(4);
        (tags, runs)
    }

    /// Materializes what the current view is about to draw. The one place
    /// the UI cycle mutates the project outside refresh.
    pub fn ensure_materialized(&mut self) -> std::io::Result<()> {
        match self.view {
            View::Scalars => {
                let Some(tag) = self.current_tag() else {
                    return Ok(());
                };
                for run in self.scoped_runs() {
                    self.materialize_with_counter(&run, &tag)?;
                }
            }
            View::Compare => {
                let (tags, runs) = self.compare_scope();
                for tag in &tags {
                    for run in &runs {
                        self.materialize_with_counter(run, tag)?;
                    }
                }
            }
            View::Distributions => {
                if let Some((run, tag)) = self.distribution_target() {
                    self.project.materialize_histograms(&run, &tag)?;
                }
            }
            View::Runs | View::Hparams => {}
        }
        Ok(())
    }

    /// Materializes a series and, when the tokens axis is active, its run's
    /// token counter too.
    fn materialize_with_counter(&mut self, run: &str, tag: &str) -> std::io::Result<()> {
        self.project.materialize(run, tag)?;
        if self.scalars.x_axis == XAxis::Tokens
            && let Some(counter_tag) = self
                .project
                .runs
                .get(run)
                .and_then(|r| r.token_counter(None))
        {
            self.project.materialize(run, &counter_tag)?;
        }
        Ok(())
    }

    /// Applies one keypress. Pure state transition (plus export I/O).
    pub fn update(&mut self, key: KeyEvent) -> Action {
        self.message = None;
        if self.help {
            self.help = false;
            return Action::Continue;
        }
        // Filter editing captures almost everything.
        let editing = match self.view {
            View::Runs | View::Hparams => self.runs.editing_filter,
            View::Scalars | View::Compare => self.scalars.editing_filter,
            View::Distributions => self.distributions.editing_filter,
        };
        if editing {
            match key.code {
                KeyCode::Esc => {
                    self.filter_mut().clear();
                    self.set_editing(false);
                }
                KeyCode::Enter => self.set_editing(false),
                KeyCode::Backspace => {
                    self.filter_mut().pop();
                }
                KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.filter_mut().push(ch);
                }
                _ => {}
            }
            return Action::Continue;
        }

        match key.code {
            KeyCode::Char('q') => return Action::Quit,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char('p') => self.paused = !self.paused,
            KeyCode::Char('r') => self.force_refresh = true,
            KeyCode::Char('e') => self.export(),
            KeyCode::Char('/') => self.set_editing(true),
            KeyCode::Tab => self.view = self.view.next(),
            KeyCode::Char('1') => self.view = View::Runs,
            KeyCode::Char('2') => self.view = View::Scalars,
            KeyCode::Char('3') => self.view = View::Compare,
            KeyCode::Char('4') => self.view = View::Hparams,
            KeyCode::Char('5') => self.view = View::Distributions,
            KeyCode::Char('K') => self.keep(),
            KeyCode::Char('X') => self.exclude(),
            KeyCode::Char('U') => self.working_set = None,
            _ => match self.view {
                View::Runs | View::Hparams => self.update_runs(key),
                View::Scalars | View::Compare => self.update_scalars(key),
                View::Distributions => self.update_distributions(key),
            },
        }
        Action::Continue
    }

    /// `K`: the working set becomes the selection (or the cursor run) —
    /// HiPlot's progressive refinement, keep half.
    fn keep(&mut self) {
        let kept: BTreeSet<String> = if self.runs.selected.is_empty() {
            self.runs.cursor.iter().cloned().collect()
        } else {
            self.runs.selected.clone()
        };
        if !kept.is_empty() {
            self.working_set = Some(kept);
            self.runs.selected.clear();
        }
    }

    /// `X`: removes the selection (or the cursor run) from the working set.
    fn exclude(&mut self) {
        let excluded: BTreeSet<String> = if self.runs.selected.is_empty() {
            self.runs.cursor.iter().cloned().collect()
        } else {
            self.runs.selected.clone()
        };
        if excluded.is_empty() {
            return;
        }
        let remaining: BTreeSet<String> = self
            .project
            .runs
            .keys()
            .filter(|name| self.in_working_set(name) && !excluded.contains(*name))
            .cloned()
            .collect();
        self.working_set = Some(remaining);
        self.runs.selected.clear();
    }

    fn set_editing(&mut self, editing: bool) {
        match self.view {
            View::Runs | View::Hparams => self.runs.editing_filter = editing,
            View::Scalars | View::Compare => self.scalars.editing_filter = editing,
            View::Distributions => self.distributions.editing_filter = editing,
        }
    }

    fn filter_mut(&mut self) -> &mut String {
        match self.view {
            View::Runs | View::Hparams => &mut self.runs.filter,
            View::Scalars | View::Compare => &mut self.scalars.filter,
            View::Distributions => &mut self.distributions.filter,
        }
    }

    fn update_distributions(&mut self, key: KeyEvent) {
        let tags = self.histogram_tags();
        let position = self
            .distributions
            .cursor
            .as_ref()
            .and_then(|cursor| tags.iter().position(|tag| tag == cursor));
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                let next = position.map_or(0, |p| (p + 1).min(tags.len().saturating_sub(1)));
                self.distributions.cursor = tags.get(next).cloned();
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let next = position.map_or(0, |p| p.saturating_sub(1));
                self.distributions.cursor = tags.get(next).cloned();
            }
            KeyCode::Esc => self.view = View::Runs,
            _ => {}
        }
    }

    fn update_runs(&mut self, key: KeyEvent) {
        let rows = self.run_rows();
        let names: Vec<&String> = rows.iter().map(|row| &row.name).collect();
        let position = self
            .runs
            .cursor
            .as_ref()
            .and_then(|cursor| names.iter().position(|name| *name == cursor));
        let move_to = |index: Option<usize>| -> Option<String> {
            index
                .and_then(|index| names.get(index))
                .map(|name| (*name).clone())
        };
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                let next = position.map_or(0, |p| (p + 1).min(names.len().saturating_sub(1)));
                self.runs.cursor = move_to(Some(next));
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let next = position.map_or(0, |p| p.saturating_sub(1));
                self.runs.cursor = move_to(Some(next));
            }
            KeyCode::Char('g') => self.runs.cursor = move_to(Some(0)),
            KeyCode::Char('G') => {
                self.runs.cursor = move_to(names.len().checked_sub(1));
            }
            KeyCode::Char(' ') => {
                if let Some(cursor) = self.cursor_or_first(&names)
                    && !self.runs.selected.remove(&cursor)
                {
                    self.runs.selected.insert(cursor);
                }
            }
            KeyCode::Char('s') => self.runs.sort = self.runs.sort.next(),
            KeyCode::Char('S') => self.runs.descending = !self.runs.descending,
            KeyCode::Enter => {
                if self.runs.selected.is_empty()
                    && let Some(cursor) = self.cursor_or_first(&names)
                {
                    self.runs.selected.insert(cursor);
                }
                self.view = View::Scalars;
            }
            _ => {}
        }
    }

    fn cursor_or_first(&self, names: &[&String]) -> Option<String> {
        self.runs
            .cursor
            .clone()
            .filter(|cursor| names.contains(&cursor))
            .or_else(|| names.first().map(|name| (*name).clone()))
    }

    fn update_scalars(&mut self, key: KeyEvent) {
        let tags = self.visible_tags();
        let position = self
            .scalars
            .cursor
            .as_ref()
            .and_then(|cursor| tags.iter().position(|tag| tag == cursor));
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                let next = position.map_or(0, |p| (p + 1).min(tags.len().saturating_sub(1)));
                self.scalars.cursor = tags.get(next).cloned();
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let next = position.map_or(0, |p| p.saturating_sub(1));
                self.scalars.cursor = tags.get(next).cloned();
            }
            KeyCode::Char('g') => self.scalars.cursor = tags.first().cloned(),
            KeyCode::Char('G') => self.scalars.cursor = tags.last().cloned(),
            KeyCode::Char('s') => {
                self.scalars.smooth = (self.scalars.smooth - 0.1).max(0.0);
            }
            KeyCode::Char('S') => {
                self.scalars.smooth = (self.scalars.smooth + 0.1).min(0.9);
            }
            KeyCode::Char('x') => {
                self.scalars.x_axis = match self.scalars.x_axis {
                    XAxis::Step => XAxis::Wall,
                    XAxis::Wall => XAxis::Relative,
                    XAxis::Relative => XAxis::Tokens,
                    XAxis::Tokens => XAxis::Step,
                };
            }
            KeyCode::Char('L') => self.scalars.log_y = !self.scalars.log_y,
            KeyCode::Char('v') => self.scalars.show_ghosts = !self.scalars.show_ghosts,
            KeyCode::Esc => self.view = View::Runs,
            _ => {}
        }
    }

    /// `e`: writes the current view as CSV next to where vertov was started
    /// — never into the logdir.
    fn export(&mut self) {
        let result = match self.view {
            View::Runs => self.export_runs(),
            View::Scalars | View::Compare => self.export_scalars(),
            View::Hparams => self.export_hparams(),
            View::Distributions => self.export_distributions(),
        };
        self.message = Some(match result {
            Ok(path) => format!("exported {path}"),
            Err(err) => format!("export failed: {err}"),
        });
    }

    /// The flat runs × (params + metrics) table, scoped like the view.
    fn export_hparams(&self) -> std::io::Result<String> {
        use crate::table::{Cell, Format, Table};
        fn cell_from(value: HparamCell) -> Cell {
            match value {
                HparamCell::Text(text) => Cell::Text(text),
                HparamCell::Number(number) => Cell::Float(number),
                HparamCell::Empty => Cell::Empty,
            }
        }
        let (columns, rows) = self.hparam_table();
        let rows = rows
            .into_iter()
            .map(|row| row.into_iter().map(cell_from).collect())
            .collect();
        let table = Table { columns, rows };
        let path = "vertov-hparams.csv";
        std::fs::write(path, table.render(Format::Csv))?;
        Ok(path.to_owned())
    }

    fn export_distributions(&self) -> std::io::Result<String> {
        use std::fmt::Write as _;
        let Some((run, tag)) = self.distribution_target() else {
            return Err(std::io::Error::other("no histogram series to export"));
        };
        let Some(series) = self.project.histogram_series(&run, &tag) else {
            return Err(std::io::Error::other("series not materialized yet"));
        };
        let mut out = String::from("run,tag,step,wall,left,right,count\n");
        for snapshot in &series.snapshots {
            for (left, right, count) in &snapshot.buckets {
                let _ = writeln!(
                    out,
                    "{run},{tag},{},{},{left},{right},{count}",
                    snapshot.step, snapshot.wall
                );
            }
        }
        let path = format!("vertov-{}-hist.csv", tag.replace(['/', '\\'], "-"));
        std::fs::write(&path, out)?;
        Ok(path)
    }

    fn export_runs(&self) -> std::io::Result<String> {
        use crate::table::{Cell, Format, Table, fmt_duration};
        let rows = self
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
        let table = Table {
            columns: ["run", "status", "series", "points", "restarts", "step", "duration"]
                .map(String::from)
                .to_vec(),
            rows,
        };
        let path = "vertov-runs.csv";
        std::fs::write(path, table.render(Format::Csv))?;
        Ok(path.to_owned())
    }

    fn export_scalars(&self) -> std::io::Result<String> {
        use std::fmt::Write as _;
        let Some(tag) = self.current_tag() else {
            return Err(std::io::Error::other("no tag to export"));
        };
        let mut out = String::from("run,tag,step,wall,value\n");
        for run in self.scoped_runs() {
            let Some(points) = self.project.points(&run, &tag) else {
                continue;
            };
            for index in 0..points.len() {
                let _ = writeln!(
                    out,
                    "{run},{tag},{},{},{}",
                    points.steps[index], points.walls[index], points.values[index]
                );
            }
        }
        let path = format!("vertov-{}.csv", tag.replace(['/', '\\'], "-"));
        std::fs::write(&path, out)?;
        Ok(path)
    }
}

/// Case-sensitive subsequence match — `tl` finds `train/loss`.
fn fuzzy_match(haystack: &str, needle: &str) -> bool {
    let mut chars = haystack.chars();
    needle
        .chars()
        .all(|wanted| chars.by_ref().any(|have| have == wanted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_subsequence() {
        assert!(fuzzy_match("train/loss", "tl"));
        assert!(fuzzy_match("train/loss", "loss"));
        assert!(fuzzy_match("train/loss", ""));
        assert!(!fuzzy_match("train/loss", "xyz"));
        assert!(!fuzzy_match("train/loss", "lt"));
    }
}
