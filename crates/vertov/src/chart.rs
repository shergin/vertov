//! Builds the scalar chart from the model. Zero rendering logic — the chart
//! is a malevich [`Plot`] and the library does the drawing.

use std::collections::BTreeSet;
use std::io;

use malevich::{Color, Line, Plot, Rule, Scale, stat};

use vertov_model::{Project, SeriesClass};

/// More series than this and the chart is soup; the title says what was cut.
const MAX_SERIES: usize = 16;

/// Which column drives the x axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum XAxis {
    /// Global step.
    Step,
    /// Wall-clock seconds since the epoch.
    Wall,
    /// Wall-clock seconds since each series' first point.
    Relative,
    /// Consumed tokens, mapped through each run's token-counter series
    /// (steps between counter points interpolate linearly; points outside
    /// its coverage are dropped and counted, never guessed).
    Tokens,
}

/// Chart-shaping options shared by `show`, `tail`, and the TUI.
pub struct ChartOptions {
    pub x_axis: XAxis,
    /// EWMA factor in `[0, 1)`: smoothed line over the faded raw one —
    /// smoothing is a labeled overlay, never a replacement.
    pub smooth: Option<f64>,
    /// Only runs whose name contains this.
    pub runs_filter: Option<String>,
    /// Log-10 y axis (values at or below zero become gaps, honestly).
    pub log_y: bool,
    /// Draw preempted ghost tails as faded lines — data honesty over
    /// tidiness, on demand.
    pub show_ghosts: bool,
    /// Explicit token-counter tag for [`XAxis::Tokens`]; `None` tries the
    /// conventional names.
    pub tokens_tag: Option<String>,
}

/// One drawn series: label, x column, raw values, the optional smoothed
/// overlay, and any ghost tails to draw faded.
struct ChartSeries {
    label: String,
    xs: Vec<f64>,
    values: Vec<f64>,
    smoothed: Option<Vec<f64>>,
    ghosts: Vec<(Vec<f64>, Vec<f64>)>,
}

/// Owned series data for one chart: labels and columns the plot borrows
/// from, plus restart boundaries drawn as vertical rules.
pub struct ChartData {
    series: Vec<ChartSeries>,
    /// X positions where a restart segment begins, across the drawn series.
    boundaries: Vec<f64>,
    /// Wall-clock x axis (rendered on malevich's calendar time scale).
    time_x: bool,
    /// Log-10 y axis.
    log_y: bool,
    /// Series beyond [`MAX_SERIES`], dropped from the chart and counted in
    /// the title rather than silently.
    pub cut: usize,
    /// Runs contributing at least one drawn series.
    pub run_count: usize,
    /// Points dropped by the tokens axis for lack of counter coverage.
    pub tokens_dropped: usize,
    /// Runs skipped by the tokens axis because no counter series was found.
    pub runs_without_counter: usize,
}

impl ChartData {
    /// Finds scalar series matching `filter` (substring), materializes up
    /// to [`MAX_SERIES`] of them, and snapshots their columns.
    pub fn collect(
        project: &mut Project,
        filter: &str,
        options: &ChartOptions,
    ) -> io::Result<ChartData> {
        let now = std::time::SystemTime::now();
        let window = std::time::Duration::from_secs(60);
        let predicate = options
            .runs_filter
            .as_deref()
            .and_then(|filter| vertov_model::Predicate::parse(filter).ok());
        let mut matches: Vec<(String, String)> = Vec::new();
        for (run_name, run) in &project.runs {
            if !crate::run_passes(
                options.runs_filter.as_deref(),
                predicate.as_ref(),
                run_name,
                crate::status_text(run, now, window),
                run,
            ) {
                continue;
            }
            for (tag, series) in &run.series {
                if series.class == SeriesClass::Scalar && tag.contains(filter) {
                    matches.push((run_name.clone(), tag.clone()));
                }
            }
        }
        let cut = matches.len().saturating_sub(MAX_SERIES);
        matches.truncate(MAX_SERIES);
        let runs_shown: BTreeSet<&String> = matches.iter().map(|(run, _)| run).collect();
        let multi_run = runs_shown.len() > 1;
        let run_count = runs_shown.len();

        let mut series = Vec::new();
        let mut boundaries = BTreeSet::new();
        let mut tokens_dropped = 0;
        let mut runs_without_counter = std::collections::BTreeSet::new();
        for (run, tag) in &matches {
            // Materialize first (target, then the token counter when the
            // axis needs it), read immutably after.
            let counter_tag = if options.x_axis == XAxis::Tokens {
                let counter_tag = project
                    .runs
                    .get(run)
                    .and_then(|r| r.token_counter(options.tokens_tag.as_deref()));
                match counter_tag {
                    Some(counter_tag) => {
                        project.materialize(run, &counter_tag)?;
                        Some(counter_tag)
                    }
                    None => {
                        runs_without_counter.insert(run.clone());
                        continue;
                    }
                }
            } else {
                None
            };
            project.materialize(run, tag)?;
            let Some(points) = project.points(run, tag) else {
                continue;
            };
            let counter = counter_tag
                .as_deref()
                .and_then(|counter_tag| project.points(run, counter_tag));
            let label = if multi_run {
                format!("{run} {tag}")
            } else {
                tag.clone()
            };
            tokens_dropped +=
                push_series(points, counter, label, options, &mut series, &mut boundaries);
        }
        let mut data = ChartData::assemble(series, boundaries, options, cut, run_count);
        data.tokens_dropped = tokens_dropped;
        data.runs_without_counter = runs_without_counter.len();
        Ok(data)
    }

    /// One exact tag across the given runs (the TUI's shape). Reads only
    /// already-materialized points ([`Project::points`]); runs not yet
    /// materialized simply do not draw until they are.
    pub fn for_tag(
        project: &Project,
        runs: &[String],
        tag: &str,
        options: &ChartOptions,
    ) -> ChartData {
        let cut = runs.len().saturating_sub(MAX_SERIES);
        let shown = &runs[..runs.len().min(MAX_SERIES)];
        let multi_run = shown.len() > 1;
        let mut series = Vec::new();
        let mut boundaries = BTreeSet::new();
        let mut run_count = 0;
        let mut tokens_dropped = 0;
        let mut runs_without_counter = 0;
        for run in shown {
            let Some(points) = project.points(run, tag) else {
                continue;
            };
            if points.is_empty() {
                continue;
            }
            let counter = if options.x_axis == XAxis::Tokens {
                let counter = project
                    .runs
                    .get(run)
                    .and_then(|r| r.token_counter(options.tokens_tag.as_deref()))
                    .and_then(|counter_tag| project.points(run, &counter_tag));
                if counter.is_none() {
                    runs_without_counter += 1;
                    continue;
                }
                counter
            } else {
                None
            };
            run_count += 1;
            let label = if multi_run { run.clone() } else { tag.to_owned() };
            tokens_dropped +=
                push_series(points, counter, label, options, &mut series, &mut boundaries);
        }
        let mut data = ChartData::assemble(series, boundaries, options, cut, run_count);
        data.tokens_dropped = tokens_dropped;
        data.runs_without_counter = runs_without_counter;
        data
    }

    fn assemble(
        series: Vec<ChartSeries>,
        boundaries: BTreeSet<u64>,
        options: &ChartOptions,
        cut: usize,
        run_count: usize,
    ) -> ChartData {
        ChartData {
            series,
            boundaries: boundaries.into_iter().map(f64::from_bits).collect(),
            time_x: options.x_axis == XAxis::Wall,
            log_y: options.log_y,
            cut,
            run_count,
            tokens_dropped: 0,
            runs_without_counter: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// The chart: one line per series (faded raw under the smoothed overlay
    /// when smoothing is on), a vertical rule per restart boundary, legend
    /// from labels, title as given.
    pub fn plot<'a>(&'a self, title: &str) -> Plot<'a> {
        let mut plot = Plot::new();
        for series in &self.series {
            // Ghost tails go under everything, faded and unlabeled.
            for (xs, values) in &series.ghosts {
                plot = plot.layer(Line::xy(&xs[..], &values[..]).color(Color::BrightBlack));
            }
            let raw = Line::xy(&series.xs[..], &series.values[..]);
            plot = match &series.smoothed {
                Some(smoothed) => plot.layer(raw.color(Color::BrightBlack)).layer(
                    Line::xy(&series.xs[..], &smoothed[..]).label(series.label.as_str()),
                ),
                None => plot.layer(raw.label(series.label.as_str())),
            };
        }
        for (index, &boundary) in self.boundaries.iter().enumerate() {
            let rule = Rule::v(boundary);
            plot = plot.layer(if index == 0 {
                rule.label("restart")
            } else {
                rule
            });
        }
        if self.time_x {
            plot = plot.x_scale(Scale::Time);
        }
        if self.log_y {
            plot = plot.y_scale(Scale::Log);
        }
        plot.title(title)
    }

    /// The x range this chart's live series span, for sharing a domain
    /// across compare panels.
    pub fn x_extent(&self) -> Option<(f64, f64)> {
        let mut extent: Option<(f64, f64)> = None;
        for series in &self.series {
            let (Some(&first), Some(&last)) = (series.xs.first(), series.xs.last()) else {
                continue;
            };
            extent = Some(match extent {
                Some((min, max)) => (min.min(first), max.max(last)),
                None => (first, last),
            });
        }
        extent
    }

    /// A small-multiples panel: the same layers as [`plot`](Self::plot) but
    /// legend-free (colors stay consistent because runs layer in the same
    /// order in every panel) and on a shared x domain.
    pub fn compare_plot<'a>(&'a self, title: &str, domain: Option<(f64, f64)>) -> Plot<'a> {
        let mut plot = Plot::new();
        for series in &self.series {
            let raw = Line::xy(&series.xs[..], &series.values[..]);
            plot = match &series.smoothed {
                Some(smoothed) => plot
                    .layer(raw.color(Color::BrightBlack))
                    .layer(Line::xy(&series.xs[..], &smoothed[..])),
                None => plot.layer(raw),
            };
        }
        for &boundary in &self.boundaries {
            plot = plot.layer(Rule::v(boundary));
        }
        if let Some((min, max)) = domain
            && min < max
        {
            plot = plot.x_domain(min, max);
        }
        if self.time_x {
            plot = plot.x_scale(Scale::Time);
        }
        if self.log_y {
            plot = plot.y_scale(Scale::Log);
        }
        plot.title(title)
    }
}

/// Ridgeline data for one histogram series: sampled snapshots as lifted
/// rectangular profiles, rendered back to front (the TensorBoard histogram
/// dashboard, malevich's documented ridgeline composition).
pub struct DistData {
    /// Rows oldest-first; each is `(xs, lifted ys)`. Oldest draws first at
    /// the highest lift so nearer rows overwrite what they cross.
    rows: Vec<(Vec<f64>, Vec<f64>)>,
    /// Steps of the back (oldest) and front (newest) drawn rows.
    pub step_range: (i64, i64),
    /// Rows drawn vs snapshots held — the title says when it sampled.
    pub drawn: usize,
    pub total: usize,
}

impl DistData {
    /// Samples up to `max_rows` snapshots evenly (always including the
    /// first and the last) and builds the lifted profiles.
    pub fn build(series: &vertov_model::HistogramSeries, max_rows: usize) -> Option<DistData> {
        let total = series.snapshots.len();
        if total == 0 || max_rows == 0 {
            return None;
        }
        let indices: Vec<usize> = if total <= max_rows {
            (0..total).collect()
        } else {
            (0..max_rows)
                .map(|row| row * (total - 1) / (max_rows - 1))
                .collect()
        };
        const LIFT: f64 = 0.55;
        const HEIGHT: f64 = 1.6;
        let count = indices.len();
        let mut rows = Vec::with_capacity(count);
        for (position, &index) in indices.iter().enumerate() {
            let snapshot = &series.snapshots[index];
            let lift = (count - 1 - position) as f64 * LIFT;
            // Density profile normalized to a fixed peak per row, drawn as
            // a rectangular silhouette: baseline, per-bucket steps,
            // baseline.
            let peak = snapshot
                .buckets
                .iter()
                .map(|(left, right, count)| count / (right - left).max(f64::MIN_POSITIVE))
                .fold(0.0, f64::max);
            let mut xs = Vec::with_capacity(snapshot.buckets.len() + 2);
            let mut ys = Vec::with_capacity(snapshot.buckets.len() + 2);
            if let Some(&(left, _, _)) = snapshot.buckets.first() {
                xs.push(left);
                ys.push(lift);
            }
            for &(left, right, count) in &snapshot.buckets {
                let density = count / (right - left).max(f64::MIN_POSITIVE);
                let height = if peak > 0.0 { density / peak * HEIGHT } else { 0.0 };
                xs.push(left);
                ys.push(lift + height);
            }
            if let Some(&(_, right, _)) = snapshot.buckets.last() {
                xs.push(right);
                ys.push(lift);
            }
            rows.push((xs, ys));
        }
        Some(DistData {
            rows,
            step_range: (
                series.snapshots[indices[0]].step,
                series.snapshots[*indices.last().expect("nonempty")].step,
            ),
            drawn: count,
            total,
        })
    }

    /// The ridgeline: rows layered oldest (farthest, highest lift) first in
    /// the corners style, so each nearer row overwrites what it crosses.
    pub fn plot<'a>(&'a self, title: &str) -> Plot<'a> {
        let mut plot = Plot::new();
        for (xs, ys) in &self.rows {
            plot = plot.layer(Line::xy(&xs[..], &ys[..]).style(malevich::LineStyle::Corners));
        }
        plot.title(title)
    }
}

/// Converts one series' points into chart columns under `options`,
/// collecting restart boundaries (as x-position bit patterns,
/// set-dedupable). Returns how many points the tokens axis had to drop for
/// lack of counter coverage.
fn push_series(
    points: &vertov_model::Points,
    counter: Option<&vertov_model::Points>,
    label: String,
    options: &ChartOptions,
    series: &mut Vec<ChartSeries>,
    boundaries: &mut BTreeSet<u64>,
) -> usize {
    if points.is_empty() {
        return 0;
    }
    let first_wall = points.walls[0];
    let mut dropped = 0;
    let mut columns = |steps: &[i64], walls: &[f64], values: &[f64]| -> (Vec<f64>, Vec<f64>) {
        match options.x_axis {
            XAxis::Step => (steps.iter().map(|&step| step as f64).collect(), values.to_vec()),
            XAxis::Wall => (walls.to_vec(), values.to_vec()),
            XAxis::Relative => (
                walls.iter().map(|wall| wall - first_wall).collect(),
                values.to_vec(),
            ),
            XAxis::Tokens => {
                let counter = counter.expect("caller guarantees a counter for tokens");
                let mut xs = Vec::with_capacity(steps.len());
                let mut kept = Vec::with_capacity(values.len());
                for (index, &step) in steps.iter().enumerate() {
                    match tokens_for(counter, step) {
                        Some(tokens) => {
                            xs.push(tokens);
                            kept.push(values[index]);
                        }
                        None => dropped += 1,
                    }
                }
                (xs, kept)
            }
        }
    };

    let (xs, values) = columns(&points.steps, &points.walls, &points.values);
    let ghosts = if options.show_ghosts {
        points
            .ghosts
            .iter()
            .map(|ghost| columns(&ghost.steps, &ghost.walls, &ghost.values))
            .filter(|(xs, _)| !xs.is_empty())
            .collect()
    } else {
        Vec::new()
    };
    if xs.is_empty() {
        return dropped;
    }
    for &boundary in &points.boundaries {
        let x = match options.x_axis {
            XAxis::Step => Some(points.steps[boundary] as f64),
            XAxis::Wall => Some(points.walls[boundary]),
            XAxis::Relative => Some(points.walls[boundary] - first_wall),
            XAxis::Tokens => tokens_for(
                counter.expect("caller guarantees a counter for tokens"),
                points.steps[boundary],
            ),
        };
        if let Some(x) = x {
            boundaries.insert(x.to_bits());
        }
    }
    let smoothed = options.smooth.map(|alpha| stat::ewma(&values, alpha));
    series.push(ChartSeries {
        label,
        xs,
        values,
        smoothed,
        ghosts,
    });
    dropped
}

/// The token count at `step`, from the counter's materialized points: exact
/// where the counter logged that step, linearly interpolated between
/// neighbors, and `None` outside its coverage — never extrapolated.
fn tokens_for(counter: &vertov_model::Points, step: i64) -> Option<f64> {
    match counter.steps.binary_search(&step) {
        Ok(index) => Some(counter.values[index]).filter(|value| value.is_finite()),
        Err(0) => None,
        Err(index) if index == counter.steps.len() => None,
        Err(index) => {
            let (step_before, step_after) =
                (counter.steps[index - 1] as f64, counter.steps[index] as f64);
            let (before, after) = (counter.values[index - 1], counter.values[index]);
            if !before.is_finite() || !after.is_finite() {
                return None;
            }
            Some(before + (after - before) * (step as f64 - step_before) / (step_after - step_before))
        }
    }
}
