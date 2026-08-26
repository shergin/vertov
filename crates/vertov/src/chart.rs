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
}

/// One drawn series: label, x column, raw values, and the optional
/// smoothed overlay.
struct ChartSeries {
    label: String,
    xs: Vec<f64>,
    values: Vec<f64>,
    smoothed: Option<Vec<f64>>,
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
}

impl ChartData {
    /// Finds scalar series matching `filter` (substring), materializes up
    /// to [`MAX_SERIES`] of them, and snapshots their columns.
    pub fn collect(
        project: &mut Project,
        filter: &str,
        options: &ChartOptions,
    ) -> io::Result<ChartData> {
        let mut matches: Vec<(String, String)> = Vec::new();
        for (run_name, run) in &project.runs {
            if let Some(runs_filter) = &options.runs_filter
                && !run_name.contains(runs_filter.as_str())
            {
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
        for (run, tag) in &matches {
            let Some(points) = project.materialize(run, tag)? else {
                continue;
            };
            let label = if multi_run {
                format!("{run} {tag}")
            } else {
                tag.clone()
            };
            push_series(points, label, options, &mut series, &mut boundaries);
        }
        Ok(ChartData::assemble(series, boundaries, options, cut, run_count))
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
        for run in shown {
            let Some(points) = project.points(run, tag) else {
                continue;
            };
            if points.is_empty() {
                continue;
            }
            run_count += 1;
            let label = if multi_run { run.clone() } else { tag.to_owned() };
            push_series(points, label, options, &mut series, &mut boundaries);
        }
        ChartData::assemble(series, boundaries, options, cut, run_count)
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
}

/// Converts one series' points into chart columns under `options`,
/// collecting restart boundaries (as x-position bit patterns, set-dedupable).
fn push_series(
    points: &vertov_model::Points,
    label: String,
    options: &ChartOptions,
    series: &mut Vec<ChartSeries>,
    boundaries: &mut BTreeSet<u64>,
) {
    if points.is_empty() {
        return;
    }
    let xs: Vec<f64> = match options.x_axis {
        XAxis::Step => points.steps.iter().map(|&step| step as f64).collect(),
        XAxis::Wall => points.walls.clone(),
        XAxis::Relative => {
            let first = points.walls[0];
            points.walls.iter().map(|wall| wall - first).collect()
        }
    };
    for &boundary in &points.boundaries {
        boundaries.insert(xs[boundary].to_bits());
    }
    let smoothed = options.smooth.map(|alpha| stat::ewma(&points.values, alpha));
    series.push(ChartSeries {
        label,
        xs,
        values: points.values.clone(),
        smoothed,
    });
}
