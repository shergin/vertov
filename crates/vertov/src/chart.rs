//! Builds the scalar chart from the model. Zero rendering logic — the chart
//! is a malevich [`Plot`] and the library does the drawing.

use std::collections::BTreeSet;
use std::io;

use malevich::{Line, Plot, Rule};

use vertov_model::{Project, SeriesClass};

/// More series than this and the chart is soup; the title says what was cut.
const MAX_SERIES: usize = 16;

/// Owned series data for one chart: labels and columns the plot borrows
/// from, plus restart boundaries drawn as vertical rules.
pub struct ChartData {
    series: Vec<(String, Vec<f64>, Vec<f64>)>,
    /// Steps where a restart segment begins, across the drawn series.
    boundaries: Vec<f64>,
    /// Series beyond [`MAX_SERIES`], dropped from the chart and counted in
    /// the title rather than silently.
    pub cut: usize,
    /// Runs contributing at least one drawn series.
    pub run_count: usize,
}

impl ChartData {
    /// Finds scalar series matching `filter` (substring), materializes up
    /// to [`MAX_SERIES`] of them, and snapshots their columns.
    pub fn collect(project: &mut Project, filter: &str) -> io::Result<ChartData> {
        let mut matches: Vec<(String, String)> = Vec::new();
        for (run_name, run) in &project.runs {
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
            if points.is_empty() {
                continue;
            }
            let steps: Vec<f64> = points.steps.iter().map(|&step| step as f64).collect();
            for &boundary in &points.boundaries {
                boundaries.insert(points.steps[boundary]);
            }
            let label = if multi_run {
                format!("{run} {tag}")
            } else {
                tag.clone()
            };
            series.push((label, steps, points.values.clone()));
        }
        Ok(ChartData {
            series,
            boundaries: boundaries.into_iter().map(|step| step as f64).collect(),
            cut,
            run_count,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// The chart: one line per series, a vertical rule per restart
    /// boundary, legend from labels, title as given.
    pub fn plot<'a>(&'a self, title: &str) -> Plot<'a> {
        let mut plot = Plot::new();
        for (label, steps, values) in &self.series {
            plot = plot.layer(Line::xy(&steps[..], &values[..]).label(label.as_str()));
        }
        for (index, &boundary) in self.boundaries.iter().enumerate() {
            let rule = Rule::v(boundary);
            plot = plot.layer(if index == 0 {
                rule.label("restart")
            } else {
                rule
            });
        }
        plot.title(title)
    }
}
