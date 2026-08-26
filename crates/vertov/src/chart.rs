//! Builds the scalar chart from watched data. Zero rendering logic — the
//! chart is a malevich [`Plot`] and the library does the drawing.

use malevich::{Line, Plot};

use crate::logdir::Watcher;

/// More series than this and the chart is soup; the title says what was cut.
const MAX_SERIES: usize = 16;

/// Owned series data for one chart: labels and columns the plot borrows from.
pub struct ChartData {
    series: Vec<(String, Vec<f64>, Vec<f64>)>,
    /// Series beyond [`MAX_SERIES`], dropped from the chart and counted in
    /// the title rather than silently.
    pub cut: usize,
    /// Runs contributing at least one drawn series.
    pub run_count: usize,
}

impl ChartData {
    /// Snapshots the watcher's matching series, capped at [`MAX_SERIES`].
    pub fn collect(watcher: &Watcher) -> ChartData {
        let multi_run = watcher.runs.len() > 1;
        let mut series = Vec::new();
        let mut cut = 0;
        let mut run_count = 0;
        for (run, tags) in &watcher.runs {
            let mut contributed = false;
            for (tag, data) in tags {
                if data.steps.is_empty() {
                    continue;
                }
                if series.len() == MAX_SERIES {
                    cut += 1;
                    continue;
                }
                let label = if multi_run {
                    format!("{run} {tag}")
                } else {
                    tag.clone()
                };
                series.push((label, data.steps.clone(), data.values.clone()));
                contributed = true;
            }
            if contributed {
                run_count += 1;
            }
        }
        ChartData {
            series,
            cut,
            run_count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// The chart: one line per series, legend from labels, title as given.
    pub fn plot<'a>(&'a self, title: &str) -> Plot<'a> {
        let mut plot = Plot::new();
        for (label, steps, values) in &self.series {
            plot = plot.layer(Line::xy(&steps[..], &values[..]).label(label.as_str()));
        }
        plot.title(title)
    }
}
