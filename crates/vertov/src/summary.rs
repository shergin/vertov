//! `vertov summary`: everything the catalog knows about one run — every
//! series' exact summary, no materialization.

use vertov_model::SeriesClass;

use crate::table::{Cell, Table};
use crate::{Args, load_project};

pub fn run(args: &Args, run_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let project = load_project(args)?;
    let Some(run) = project.runs.get(run_name) else {
        let known = project.runs.keys().cloned().collect::<Vec<_>>().join(", ");
        return Err(format!(
            "no run `{run_name}` in {} (runs: {known})",
            args.logdir
        )
        .into());
    };

    let mut rows = Vec::new();
    for (tag, series) in &run.series {
        let summary = &series.summary;
        let last = summary.last();
        rows.push(vec![
            Cell::Text(tag.clone()),
            Cell::Text(class_text(series.class).to_owned()),
            Cell::Int(summary.count() as i64),
            summary.min().map_or(Cell::Empty, Cell::Float),
            summary.max().map_or(Cell::Empty, Cell::Float),
            summary.moments().mean().map_or(Cell::Empty, Cell::Float),
            last.map_or(Cell::Empty, |point| {
                if point.value.is_nan() {
                    Cell::Empty
                } else {
                    Cell::Float(point.value)
                }
            }),
            last.map_or(Cell::Empty, |point| Cell::Int(point.step)),
            Cell::Int(summary.segments.len() as i64),
        ]);
    }
    if !run.hparams.is_empty() && args.format == crate::table::Format::Text {
        let pairs: Vec<String> = run
            .hparams
            .iter()
            .map(|(key, value)| format!("{key}={}", crate::hparam_text(value)))
            .collect();
        println!("hparams: {}", pairs.join("  "));
    }

    let table = Table {
        columns: [
            "tag", "class", "count", "min", "max", "mean", "last", "step", "segments",
        ]
        .map(String::from)
        .to_vec(),
        rows,
    };
    print!("{}", table.render(args.format));
    Ok(())
}

fn class_text(class: SeriesClass) -> &'static str {
    match class {
        SeriesClass::Scalar => "scalar",
        SeriesClass::Histogram => "histogram",
        SeriesClass::Image => "image",
        SeriesClass::Text => "text",
        _ => "?",
    }
}
