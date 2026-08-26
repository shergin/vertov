//! `vertov export`: the flat runs × (params + metrics) table — the
//! HiPlot-shaped interchange format. One row per run; columns are the union
//! of hyperparameter keys and scalar tags (last value), gaps left empty.

use std::collections::BTreeSet;

use vertov_model::{HparamValue, SeriesClass};

use crate::table::{Cell, Table};
use crate::{Args, load_project};

pub fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let project = load_project(args)?;
    let now = std::time::SystemTime::now();
    let window = std::time::Duration::from_secs(60);
    let predicate = args
        .runs_filter
        .as_deref()
        .and_then(|filter| vertov_model::Predicate::parse(filter).ok());
    let passes = |name: &str, run: &vertov_model::Run| {
        crate::run_passes(
            args.runs_filter.as_deref(),
            predicate.as_ref(),
            name,
            crate::status_text(run, now, window),
            run,
        )
    };

    let mut param_keys = BTreeSet::new();
    let mut metric_tags = BTreeSet::new();
    for (name, run) in &project.runs {
        if !passes(name, run) {
            continue;
        }
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

    let mut rows = Vec::new();
    for (name, run) in &project.runs {
        if !passes(name, run) {
            continue;
        }
        let mut row = vec![Cell::Text(name.clone())];
        for key in &param_keys {
            row.push(match run.hparams.get(key) {
                Some(HparamValue::F64(value)) => Cell::Float(*value),
                Some(HparamValue::Bool(value)) => Cell::Text(value.to_string()),
                Some(HparamValue::String(value)) => Cell::Text(value.clone()),
                None => Cell::Empty,
            });
        }
        for tag in &metric_tags {
            row.push(
                run.series
                    .get(tag)
                    .and_then(|series| series.summary.last())
                    .map_or(Cell::Empty, |point| Cell::Float(point.value)),
            );
        }
        rows.push(row);
    }

    let table = Table { columns, rows };
    print!("{}", table.render(args.format));
    Ok(())
}
