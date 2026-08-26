//! `vertov ls`: the runs table, one row per run.

use crate::table::{Cell, Table, fmt_duration};
use crate::{Args, load_project};

pub fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let project = load_project(args)?;
    let now = std::time::SystemTime::now();
    let window = std::time::Duration::from_secs(60);
    let predicate = args
        .runs_filter
        .as_deref()
        .and_then(|filter| vertov_model::Predicate::parse(filter).ok());

    let mut rows = Vec::new();
    for (name, run) in &project.runs {
        let status = crate::status_text(run, now, window);
        if !crate::run_passes(
            args.runs_filter.as_deref(),
            predicate.as_ref(),
            name,
            status,
            run,
        ) {
            continue;
        }
        let points: u64 = run.series.values().map(|series| series.summary.count()).sum();
        let last_step = run
            .series
            .values()
            .filter_map(|series| series.summary.last().map(|point| point.step))
            .max();
        let duration = match (run.first_wall, run.last_wall) {
            (Some(first), Some(last)) if last > first => {
                Cell::Text(fmt_duration(last - first))
            }
            _ => Cell::Empty,
        };
        rows.push(vec![
            Cell::Text(name.clone()),
            Cell::Text(crate::status_text(run, now, window).to_owned()),
            Cell::Int(run.series.len() as i64),
            Cell::Int(points as i64),
            Cell::Int(run.preemptions as i64),
            last_step.map_or(Cell::Empty, Cell::Int),
            duration,
        ]);
    }

    let table = Table {
        columns: ["run", "status", "series", "points", "restarts", "step", "duration"]
            .map(String::from)
            .to_vec(),
        rows,
    };
    print!("{}", table.render(args.format));
    Ok(())
}
