//! Catalog benchmarks against the Phase 2 budgets: cold-scan a 1000-run
//! logdir, and the steady-state cost of one quiet refresh tick.
//!
//! The logdir is generated on disk once per bench process (1000 runs × 5
//! series × 100 points), so numbers include real file I/O through the page
//! cache — the situation a warm interactive session actually sees.

use std::fs;
use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tfevents::writer::{events_file, scalar_event};
use vertov_model::Project;

const RUNS: usize = 1000;
const SERIES: usize = 5;
const POINTS: i64 = 100;

fn build_logdir() -> PathBuf {
    let root = std::env::temp_dir().join(format!("vertov-scan-bench-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    for run in 0..RUNS {
        let dir = root.join(format!("run-{run:04}"));
        fs::create_dir_all(&dir).unwrap();
        let mut events = Vec::new();
        for step in 0..POINTS {
            for series in 0..SERIES {
                events.push(scalar_event(
                    1.7e9 + step as f64,
                    step,
                    &format!("metrics/series-{series}"),
                    (run as f32).mul_add(0.001, step as f32),
                ));
            }
        }
        fs::write(dir.join("events.out.tfevents.1000.host"), events_file(&events)).unwrap();
    }
    root
}

fn scan(criterion: &mut Criterion) {
    let root = build_logdir();
    let mut group = criterion.benchmark_group("scan");
    group.sample_size(10);

    // Cold scan: discover 1000 runs and ingest every point into summaries.
    group.bench_function("cold_1000_runs", |bencher| {
        bencher.iter(|| {
            let mut project = Project::new(&root);
            let report = project.refresh().unwrap();
            assert_eq!(report.new_points, (RUNS * SERIES) as u64 * POINTS as u64);
            black_box(project.runs.len())
        });
    });

    // Steady state: nothing changed; one tick is a walk plus 1000 stats.
    group.bench_function("quiet_tick_1000_runs", |bencher| {
        let mut project = Project::new(&root);
        project.refresh().unwrap();
        bencher.iter(|| {
            let report = project.refresh().unwrap();
            assert_eq!(report.new_points, 0);
            black_box(report)
        });
    });

    group.finish();
    let _ = fs::remove_dir_all(&root);
}

criterion_group!(benches, scan);
criterion_main!(benches);
