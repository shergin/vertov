//! Integration tests over real directories: discovery, summaries, restart
//! preemption, live growth, frontier consistency, vanished runs.

use std::fs;
use std::path::{Path, PathBuf};

use tfevents::writer::{events_file, histogram_event, scalar_event, write_record};
use vertov_model::{Project, RunStatus, SeriesClass};

/// A scratch logdir, removed on drop.
struct Logdir(PathBuf);

impl Logdir {
    fn new(name: &str) -> Logdir {
        let dir = std::env::temp_dir().join(format!(
            "vertov-model-test-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Logdir(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, relative: &str, bytes: &[u8]) {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn append(&self, relative: &str, bytes: &[u8]) {
        use std::io::Write as _;
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(self.0.join(relative))
            .unwrap();
        file.write_all(bytes).unwrap();
    }
}

impl Drop for Logdir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn wall(step: i64) -> f64 {
    1.7e9 + step as f64
}

#[test]
fn discovery_and_summaries() {
    let logdir = Logdir::new("discovery");
    for (run, scale) in [("a", 1.0f32), ("b/nested", 10.0)] {
        let events: Vec<Vec<u8>> = (0..50)
            .map(|step| scalar_event(wall(step), step, "train/loss", step as f32 * scale))
            .collect();
        logdir.write(
            &format!("{run}/events.out.tfevents.1000.host"),
            &events_file(&events),
        );
    }

    let mut project = Project::new(logdir.path());
    let report = project.refresh().unwrap();
    assert_eq!(report.new_files, 2);
    assert_eq!(report.new_points, 100);
    assert_eq!(report.preemptions, 0);

    assert_eq!(
        project.runs.keys().collect::<Vec<_>>(),
        vec!["a", "b/nested"]
    );
    let series = &project.runs["a"].series["train/loss"];
    assert_eq!(series.class, SeriesClass::Scalar);
    let summary = &series.summary;
    assert_eq!(summary.count(), 50);
    assert_eq!(summary.min(), Some(0.0));
    assert_eq!(summary.max(), Some(49.0));
    assert_eq!(summary.first().unwrap().step, 0);
    assert_eq!(summary.last().unwrap().step, 49);
    assert!((summary.moments().mean().unwrap() - 24.5).abs() < 1e-9);
    assert!(!summary.preempted());
    assert_eq!(
        project.runs["b/nested"].series["train/loss"].summary.max(),
        Some(490.0)
    );
    // Wall-clock range landed on the run.
    assert_eq!(project.runs["a"].first_wall, Some(wall(0)));
    assert_eq!(project.runs["a"].last_wall, Some(wall(49)));
}

#[test]
fn restart_preemption_across_files() {
    let logdir = Logdir::new("restart");
    // First writer: steps 0..=10. Crash. Restart resumes from step 8.
    let first: Vec<Vec<u8>> = (0..=10)
        .map(|step| scalar_event(wall(step), step, "loss", step as f32))
        .collect();
    let second: Vec<Vec<u8>> = (8..=12)
        .map(|step| scalar_event(wall(100 + step), step, "loss", 100.0 + step as f32))
        .collect();
    // Filename order must match write order.
    logdir.write("run/events.out.tfevents.1000.host", &events_file(&first));
    logdir.write("run/events.out.tfevents.2000.host", &events_file(&second));

    let mut project = Project::new(logdir.path());
    let report = project.refresh().unwrap();
    assert_eq!(report.preemptions, 1);
    let run = &project.runs["run"];
    assert_eq!(run.preemptions, 1);

    let summary = &run.series["loss"].summary;
    assert_eq!(summary.segments.len(), 2);
    assert_eq!(summary.segments[0].preempted_at, Some(8));
    assert!(summary.preempted());
    assert_eq!(summary.last().unwrap().value, 112.0);

    // Materialized: live prefix 0..=7 from the first writer, 8..=12 from
    // the second; the first writer's 8..=10 is ghost.
    let points = project.materialize("run", "loss").unwrap().unwrap();
    assert_eq!(points.steps, (0..=12).collect::<Vec<i64>>());
    assert_eq!(points.values[7], 7.0);
    assert_eq!(points.values[8], 108.0);
    assert_eq!(points.boundaries, vec![8]);
    assert_eq!(points.ghosts.len(), 1);
    assert_eq!(points.ghosts[0].at, 8);
    assert_eq!(points.ghosts[0].steps, vec![8, 9, 10]);
    assert_eq!(points.ghosts[0].values, vec![8.0, 9.0, 10.0]);
}

#[test]
fn live_growth_appends_without_duplicates() {
    let logdir = Logdir::new("growth");
    let file = "run/events.out.tfevents.1000.host";
    logdir.write(
        file,
        &events_file(&[scalar_event(wall(0), 0, "loss", 1.0)]),
    );

    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    let points = project.materialize("run", "loss").unwrap().unwrap();
    assert_eq!(points.steps, vec![0]);

    // The writer appends: one whole record plus a torn tail.
    let mut growth = Vec::new();
    write_record(&mut growth, &scalar_event(wall(1), 1, "loss", 2.0));
    let mut torn = Vec::new();
    write_record(&mut torn, &scalar_event(wall(2), 2, "loss", 3.0));
    growth.extend_from_slice(&torn[..torn.len() - 5]);
    logdir.append(file, &growth);

    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 1);
    assert_eq!(project.points("run", "loss").unwrap().steps, vec![0, 1]);

    // The torn record completes.
    logdir.append(file, &torn[torn.len() - 5..]);
    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 1);
    let points = project.points("run", "loss").unwrap();
    assert_eq!(points.steps, vec![0, 1, 2]);
    assert_eq!(points.values, vec![1.0, 2.0, 3.0]);
    assert_eq!(project.runs["run"].series["loss"].summary.count(), 3);
}

#[test]
fn materialize_respects_the_frontier() {
    let logdir = Logdir::new("frontier");
    let file = "run/events.out.tfevents.1000.host";
    logdir.write(
        file,
        &events_file(&[scalar_event(wall(0), 0, "loss", 1.0)]),
    );

    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();

    // New data lands on disk *after* the refresh; materialize must not read
    // past the frontier, or the next refresh would double-ingest.
    let mut growth = Vec::new();
    write_record(&mut growth, &scalar_event(wall(1), 1, "loss", 2.0));
    logdir.append(file, &growth);

    let points = project.materialize("run", "loss").unwrap().unwrap();
    assert_eq!(points.steps, vec![0]);

    project.refresh().unwrap();
    let points = project.points("run", "loss").unwrap();
    assert_eq!(points.steps, vec![0, 1]);
    assert_eq!(points.values, vec![1.0, 2.0]);
}

#[test]
fn nan_values_are_gaps_not_lies() {
    let logdir = Logdir::new("nan");
    logdir.write(
        "run/events.out.tfevents.1000.host",
        &events_file(&[
            scalar_event(wall(0), 0, "loss", 4.0),
            scalar_event(wall(1), 1, "loss", f32::NAN),
            scalar_event(wall(2), 2, "loss", 2.0),
        ]),
    );
    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    let summary = &project.runs["run"].series["loss"].summary;
    assert_eq!(summary.count(), 3);
    assert_eq!(summary.segments[0].non_finite, 1);
    assert_eq!(summary.min(), Some(2.0));
    assert_eq!(summary.moments().count(), 2);
    let points = project.materialize("run", "loss").unwrap().unwrap();
    assert!(points.values[1].is_nan());
}

#[test]
fn histogram_series_materialize_and_tail() {
    let logdir = Logdir::new("histograms");
    let file = "run/events.out.tfevents.1000.host";
    let buckets = |scale: f64| vec![(-scale, 0.0, 3.0), (0.0, scale, 5.0)];
    logdir.write(
        file,
        &events_file(&[
            histogram_event(wall(0), 0, "params/w", &buckets(1.0)),
            histogram_event(wall(1), 1, "params/w", &buckets(2.0)),
        ]),
    );

    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    assert_eq!(
        project.runs["run"].series["params/w"].class,
        SeriesClass::Histogram
    );

    let series = project
        .materialize_histograms("run", "params/w")
        .unwrap()
        .unwrap();
    assert_eq!(series.snapshots.len(), 2);
    assert_eq!(series.snapshots[0].buckets, buckets(1.0));
    assert_eq!(series.snapshots[1].buckets, buckets(2.0));

    // Live growth appends into the materialized series without a re-read;
    // a step rewind truncates (preemption applies to every series kind).
    let mut growth = Vec::new();
    write_record(&mut growth, &histogram_event(wall(2), 2, "params/w", &buckets(3.0)));
    write_record(&mut growth, &histogram_event(wall(3), 1, "params/w", &buckets(9.0)));
    logdir.append(file, &growth);
    project.refresh().unwrap();

    let series = project.histogram_series("run", "params/w").unwrap();
    let steps: Vec<i64> = series.snapshots.iter().map(|snapshot| snapshot.step).collect();
    assert_eq!(steps, vec![0, 1]);
    assert_eq!(series.snapshots[1].buckets, buckets(9.0));
    assert_eq!(series.boundaries, vec![1]);
    // Scalars are untouched by a histogram preemption... and vice versa:
    // the summary records it per tag.
    assert_eq!(project.runs["run"].preemptions, 1);
}

#[test]
fn real_fixture_histograms_materialize() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let mut project = Project::new(fixtures.join("tensorboardx"));
    project.refresh().unwrap();
    let series = project
        .materialize_histograms(".", "params/weights")
        .unwrap()
        .expect("fixture has histogram series");
    assert_eq!(series.snapshots.len(), 4);
    for snapshot in &series.snapshots {
        let total: f64 = snapshot.buckets.iter().map(|(_, _, count)| count).sum();
        assert_eq!(total, 101.0, "all 101 samples accounted per snapshot");
        for pair in snapshot.buckets.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "contiguous buckets");
        }
    }
}

#[test]
fn vanished_run_is_dropped() {
    let logdir = Logdir::new("vanish");
    logdir.write(
        "gone/events.out.tfevents.1000.host",
        &events_file(&[scalar_event(wall(0), 0, "loss", 1.0)]),
    );
    logdir.write(
        "stays/events.out.tfevents.1000.host",
        &events_file(&[scalar_event(wall(0), 0, "loss", 1.0)]),
    );
    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    assert_eq!(project.runs.len(), 2);

    fs::remove_dir_all(logdir.path().join("gone")).unwrap();
    project.refresh().unwrap();
    assert_eq!(project.runs.keys().collect::<Vec<_>>(), vec!["stays"]);
}

#[test]
fn missing_root_is_a_state_not_an_error() {
    let logdir = Logdir::new("missing-root");
    let root = logdir.path().join("does-not-exist-yet");
    let mut project = Project::new(&root);
    let report = project.refresh().unwrap();
    assert_eq!(report.new_files, 0);
    assert!(project.runs.is_empty());

    // The trainer starts later; the same project picks it up.
    fs::create_dir_all(root.join("run")).unwrap();
    fs::write(
        root.join("run/events.out.tfevents.1000.host"),
        events_file(&[scalar_event(wall(0), 0, "loss", 1.0)]),
    )
    .unwrap();
    let report = project.refresh().unwrap();
    assert_eq!(report.new_files, 1);
    assert_eq!(project.runs.len(), 1);
}

#[test]
fn status_from_mtime_recency() {
    let logdir = Logdir::new("status");
    logdir.write(
        "run/events.out.tfevents.1000.host",
        &events_file(&[scalar_event(wall(0), 0, "loss", 1.0)]),
    );
    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    let run = &project.runs["run"];
    let now = std::time::SystemTime::now();
    let minute = std::time::Duration::from_secs(60);
    assert_eq!(run.status(now, minute), RunStatus::Active);
    assert_eq!(run.status(now + minute * 10, minute), RunStatus::Idle);
}
