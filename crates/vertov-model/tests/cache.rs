//! Summary-cache tests: warm starts must not re-read unchanged files, grown
//! files resume from their offsets, anything else invalidates honestly, and
//! a torn cache means nothing worse than a cold start.

use std::fs;
use std::path::{Path, PathBuf};

use tfevents::writer::{events_file, scalar_event, write_record};
use vertov_model::Project;

struct Scratch {
    root: PathBuf,
    cache: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Scratch {
        let base = std::env::temp_dir().join(format!(
            "vertov-cache-test-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        let root = base.join("logs");
        let cache = base.join("cache");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&cache).unwrap();
        Scratch { root, cache }
    }

    fn project(&self) -> Project {
        let mut project = Project::new(&self.root);
        project.set_cache_dir(&self.cache);
        project
    }

    fn write(&self, relative: &str, bytes: &[u8]) {
        let path = self.root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn cache_file(&self) -> PathBuf {
        let entries: Vec<PathBuf> = fs::read_dir(&self.cache)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(entries.len(), 1, "expected exactly one cache file");
        entries[0].clone()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(self.root.parent().unwrap_or(Path::new("")));
    }
}

fn wall(step: i64) -> f64 {
    1.7e9 + step as f64
}

fn write_run(scratch: &Scratch, run: &str, steps: std::ops::Range<i64>) {
    let events: Vec<Vec<u8>> = steps
        .map(|step| scalar_event(wall(step), step, "loss", step as f32))
        .collect();
    scratch.write(
        &format!("{run}/events.out.tfevents.1000.host"),
        &events_file(&events),
    );
}

#[test]
fn warm_start_is_a_metadata_walk() {
    let scratch = Scratch::new("warm");
    write_run(&scratch, "a", 0..50);
    write_run(&scratch, "b", 0..30);

    let mut cold = scratch.project();
    cold.refresh().unwrap();
    cold.save_cache().unwrap();

    let mut warm = scratch.project();
    assert!(warm.load_cache());
    let report = warm.refresh().unwrap();
    // Every summary is present without a single event re-read.
    assert_eq!(report.new_points, 0);
    assert_eq!(warm.runs["a"].series["loss"].summary.count(), 50);
    assert_eq!(warm.runs["b"].series["loss"].summary.count(), 30);
    assert_eq!(warm.runs["a"].series["loss"].summary.max(), Some(49.0));
    assert_eq!(warm.runs["a"].first_wall, Some(wall(0)));

    // Materialization still yields full-fidelity points (the files are the
    // database; the cache holds only summaries and offsets).
    let points = warm.materialize("a", "loss").unwrap().unwrap();
    assert_eq!(points.len(), 50);
    assert_eq!(points.values[49], 49.0);
}

#[test]
fn grown_file_resumes_from_cached_offset() {
    let scratch = Scratch::new("grown");
    let file = "run/events.out.tfevents.1000.host";
    write_run(&scratch, "run", 0..10);

    let mut cold = scratch.project();
    cold.refresh().unwrap();
    cold.save_cache().unwrap();

    // The trainer keeps writing after the save.
    let mut growth = Vec::new();
    for step in 10..15 {
        write_record(&mut growth, &scalar_event(wall(step), step, "loss", step as f32));
    }
    use std::io::Write as _;
    let mut handle = fs::OpenOptions::new()
        .append(true)
        .open(scratch.root.join(file))
        .unwrap();
    handle.write_all(&growth).unwrap();
    drop(handle);

    let mut warm = scratch.project();
    assert!(warm.load_cache());
    let report = warm.refresh().unwrap();
    // Only the five appended points are read.
    assert_eq!(report.new_points, 5);
    let summary = &warm.runs["run"].series["loss"].summary;
    assert_eq!(summary.count(), 15);
    assert_eq!(summary.last().unwrap().step, 14);
    assert!(!summary.preempted());
}

#[test]
fn shrunk_file_invalidates_the_run() {
    let scratch = Scratch::new("shrunk");
    write_run(&scratch, "run", 0..20);

    let mut cold = scratch.project();
    cold.refresh().unwrap();
    cold.save_cache().unwrap();

    // The run is rewritten shorter (replaced, restarted from scratch).
    write_run(&scratch, "run", 0..5);

    let mut warm = scratch.project();
    assert!(warm.load_cache());
    let report = warm.refresh().unwrap();
    // Full honest re-read: exactly the five real points, no stale summary.
    assert_eq!(report.new_points, 5);
    assert_eq!(warm.runs["run"].series["loss"].summary.count(), 5);
    assert_eq!(warm.runs["run"].series["loss"].summary.max(), Some(4.0));
}

#[test]
fn torn_cache_is_a_cold_start() {
    let scratch = Scratch::new("torn");
    write_run(&scratch, "run", 0..10);

    let mut cold = scratch.project();
    cold.refresh().unwrap();
    cold.save_cache().unwrap();

    // Truncate the cache file mid-record.
    let cache_file = scratch.cache_file();
    let bytes = fs::read(&cache_file).unwrap();
    fs::write(&cache_file, &bytes[..bytes.len() / 2]).unwrap();

    let mut warm = scratch.project();
    assert!(!warm.load_cache());
    let report = warm.refresh().unwrap();
    assert_eq!(report.new_points, 10);
    assert_eq!(warm.runs["run"].series["loss"].summary.count(), 10);
}

#[test]
fn cache_preserves_segments_and_hparams() {
    let scratch = Scratch::new("segments");
    // A run with a restart: 0..=10 then resumed from 8.
    let first: Vec<Vec<u8>> = (0..=10)
        .map(|step| scalar_event(wall(step), step, "loss", step as f32))
        .collect();
    let second: Vec<Vec<u8>> = (8..=12)
        .map(|step| scalar_event(wall(step), step, "loss", 100.0 + step as f32))
        .collect();
    scratch.write("run/events.out.tfevents.1000.host", &events_file(&first));
    scratch.write("run/events.out.tfevents.2000.host", &events_file(&second));

    let mut cold = scratch.project();
    cold.refresh().unwrap();
    cold.save_cache().unwrap();

    let mut warm = scratch.project();
    assert!(warm.load_cache());
    warm.refresh().unwrap();
    let summary = &warm.runs["run"].series["loss"].summary;
    assert_eq!(summary.segments.len(), 2);
    assert_eq!(summary.segments[0].preempted_at, Some(8));
    assert!(summary.preempted());
    assert_eq!(warm.runs["run"].preemptions, 1);
}
