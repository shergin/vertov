//! The catalog and its reload loop: scan the root, drain open readers,
//! keep every summary current — RustBoard-shaped, no async runtime.
//!
//! Readers stay open across refreshes; the file offset is the resume state,
//! so a live tick is an incremental read, not a re-parse. Materialization is
//! a transient full re-read of one series up to the reload loop's committed
//! frontier (never past it, so a later refresh can append incrementally
//! without duplicating points), bounded by an LRU cap.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use tfevents::{
    EventFileReader, EventPayload, HparamValue, ReadEventError, SummaryPayload, SummaryValue,
};

use crate::series::{PointStamp, Points, Series, SeriesClass, SeriesSummary};

/// How a run is currently judged, from file-modification recency — the only
/// signal tfevents offers. Display it with that provenance: "active" means
/// "its files changed recently", nothing stronger.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunStatus {
    /// A file of this run changed within the window.
    Active,
    /// Nothing changed within the window.
    Idle,
    /// No modification time is available.
    Unknown,
}

/// One run: a directory containing at least one events file, named by its
/// path relative to the scanned root (`.` for the root itself).
#[derive(Debug)]
pub struct Run {
    /// The run directory.
    pub dir: PathBuf,
    /// Typed hyperparameters from the hparams plugin, when logged.
    pub hparams: BTreeMap<String, HparamValue>,
    /// Every series of the run, by tag — summaries always current.
    pub series: BTreeMap<String, Series>,
    /// Earliest event wall time seen.
    pub first_wall: Option<f64>,
    /// Latest event wall time seen.
    pub last_wall: Option<f64>,
    /// Latest file modification time, from the most recent refresh.
    pub last_write: Option<SystemTime>,
    /// Total step preemptions observed across the run's series.
    pub preemptions: u64,
}

impl Run {
    fn new(dir: PathBuf) -> Run {
        Run {
            dir,
            hparams: BTreeMap::new(),
            series: BTreeMap::new(),
            first_wall: None,
            last_wall: None,
            last_write: None,
            preemptions: 0,
        }
    }

    /// Status by modification recency: `Active` if a file changed within
    /// `window` of `now`.
    pub fn status(&self, now: SystemTime, window: Duration) -> RunStatus {
        match self.last_write {
            Some(last) => match now.duration_since(last) {
                Ok(elapsed) if elapsed > window => RunStatus::Idle,
                _ => RunStatus::Active,
            },
            None => RunStatus::Unknown,
        }
    }
}

/// What one [`Project::refresh`] pass did.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct RefreshReport {
    /// Newly discovered files opened this pass.
    pub new_files: u64,
    /// Points observed across all series.
    pub new_points: u64,
    /// Step preemptions (restart boundaries) recorded.
    pub preemptions: u64,
    /// Records lost to corruption or undecodable bytes.
    pub dropped_records: u64,
    /// Files that died this pass (bad framing or I/O error).
    pub dead_files: u64,
}

struct FileState {
    run: String,
    reader: EventFileReader<File>,
    dead: bool,
}

struct MaterializedSeries {
    points: Points,
    touched: u64,
}

/// Series materialized concurrently before the least-recently-used is
/// dropped (re-materialization is always possible: the files are the
/// database).
const MATERIALIZE_CAP: usize = 64;

/// A scanned root: runs, series, summaries, open readers, and the
/// materialization table.
pub struct Project {
    root: PathBuf,
    /// The catalog, by run name.
    pub runs: BTreeMap<String, Run>,
    /// Cumulative records lost to corruption or undecodable bytes.
    pub dropped_records: u64,
    /// Cumulative files whose framing died; their valid prefix is retained.
    pub dead_files: u64,
    files: BTreeMap<PathBuf, FileState>,
    /// run name → tag → points, so the per-point hot path looks up with
    /// borrowed keys.
    materialized: BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
    clock: u64,
    /// Resume state loaded from the summary cache, consumed as files open.
    cached: BTreeMap<PathBuf, crate::cache::CachedFile>,
    /// Override for the cache directory (tests, `--no-cache` alternatives);
    /// `None` uses `$XDG_CACHE_HOME/vertov` or `~/.cache/vertov`.
    cache_dir: Option<PathBuf>,
}

impl Project {
    /// A project over `root`. The root may not exist yet — a trainer that
    /// has not started is a normal state, and refreshes simply find nothing.
    pub fn new(root: impl Into<PathBuf>) -> Project {
        Project {
            root: root.into(),
            runs: BTreeMap::new(),
            dropped_records: 0,
            dead_files: 0,
            files: BTreeMap::new(),
            materialized: BTreeMap::new(),
            clock: 0,
            cached: BTreeMap::new(),
            cache_dir: None,
        }
    }

    /// The scanned root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Uses `dir` for the summary cache instead of the default
    /// (`$XDG_CACHE_HOME/vertov` or `~/.cache/vertov`).
    pub fn set_cache_dir(&mut self, dir: impl Into<PathBuf>) {
        self.cache_dir = Some(dir.into());
    }

    /// Loads the summary cache for this root, if present and well-formed.
    /// Call before the first [`refresh`](Self::refresh): summaries install
    /// immediately, and files unchanged since the save are not re-read —
    /// grown files resume from their cached offsets. Returns whether a
    /// cache was loaded. The cache is disposable; a missing or torn one
    /// just means a cold start.
    pub fn load_cache(&mut self) -> bool {
        if !self.files.is_empty() {
            return false;
        }
        match crate::cache::load(&self.root, self.cache_dir.as_deref()) {
            Some(cached) => {
                self.runs = cached.runs;
                self.cached = cached.files;
                true
            }
            None => false,
        }
    }

    /// Saves the summary cache: per live file, its identity
    /// `(path, size, mtime)` and committed offset, plus every run's
    /// summaries. Runs with dead files are left uncached so their damage is
    /// re-detected (and re-reported) next session.
    pub fn save_cache(&self) -> io::Result<()> {
        let dead_runs: std::collections::BTreeSet<&String> = self
            .files
            .values()
            .filter(|state| state.dead)
            .map(|state| &state.run)
            .collect();
        let files: Vec<(PathBuf, String, u64)> = self
            .files
            .iter()
            .filter(|(_, state)| !dead_runs.contains(&state.run))
            .map(|(path, state)| {
                (
                    path.clone(),
                    state.run.clone(),
                    state.reader.committed_offset(),
                )
            })
            .collect();
        crate::cache::save(&self.root, self.cache_dir.as_deref(), &self.runs, &files)
    }

    /// One reload pass: discover new runs and files, drain every live
    /// reader into summaries (and any materialized series), refresh file
    /// modification times, and drop runs whose files vanished.
    pub fn refresh(&mut self) -> io::Result<RefreshReport> {
        let mut report = RefreshReport::default();
        let discovered = discover(&self.root)?;

        // Cache validation pre-pass: a cached file that vanished, shrank,
        // or changed without growing invalidates its whole run — the run's
        // summaries came from bytes we can no longer trust, so drop them
        // and let this pass rebuild from disk. A grown file is the normal
        // live case (tfevents is append-only) and resumes from its offset.
        if !self.cached.is_empty() {
            let present: std::collections::BTreeSet<&PathBuf> =
                discovered.iter().map(|(_, path)| path).collect();
            let mut tainted = std::collections::BTreeSet::new();
            for (path, cached) in &self.cached {
                let intact = present.contains(path)
                    && std::fs::metadata(path).is_ok_and(|metadata| {
                        metadata.len() > cached.size
                            || (metadata.len() == cached.size
                                && metadata.modified().ok() == Some(cached.mtime))
                    });
                if !intact {
                    tainted.insert(cached.run.clone());
                }
            }
            for run in tainted {
                self.runs.remove(&run);
                self.cached.retain(|_, cached| cached.run != run);
            }
        }

        for (run_name, path) in discovered {
            if self.files.contains_key(&path) {
                continue;
            }
            let file = match File::open(&path) {
                Ok(file) => file,
                // Vanished between discovery and open: next pass sees truth.
                Err(err) if err.kind() == ErrorKind::NotFound => continue,
                Err(err) => return Err(err),
            };
            let reader = match self.cached.remove(&path) {
                Some(cached) => EventFileReader::resume(file, cached.offset)?,
                None => EventFileReader::new(file),
            };
            let dir = path.parent().unwrap_or(&self.root).to_path_buf();
            self.runs
                .entry(run_name.clone())
                .or_insert_with(|| Run::new(dir));
            self.files.insert(
                path,
                FileState {
                    run: run_name,
                    reader,
                    dead: false,
                },
            );
            report.new_files += 1;
        }

        // BTreeMap order is lexicographic by path: files within a run drain
        // in filename order, matching tfevents' timestamped naming.
        for state in self.files.values_mut() {
            if state.dead {
                continue;
            }
            let run = self
                .runs
                .get_mut(&state.run)
                .expect("every file state has a run");
            loop {
                match state.reader.next_event() {
                    Ok(event) => {
                        ingest(run, &mut self.materialized, &state.run, &event, &mut report);
                    }
                    Err(ReadEventError::Truncated) => break,
                    Err(
                        ReadEventError::Corrupt { .. } | ReadEventError::Malformed { .. },
                    ) => {
                        report.dropped_records += 1;
                        self.dropped_records += 1;
                    }
                    Err(ReadEventError::BadLengthCrc { .. } | ReadEventError::Io(_)) => {
                        state.dead = true;
                        report.dead_files += 1;
                        self.dead_files += 1;
                        break;
                    }
                }
            }
        }

        // Modification times, and the vanished-file sweep. A vanished file
        // taints its whole run: the sibling readers hold mid-file offsets,
        // so the only honest recovery is to drop the run's state and let
        // the next refresh rebuild it from what is actually on disk
        // (RustBoard semantics: runs whose files vanish are dropped).
        let mut tainted = std::collections::BTreeSet::new();
        for (path, state) in &self.files {
            match std::fs::metadata(path) {
                Ok(metadata) => {
                    if let (Ok(mtime), Some(run)) =
                        (metadata.modified(), self.runs.get_mut(&state.run))
                    {
                        run.last_write =
                            Some(run.last_write.map_or(mtime, |known| known.max(mtime)));
                    }
                }
                Err(err) if err.kind() == ErrorKind::NotFound => {
                    tainted.insert(state.run.clone());
                }
                Err(_) => {}
            }
        }
        for run in tainted {
            self.runs.remove(&run);
            self.materialized.remove(&run);
            self.files.retain(|_, state| state.run != run);
        }

        Ok(report)
    }

    /// Materializes one scalar series: a transient full re-read of the
    /// run's files up to the reload loop's committed frontier, preemption
    /// applied. Returns the points, `None` when the series does not exist
    /// or is not scalar. Subsequent [`refresh`](Self::refresh) passes append
    /// to it incrementally; an LRU cap bounds memory.
    pub fn materialize(&mut self, run_name: &str, tag: &str) -> io::Result<Option<&Points>> {
        self.clock += 1;
        if let Some(entry) = self
            .materialized
            .get_mut(run_name)
            .and_then(|tags| tags.get_mut(tag))
        {
            entry.touched = self.clock;
            return Ok(Some(&self.materialized[run_name][tag].points));
        }
        let Some(series) = self
            .runs
            .get(run_name)
            .and_then(|run| run.series.get(tag))
        else {
            return Ok(None);
        };
        if series.class != SeriesClass::Scalar {
            return Ok(None);
        }

        let mut points = Points::default();
        for (path, state) in &self.files {
            if state.run != run_name {
                continue;
            }
            // Reading past the frontier would double-ingest once the main
            // reader catches up; stopping exactly at it keeps the two paths
            // consistent.
            let frontier = state.reader.committed_offset();
            if frontier == 0 {
                continue;
            }
            let file = match File::open(path) {
                Ok(file) => file,
                Err(err) if err.kind() == ErrorKind::NotFound => continue,
                Err(err) => return Err(err),
            };
            let mut reader = EventFileReader::new(file.take(frontier));
            loop {
                match reader.next_event() {
                    Ok(event) => {
                        let EventPayload::Summary(values) = &event.payload else {
                            continue;
                        };
                        for value in values {
                            if value.tag == tag {
                                points.push(PointStamp {
                                    step: event.step,
                                    wall: event.wall_time,
                                    value: value.scalar().unwrap_or(f64::NAN),
                                });
                            }
                        }
                    }
                    // Already accounted by the main drain pass.
                    Err(
                        ReadEventError::Corrupt { .. } | ReadEventError::Malformed { .. },
                    ) => {}
                    Err(ReadEventError::Truncated | ReadEventError::BadLengthCrc { .. }) => {
                        break;
                    }
                    Err(ReadEventError::Io(err)) => return Err(err),
                }
            }
        }

        let held: usize = self.materialized.values().map(BTreeMap::len).sum();
        if held >= MATERIALIZE_CAP
            && let Some(oldest) = self
                .materialized
                .iter()
                .flat_map(|(run, tags)| {
                    tags.iter()
                        .map(move |(tag, entry)| (entry.touched, run.clone(), tag.clone()))
                })
                .min()
        {
            let (_, run, tag) = oldest;
            if let Some(tags) = self.materialized.get_mut(&run) {
                tags.remove(&tag);
                if tags.is_empty() {
                    self.materialized.remove(&run);
                }
            }
        }
        self.materialized
            .entry(run_name.to_owned())
            .or_default()
            .insert(
                tag.to_owned(),
                MaterializedSeries {
                    points,
                    touched: self.clock,
                },
            );
        Ok(Some(&self.materialized[run_name][tag].points))
    }

    /// The materialized points of a series, if currently held.
    pub fn points(&self, run_name: &str, tag: &str) -> Option<&Points> {
        self.materialized
            .get(run_name)
            .and_then(|tags| tags.get(tag))
            .map(|entry| &entry.points)
    }
}

fn ingest(
    run: &mut Run,
    materialized: &mut BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
    run_name: &str,
    event: &tfevents::Event,
    report: &mut RefreshReport,
) {
    if event.wall_time != 0.0 {
        run.first_wall = Some(run.first_wall.map_or(event.wall_time, |w| w.min(event.wall_time)));
        run.last_wall = Some(run.last_wall.map_or(event.wall_time, |w| w.max(event.wall_time)));
    }
    let EventPayload::Summary(values) = &event.payload else {
        return;
    };
    for value in values {
        // Hparams markers carry run-level data in metadata, not a series.
        if value.tag.starts_with("_hparams_/") {
            if let Some(Ok(hparams)) = tfevents::session_start_hparams(value) {
                run.hparams.extend(hparams);
            }
            continue;
        }
        let series = run.series.entry(value.tag.clone()).or_insert_with(|| Series {
            class: classify(value),
            plugin: value
                .metadata
                .as_ref()
                .map(|metadata| metadata.plugin_name.clone())
                .filter(|name| !name.is_empty()),
            summary: SeriesSummary::default(),
        });
        if series.class == SeriesClass::Unknown {
            series.class = classify(value);
        }
        let point = PointStamp {
            step: event.step,
            wall: event.wall_time,
            value: value.scalar().unwrap_or(f64::NAN),
        };
        if series.summary.observe(point) {
            run.preemptions += 1;
            report.preemptions += 1;
        }
        report.new_points += 1;
        if series.class == SeriesClass::Scalar
            && let Some(entry) = materialized
                .get_mut(run_name)
                .and_then(|tags| tags.get_mut(value.tag.as_str()))
        {
            entry.points.push(point);
        }
    }
}

fn classify(value: &SummaryValue) -> SeriesClass {
    if let Some(metadata) = &value.metadata {
        match metadata.plugin_name.as_str() {
            "scalars" => return SeriesClass::Scalar,
            "histograms" => return SeriesClass::Histogram,
            "images" => return SeriesClass::Image,
            "text" => return SeriesClass::Text,
            _ => {}
        }
    }
    if value.scalar().is_some() {
        return SeriesClass::Scalar;
    }
    match &value.payload {
        SummaryPayload::Histogram(_) => SeriesClass::Histogram,
        SummaryPayload::Image(_) => SeriesClass::Image,
        SummaryPayload::Tensor(tensor) => {
            if tensor.strings().is_some() {
                SeriesClass::Text
            } else if tensor.shape.len() == 2 {
                SeriesClass::Histogram
            } else {
                SeriesClass::Unknown
            }
        }
        _ => SeriesClass::Unknown,
    }
}

/// Walks `root` for tfevents files: `(run name, file path)` pairs, sorted.
/// A run is a directory containing at least one file whose basename contains
/// `tfevents`; its name is the directory's root-relative path.
fn discover(root: &Path) -> io::Result<Vec<(String, PathBuf)>> {
    let mut found = Vec::new();
    // Depth cap in place of true symlink-loop detection; 32 levels is
    // beyond any sane logdir.
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            // A missing root means the trainer has not started; a dir gone
            // mid-walk means a run was deleted. States, not errors.
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) if dir == root => return Err(err),
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                if depth < 32 {
                    stack.push((path, depth + 1));
                }
            } else if name.contains("tfevents") {
                found.push((run_name(root, &dir), path));
            }
        }
    }
    found.sort();
    Ok(found)
}

fn run_name(root: &Path, dir: &Path) -> String {
    let relative = dir.strip_prefix(root).unwrap_or(dir);
    if relative.as_os_str().is_empty() {
        ".".to_owned()
    } else {
        relative
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/")
    }
}
