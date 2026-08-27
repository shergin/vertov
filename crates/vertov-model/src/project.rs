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

use crate::series::{
    HistogramSeries, HistogramSnapshot, PointStamp, Points, Series, SeriesClass, SeriesSummary,
};

/// How a run is currently judged, from file-modification recency — the only
/// signal tfevents offers. Display it with that provenance: "active" means
/// "its files changed recently", nothing stronger.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunStatus {
    /// A file of this run changed within the window.
    Active,
    /// Nothing changed within the window.
    Idle,
    /// The backend recorded a clean exit (wandb's exit record) — stronger
    /// provenance than modification recency.
    Finished,
    /// No modification time is available.
    Unknown,
}

/// Which format a run's data comes from. The catalog above this line is
/// backend-blind: every backend normalizes into the same series, summaries,
/// and preemption semantics.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum Backend {
    /// TensorBoard event files.
    Tfevents,
    /// dvclive: tailable TSVs under `dvclive/plots/metrics/`.
    Dvclive,
    /// The MLflow file store: `mlruns/<exp>/<run>/metrics/*`.
    Mlflow,
    /// wandb offline: `offline-run-*/run-*.wandb` transaction logs.
    Wandb,
}

impl Backend {
    /// The lowercase label shown in tables.
    pub fn label(self) -> &'static str {
        match self {
            Backend::Tfevents => "tfevents",
            Backend::Dvclive => "dvclive",
            Backend::Mlflow => "mlflow",
            Backend::Wandb => "wandb",
        }
    }
}

/// One run, named by its path relative to the scanned root (`.` for the
/// root itself) or, for MLflow, by its recorded run name.
#[derive(Debug)]
pub struct Run {
    /// Where the run's data comes from.
    pub backend: Backend,
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
    /// A backend-recorded clean exit (wandb); `None` when the backend has
    /// no such signal — status then falls back to modification recency.
    pub finished: Option<bool>,
    /// Total step preemptions observed across the run's series.
    pub preemptions: u64,
}

impl Run {
    fn new(backend: Backend, dir: PathBuf) -> Run {
        Run {
            backend,
            dir,
            hparams: BTreeMap::new(),
            series: BTreeMap::new(),
            first_wall: None,
            last_wall: None,
            last_write: None,
            finished: None,
            preemptions: 0,
        }
    }

    /// The series counting consumed tokens, for a tokens x axis. An
    /// `explicit` tag wins (resolved exactly, then as a unique
    /// `/`-suffix); otherwise the conventional names are tried in order.
    /// LR schedules and anneal points are defined in tokens — this is what
    /// lets plots be too.
    pub fn token_counter(&self, explicit: Option<&str>) -> Option<String> {
        const CONVENTIONS: &[&str] = &[
            "tokens",
            "consumed_tokens",
            "total_tokens",
            "num_tokens",
            "tokens_seen",
        ];
        let resolve = |wanted: &str| -> Option<String> {
            if self.series.contains_key(wanted) {
                return Some(wanted.to_owned());
            }
            let suffix = format!("/{wanted}");
            let mut matches = self.series.keys().filter(|tag| tag.ends_with(&suffix));
            if let (Some(tag), None) = (matches.next(), matches.next()) {
                return Some(tag.clone());
            }
            None
        };
        match explicit {
            Some(explicit) => resolve(explicit),
            None => CONVENTIONS.iter().find_map(|candidate| resolve(candidate)),
        }
    }

    /// Status: a backend-recorded clean exit wins; otherwise modification
    /// recency — `Active` if a file changed within `window` of `now`.
    pub fn status(&self, now: SystemTime, window: Duration) -> RunStatus {
        if self.finished == Some(true) {
            return RunStatus::Finished;
        }
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

/// Resume state for one dvclive root: per-TSV byte offsets (the files are
/// append-only, so the offset after the last complete line is the whole
/// resume story) and the params file's last-seen mtime.
struct DvcliveState {
    run: String,
    offsets: BTreeMap<PathBuf, u64>,
    schemas: BTreeMap<PathBuf, vertov_formats::dvclive::TsvSchema>,
    params_mtime: Option<SystemTime>,
}

/// Resume state for one MLflow run dir: per-metric-file offsets and which
/// (immutable) param files have been read.
struct MlflowState {
    run: String,
    offsets: BTreeMap<PathBuf, u64>,
    params_seen: std::collections::BTreeSet<PathBuf>,
}

/// Resume state for one wandb offline run: the committed byte offset into
/// its `.wandb` log (0 until the header is validated) and whether the file
/// was rejected (bad header or newer version — dead, prefix retained).
struct WandbState {
    run: String,
    offset: u64,
    dead: bool,
}

struct MaterializedSeries {
    points: Points,
    touched: u64,
}

struct MaterializedHistograms {
    series: HistogramSeries,
    touched: u64,
}

/// Series materialized concurrently before the least-recently-used is
/// dropped (re-materialization is always possible: the files are the
/// database).
const MATERIALIZE_CAP: usize = 64;

/// Histogram series held concurrently — snapshots are bulkier than points,
/// and the distributions view looks at one tag at a time.
const HISTOGRAM_CAP: usize = 8;

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
    /// dvclive roots (the `dvclive` directory) by path.
    dvclive: BTreeMap<PathBuf, DvcliveState>,
    /// MLflow run directories by path.
    mlflow: BTreeMap<PathBuf, MlflowState>,
    /// wandb `.wandb` log files by path.
    wandb: BTreeMap<PathBuf, WandbState>,
    /// run name → tag → points, so the per-point hot path looks up with
    /// borrowed keys.
    materialized: BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
    /// Same shape for histogram series.
    histograms: BTreeMap<String, BTreeMap<String, MaterializedHistograms>>,
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
            dvclive: BTreeMap::new(),
            mlflow: BTreeMap::new(),
            wandb: BTreeMap::new(),
            materialized: BTreeMap::new(),
            histograms: BTreeMap::new(),
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
        let Discovery {
            tfevents: discovered_tfevents,
            dvclive: discovered_dvclive,
            mlflow: discovered_mlflow,
            wandb: discovered_wandb,
        } = discover(&self.root)?;

        // Cache validation pre-pass: a cached file that vanished, shrank,
        // or changed without growing invalidates its whole run — the run's
        // summaries came from bytes we can no longer trust, so drop them
        // and let this pass rebuild from disk. A grown file is the normal
        // live case (tfevents is append-only) and resumes from its offset.
        if !self.cached.is_empty() {
            let present: std::collections::BTreeSet<&PathBuf> =
                discovered_tfevents.iter().map(|(_, path)| path).collect();
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

        for (run_name, path) in discovered_tfevents {
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
                .or_insert_with(|| Run::new(Backend::Tfevents, dir));
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
                        ingest(
                            run,
                            &mut self.materialized,
                            &mut self.histograms,
                            &state.run,
                            &event,
                            &mut report,
                        );
                    }
                    Err(ReadEventError::Truncated) => break,
                    Err(
                        ReadEventError::Corrupt { .. } | ReadEventError::Malformed { .. },
                    ) => {
                        report.dropped_records += 1;
                    }
                    Err(ReadEventError::BadLengthCrc { .. } | ReadEventError::Io(_)) => {
                        state.dead = true;
                        report.dead_files += 1;
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
            self.histograms.remove(&run);
            self.files.retain(|_, state| state.run != run);
        }

        // dvclive and MLflow: sync states with what the walk found (a
        // vanished root drops its run), then drain each — the same
        // append-only, offset-resumed discipline, over text lines.
        let wanted_dvclive: BTreeMap<PathBuf, String> = discovered_dvclive.into_iter()
            .map(|(run, path)| (path, run))
            .collect();
        let gone: Vec<PathBuf> = self
            .dvclive
            .keys()
            .filter(|path| !wanted_dvclive.contains_key(*path))
            .cloned()
            .collect();
        for path in gone {
            if let Some(state) = self.dvclive.remove(&path) {
                self.runs.remove(&state.run);
                self.materialized.remove(&state.run);
            }
        }
        for (path, base_name) in wanted_dvclive {
            if !self.dvclive.contains_key(&path) {
                let name = unique_run_name(&self.runs, base_name);
                self.runs.entry(name.clone()).or_insert_with(|| {
                    Run::new(
                        Backend::Dvclive,
                        path.parent().unwrap_or(&self.root).to_path_buf(),
                    )
                });
                self.dvclive.insert(
                    path.clone(),
                    DvcliveState {
                        run: name,
                        offsets: BTreeMap::new(),
                        schemas: BTreeMap::new(),
                        params_mtime: None,
                    },
                );
            }
            let state = self.dvclive.get_mut(&path).expect("just ensured");
            let run_name = state.run.clone();
            let run = self.runs.get_mut(&run_name).expect("state has a run");
            drain_dvclive(&path, state, run, &run_name, &mut self.materialized, &mut report)?;
        }

        let wanted_mlflow: BTreeMap<PathBuf, String> = discovered_mlflow.into_iter()
            .map(|(run, path)| (path, run))
            .collect();
        let gone: Vec<PathBuf> = self
            .mlflow
            .keys()
            .filter(|path| !wanted_mlflow.contains_key(*path))
            .cloned()
            .collect();
        for path in gone {
            if let Some(state) = self.mlflow.remove(&path) {
                self.runs.remove(&state.run);
                self.materialized.remove(&state.run);
            }
        }
        for (path, base_name) in wanted_mlflow {
            if !self.mlflow.contains_key(&path) {
                let name = unique_run_name(&self.runs, base_name);
                self.runs
                    .entry(name.clone())
                    .or_insert_with(|| Run::new(Backend::Mlflow, path.clone()));
                self.mlflow.insert(
                    path.clone(),
                    MlflowState {
                        run: name,
                        offsets: BTreeMap::new(),
                        params_seen: std::collections::BTreeSet::new(),
                    },
                );
            }
            let state = self.mlflow.get_mut(&path).expect("just ensured");
            let run_name = state.run.clone();
            let run = self.runs.get_mut(&run_name).expect("state has a run");
            drain_mlflow(&path, state, run, &run_name, &mut self.materialized, &mut report)?;
        }

        let wanted_wandb: BTreeMap<PathBuf, String> = discovered_wandb.into_iter()
            .map(|(run, path)| (path, run))
            .collect();
        let gone: Vec<PathBuf> = self
            .wandb
            .keys()
            .filter(|path| !wanted_wandb.contains_key(*path))
            .cloned()
            .collect();
        for path in gone {
            if let Some(state) = self.wandb.remove(&path) {
                self.runs.remove(&state.run);
                self.materialized.remove(&state.run);
            }
        }
        for (path, base_name) in wanted_wandb {
            if !self.wandb.contains_key(&path) {
                let name = unique_run_name(&self.runs, base_name);
                self.runs.entry(name.clone()).or_insert_with(|| {
                    Run::new(
                        Backend::Wandb,
                        path.parent().unwrap_or(&self.root).to_path_buf(),
                    )
                });
                self.wandb.insert(
                    path.clone(),
                    WandbState {
                        run: name,
                        offset: 0,
                        dead: false,
                    },
                );
                report.new_files += 1;
            }
            let state = self.wandb.get_mut(&path).expect("just ensured");
            let run_name = state.run.clone();
            let run = self.runs.get_mut(&run_name).expect("state has a run");
            drain_wandb(&path, state, run, &run_name, &mut self.materialized, &mut report)?;
        }

        // Cumulative loss accounting, from whichever backend suffered it.
        self.dropped_records += report.dropped_records;
        self.dead_files += report.dead_files;
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
        let Some(run) = self.runs.get(run_name) else {
            return Ok(None);
        };
        let backend = run.backend;
        let Some(series) = run.series.get(tag) else {
            return Ok(None);
        };
        if series.class != SeriesClass::Scalar {
            return Ok(None);
        }

        let points = match backend {
            Backend::Tfevents => self.materialize_tfevents(run_name, tag)?,
            Backend::Dvclive => {
                let Some((root, state)) = self
                    .dvclive
                    .iter()
                    .find(|(_, state)| state.run == run_name)
                else {
                    return Ok(None);
                };
                let path = root
                    .join("plots")
                    .join("metrics")
                    .join(format!("{tag}.tsv"));
                let frontier = state.offsets.get(&path).copied().unwrap_or(0);
                materialize_tsv(&path, frontier)?
            }
            Backend::Mlflow => {
                let Some((root, state)) = self
                    .mlflow
                    .iter()
                    .find(|(_, state)| state.run == run_name)
                else {
                    return Ok(None);
                };
                let path = root.join("metrics").join(tag);
                let frontier = state.offsets.get(&path).copied().unwrap_or(0);
                materialize_metric_lines(&path, frontier)?
            }
            Backend::Wandb => {
                let Some((path, state)) = self
                    .wandb
                    .iter()
                    .find(|(_, state)| state.run == run_name)
                else {
                    return Ok(None);
                };
                materialize_wandb(path, state.offset, tag)?
            }
        };

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

    /// The tfevents materialization body: a transient full re-read of the
    /// run's files up to each reader's committed frontier.
    fn materialize_tfevents(&self, run_name: &str, tag: &str) -> io::Result<Points> {
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
        Ok(points)
    }

    /// The materialized points of a series, if currently held.
    pub fn points(&self, run_name: &str, tag: &str) -> Option<&Points> {
        self.materialized
            .get(run_name)
            .and_then(|tags| tags.get(tag))
            .map(|entry| &entry.points)
    }

    /// Materializes one histogram series — the same transient full re-read
    /// up to the committed frontier as [`materialize`](Self::materialize),
    /// collecting normalized bucket snapshots instead of points.
    pub fn materialize_histograms(
        &mut self,
        run_name: &str,
        tag: &str,
    ) -> io::Result<Option<&HistogramSeries>> {
        self.clock += 1;
        if let Some(entry) = self
            .histograms
            .get_mut(run_name)
            .and_then(|tags| tags.get_mut(tag))
        {
            entry.touched = self.clock;
            return Ok(Some(&self.histograms[run_name][tag].series));
        }
        let Some(series) = self
            .runs
            .get(run_name)
            .and_then(|run| run.series.get(tag))
        else {
            return Ok(None);
        };
        if series.class != SeriesClass::Histogram {
            return Ok(None);
        }

        let mut collected = HistogramSeries::default();
        for (path, state) in &self.files {
            if state.run != run_name {
                continue;
            }
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
                            if value.tag == tag
                                && let Some(buckets) = value.histogram_buckets()
                            {
                                collected.push(HistogramSnapshot {
                                    step: event.step,
                                    wall: event.wall_time,
                                    buckets,
                                });
                            }
                        }
                    }
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

        let held: usize = self.histograms.values().map(BTreeMap::len).sum();
        if held >= HISTOGRAM_CAP
            && let Some((_, run, tag)) = self
                .histograms
                .iter()
                .flat_map(|(run, tags)| {
                    tags.iter()
                        .map(move |(tag, entry)| (entry.touched, run.clone(), tag.clone()))
                })
                .min()
            && let Some(tags) = self.histograms.get_mut(&run)
        {
            tags.remove(&tag);
            if tags.is_empty() {
                self.histograms.remove(&run);
            }
        }
        self.histograms
            .entry(run_name.to_owned())
            .or_default()
            .insert(
                tag.to_owned(),
                MaterializedHistograms {
                    series: collected,
                    touched: self.clock,
                },
            );
        Ok(Some(&self.histograms[run_name][tag].series))
    }

    /// The materialized histogram series, if currently held.
    pub fn histogram_series(&self, run_name: &str, tag: &str) -> Option<&HistogramSeries> {
        self.histograms
            .get(run_name)
            .and_then(|tags| tags.get(tag))
            .map(|entry| &entry.series)
    }
}

fn ingest(
    run: &mut Run,
    materialized: &mut BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
    histograms: &mut BTreeMap<String, BTreeMap<String, MaterializedHistograms>>,
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
        if series.class == SeriesClass::Histogram
            && let Some(entry) = histograms
                .get_mut(run_name)
                .and_then(|tags| tags.get_mut(value.tag.as_str()))
            && let Some(buckets) = value.histogram_buckets()
        {
            entry.series.push(HistogramSnapshot {
                step: event.step,
                wall: event.wall_time,
                buckets,
            });
        }
    }
}

/// Observes one scalar point from a text backend: summary, walls,
/// preemption accounting, and the materialized append — the same path
/// tfevents scalars take.
fn observe_scalar(
    run: &mut Run,
    materialized: &mut BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
    run_name: &str,
    tag: &str,
    point: PointStamp,
    report: &mut RefreshReport,
) {
    if point.wall != 0.0 {
        run.first_wall = Some(run.first_wall.map_or(point.wall, |wall| wall.min(point.wall)));
        run.last_wall = Some(run.last_wall.map_or(point.wall, |wall| wall.max(point.wall)));
    }
    let series = run
        .series
        .entry(tag.to_owned())
        .or_insert_with(|| Series {
            class: SeriesClass::Scalar,
            plugin: None,
            summary: SeriesSummary::default(),
        });
    if series.summary.observe(point) {
        run.preemptions += 1;
        report.preemptions += 1;
    }
    report.new_points += 1;
    if let Some(entry) = materialized
        .get_mut(run_name)
        .and_then(|tags| tags.get_mut(tag))
    {
        entry.points.push(point);
    }
}

/// A run name not yet taken — appends `~2`, `~3`, … on collision (two
/// MLflow runs may share a `run_name`; different backends may share a
/// directory).
fn unique_run_name(runs: &BTreeMap<String, Run>, base: String) -> String {
    if !runs.contains_key(&base) {
        return base;
    }
    (2..)
        .map(|counter| format!("{base}~{counter}"))
        .find(|candidate| !runs.contains_key(candidate))
        .expect("the counter is unbounded")
}

/// Reads the complete lines between `offset` and end-of-file. The trailing
/// incomplete line — a torn write from a live logger — stays unconsumed,
/// exactly like a torn tfevents record: a state, not an error. Returns
/// `(text, new offset, mtime)`; `None` when the file is gone.
fn read_new_lines(path: &Path, offset: u64) -> io::Result<Option<(String, u64, Option<SystemTime>)>> {
    use std::io::{Read as _, Seek as _, SeekFrom};
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err),
    };
    let metadata = file.metadata()?;
    let mtime = metadata.modified().ok();
    let len = metadata.len();
    if len <= offset {
        return Ok(Some((String::new(), offset, mtime)));
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = Vec::with_capacity((len - offset) as usize);
    file.take(len - offset).read_to_end(&mut buf)?;
    let complete = buf
        .iter()
        .rposition(|&byte| byte == b'\n')
        .map_or(0, |position| position + 1);
    buf.truncate(complete);
    let text = String::from_utf8_lossy(&buf).into_owned();
    Ok(Some((text, offset + complete as u64, mtime)))
}

/// Collects the files under `base`, as `(absolute path, `/`-joined relative
/// name)`, sorted. Shallow recursion with a depth cap.
fn walk_files(base: &Path) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let mut stack = vec![(base.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
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
                if depth < 8 {
                    stack.push((path, depth + 1));
                }
            } else {
                let relative = path
                    .strip_prefix(base)
                    .map(|relative| {
                        relative
                            .to_string_lossy()
                            .replace(std::path::MAIN_SEPARATOR, "/")
                    })
                    .unwrap_or_else(|_| name.to_owned());
                out.push((path, relative));
            }
        }
    }
    out.sort();
    out
}

fn convert_param(value: vertov_formats::ParamValue) -> HparamValue {
    match value {
        vertov_formats::ParamValue::Number(number) => HparamValue::F64(number),
        vertov_formats::ParamValue::Bool(flag) => HparamValue::Bool(flag),
        vertov_formats::ParamValue::Text(text) => HparamValue::String(text),
    }
}

/// Materializes a dvclive TSV up to `frontier`: header defines the schema,
/// rows become points, preemption applies on push.
fn materialize_tsv(path: &Path, frontier: u64) -> io::Result<Points> {
    let mut points = Points::default();
    let text = read_prefix(path, frontier)?;
    let mut lines = text.lines();
    let Some(schema) = lines
        .next()
        .and_then(vertov_formats::dvclive::TsvSchema::from_header)
    else {
        return Ok(points);
    };
    for line in lines {
        if let Some(row) = schema.parse_row(line) {
            points.push(PointStamp {
                step: row.step,
                wall: row.wall,
                value: row.value,
            });
        }
    }
    Ok(points)
}

/// Materializes an MLflow metric file up to `frontier`.
fn materialize_metric_lines(path: &Path, frontier: u64) -> io::Result<Points> {
    let mut points = Points::default();
    for line in read_prefix(path, frontier)?.lines() {
        if let Some(metric) = vertov_formats::mlflow::parse_metric_line(line) {
            points.push(PointStamp {
                step: metric.step,
                wall: metric.wall,
                value: metric.value,
            });
        }
    }
    Ok(points)
}

/// The first `frontier` bytes of a file as (lossy) text; empty when the
/// frontier is zero or the file is gone.
fn read_prefix(path: &Path, frontier: u64) -> io::Result<String> {
    use std::io::Read as _;
    if frontier == 0 {
        return Ok(String::new());
    }
    let file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(String::new()),
        Err(err) => return Err(err),
    };
    let mut buf = Vec::with_capacity(frontier as usize);
    file.take(frontier).read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Resets a run whose backing files shrank or were replaced: the honest
/// recovery is a full re-read this same pass.
fn reset_run(
    run: &mut Run,
    run_name: &str,
    materialized: &mut BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
) {
    run.series.clear();
    run.preemptions = 0;
    run.first_wall = None;
    run.last_wall = None;
    materialized.remove(run_name);
}

/// One dvclive tick: params on change, then every metrics TSV drained from
/// its committed offset.
fn drain_dvclive(
    root: &Path,
    state: &mut DvcliveState,
    run: &mut Run,
    run_name: &str,
    materialized: &mut BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
    report: &mut RefreshReport,
) -> io::Result<()> {
    let params = root.join("params.yaml");
    if let Ok(metadata) = std::fs::metadata(&params) {
        let mtime = metadata.modified().ok();
        if mtime != state.params_mtime {
            state.params_mtime = mtime;
            if let Ok(text) = std::fs::read_to_string(&params) {
                run.hparams = vertov_formats::dvclive::parse_params_yaml(&text)
                    .into_iter()
                    .map(|(key, value)| (key, convert_param(value)))
                    .collect();
            }
        }
    }

    let files = walk_files(&root.join("plots").join("metrics"));
    let shrank = files.iter().any(|(path, _)| {
        state.offsets.get(path).is_some_and(|&offset| {
            std::fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0) < offset
        })
    });
    if shrank {
        reset_run(run, run_name, materialized);
        state.offsets.clear();
        state.schemas.clear();
    }

    for (path, relative) in files {
        let Some(tag) = relative.strip_suffix(".tsv").map(str::to_owned) else {
            continue;
        };
        let offset = state.offsets.get(&path).copied().unwrap_or(0);
        let Some((text, new_offset, mtime)) = read_new_lines(&path, offset)? else {
            continue;
        };
        if let Some(mtime) = mtime {
            run.last_write = Some(run.last_write.map_or(mtime, |known| known.max(mtime)));
        }
        if new_offset == offset {
            continue;
        }
        let mut lines = text.lines();
        // The first line of the file is the header; it defines the schema
        // and is re-read whenever we start from the top.
        let schema = if offset == 0 {
            match lines.next().and_then(vertov_formats::dvclive::TsvSchema::from_header) {
                Some(schema) => {
                    state.schemas.insert(path.clone(), schema.clone());
                    schema
                }
                None => {
                    // Not a metrics TSV after all; never look again.
                    state.offsets.insert(path, new_offset);
                    continue;
                }
            }
        } else {
            match state.schemas.get(&path) {
                Some(schema) => schema.clone(),
                None => continue,
            }
        };
        for line in lines {
            match schema.parse_row(line) {
                Some(row) => observe_scalar(
                    run,
                    materialized,
                    run_name,
                    &tag,
                    PointStamp {
                        step: row.step,
                        wall: row.wall,
                        value: row.value,
                    },
                    report,
                ),
                None => {
                    report.dropped_records += 1;
                }
            }
        }
        state.offsets.insert(path, new_offset);
    }
    Ok(())
}

/// One MLflow tick: new param files once (they are immutable), then every
/// metric file drained from its committed offset.
fn drain_mlflow(
    root: &Path,
    state: &mut MlflowState,
    run: &mut Run,
    run_name: &str,
    materialized: &mut BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
    report: &mut RefreshReport,
) -> io::Result<()> {
    for (path, relative) in walk_files(&root.join("params")) {
        if state.params_seen.insert(path.clone())
            && let Ok(text) = std::fs::read_to_string(&path)
        {
            run.hparams
                .insert(relative, convert_param(vertov_formats::mlflow::parse_param(&text)));
        }
    }

    let files = walk_files(&root.join("metrics"));
    let shrank = files.iter().any(|(path, _)| {
        state.offsets.get(path).is_some_and(|&offset| {
            std::fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0) < offset
        })
    });
    if shrank {
        reset_run(run, run_name, materialized);
        state.offsets.clear();
    }

    for (path, tag) in files {
        let offset = state.offsets.get(&path).copied().unwrap_or(0);
        let Some((text, new_offset, mtime)) = read_new_lines(&path, offset)? else {
            continue;
        };
        if let Some(mtime) = mtime {
            run.last_write = Some(run.last_write.map_or(mtime, |known| known.max(mtime)));
        }
        if new_offset == offset {
            continue;
        }
        for line in text.lines() {
            match vertov_formats::mlflow::parse_metric_line(line) {
                Some(metric) => observe_scalar(
                    run,
                    materialized,
                    run_name,
                    &tag,
                    PointStamp {
                        step: metric.step,
                        wall: metric.wall,
                        value: metric.value,
                    },
                    report,
                ),
                None => {
                    report.dropped_records += 1;
                }
            }
        }
        state.offsets.insert(path, new_offset);
    }
    Ok(())
}

/// One wandb tick: validate the header once, then parse the new bytes past
/// the committed offset into records — history rows become scalar points,
/// config updates become hparams, the exit record marks a clean finish.
fn drain_wandb(
    path: &Path,
    state: &mut WandbState,
    run: &mut Run,
    run_name: &str,
    materialized: &mut BTreeMap<String, BTreeMap<String, MaterializedSeries>>,
    report: &mut RefreshReport,
) -> io::Result<()> {
    use vertov_formats::wandb;
    if state.dead {
        return Ok(());
    }
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    if let Ok(mtime) = metadata.modified() {
        run.last_write = Some(run.last_write.map_or(mtime, |known| known.max(mtime)));
    }
    if metadata.len() < state.offset {
        // Rewritten shorter: honest recovery is a full re-read.
        reset_run(run, run_name, materialized);
        state.offset = 0;
    }
    if state.offset == 0 {
        if metadata.len() < wandb::HEADER_LEN {
            return Ok(());
        }
        let mut header = [0u8; wandb::HEADER_LEN as usize];
        {
            use std::io::Read as _;
            let mut file = File::open(path)?;
            file.read_exact(&mut header)?;
        }
        if wandb::check_header(&header).is_err() {
            state.dead = true;
            report.dead_files += 1;
            return Ok(());
        }
        state.offset = wandb::HEADER_LEN;
    }
    if metadata.len() <= state.offset {
        return Ok(());
    }

    let tail = {
        use std::io::{Read as _, Seek as _, SeekFrom};
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(state.offset))?;
        let mut buf = Vec::with_capacity((metadata.len() - state.offset) as usize);
        file.take(metadata.len() - state.offset).read_to_end(&mut buf)?;
        buf
    };
    let (payloads, committed) = wandb::read_records(&tail, state.offset);
    for payload in payloads {
        match wandb::parse_record(&payload) {
            Some(wandb::WandbRecord::History { step, wall, values }) => {
                for (key, value) in values {
                    observe_scalar(
                        run,
                        materialized,
                        run_name,
                        &key,
                        PointStamp { step, wall, value },
                        report,
                    );
                }
            }
            Some(wandb::WandbRecord::Config(updates)) => {
                for (key, value) in updates {
                    run.hparams.insert(key, convert_param(value));
                }
            }
            Some(wandb::WandbRecord::Exit) => run.finished = Some(true),
            Some(wandb::WandbRecord::Other) => {}
            None => report.dropped_records += 1,
        }
    }
    state.offset = committed;
    Ok(())
}

/// Materializes one wandb series up to `frontier`.
fn materialize_wandb(path: &Path, frontier: u64, tag: &str) -> io::Result<Points> {
    use vertov_formats::wandb;
    let mut points = Points::default();
    if frontier <= wandb::HEADER_LEN {
        return Ok(points);
    }
    let bytes = {
        use std::io::Read as _;
        let file = match File::open(path) {
            Ok(file) => file,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(points),
            Err(err) => return Err(err),
        };
        let mut buf = Vec::with_capacity(frontier as usize);
        file.take(frontier).read_to_end(&mut buf)?;
        buf
    };
    if bytes.len() < wandb::HEADER_LEN as usize {
        return Ok(points);
    }
    let (payloads, _) = wandb::read_records(&bytes[wandb::HEADER_LEN as usize..], wandb::HEADER_LEN);
    for payload in payloads {
        if let Some(wandb::WandbRecord::History { step, wall, values }) =
            wandb::parse_record(&payload)
        {
            for (key, value) in values {
                if key == tag {
                    points.push(PointStamp { step, wall, value });
                }
            }
        }
    }
    Ok(points)
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

/// Everything one walk of the root found, per backend.
#[derive(Default)]
struct Discovery {
    /// tfevents: `(run name, file path)`, sorted — a run is a directory
    /// containing at least one file whose basename contains `tfevents`.
    tfevents: Vec<(String, PathBuf)>,
    /// dvclive: `(run name, dvclive dir)` — a directory containing a
    /// `dvclive` child with `metrics.json` or `plots/` marks a root, named
    /// by the *parent*'s root-relative path.
    dvclive: Vec<(String, PathBuf)>,
    /// MLflow: `(run name, run dir)` — a directory whose `meta.yaml`
    /// carries a run id; named by its recorded `run_name`, falling back to
    /// the root-relative path.
    mlflow: Vec<(String, PathBuf)>,
    /// wandb: `(run name, .wandb file)` — a `run-*.wandb` transaction log,
    /// named by its directory's root-relative path.
    wandb: Vec<(String, PathBuf)>,
}

/// Walks `root` once, classifying what it finds. Mixed roots just work:
/// each directory declares itself independently.
fn discover(root: &Path) -> io::Result<Discovery> {
    let mut found = Discovery::default();
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
                // Directory symlinks are skipped: they alias runs already
                // found by their real path (wandb's `latest-run`) and are
                // the vector for walk loops. Real loop-tolerant following
                // can come later; aliasing bugs cannot.
                if entry.file_type().is_ok_and(|kind| kind.is_symlink()) {
                    continue;
                }
                if name == "dvclive"
                    && (path.join("metrics.json").is_file() || path.join("plots").is_dir())
                {
                    found.dvclive.push((run_name(root, &dir), path));
                    continue;
                }
                if depth < 32 {
                    stack.push((path, depth + 1));
                }
            } else if name.contains("tfevents") {
                found.tfevents.push((run_name(root, &dir), path));
            } else if name.starts_with("run-") && name.ends_with(".wandb") {
                found.wandb.push((run_name(root, &dir), path));
            } else if name == "meta.yaml"
                && let Ok(text) = std::fs::read_to_string(&path)
            {
                let meta = vertov_formats::mlflow::parse_meta(&text);
                if meta.run_id.is_some() {
                    let run = meta.run_name.unwrap_or_else(|| run_name(root, &dir));
                    found.mlflow.push((run, dir.clone()));
                }
            }
        }
    }
    found.tfevents.sort();
    found.dvclive.sort();
    found.mlflow.sort();
    found.wandb.sort();
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
