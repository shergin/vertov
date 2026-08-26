//! Logdir discovery and scalar loading — the spike-sized ancestor of the
//! future `vertov-model` crate.
//!
//! RustBoard's run semantics: a run is a directory containing at least one
//! file whose basename contains `tfevents`; its name is the directory's path
//! relative to the root (`.` for the root itself). Files within a run are
//! read in filename order into shared per-tag series. Readers stay open
//! across polls — the file offset is the resume state — so a tail tick is an
//! incremental read, not a re-parse.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use tfevents::{EventFileReader, EventPayload, ReadEventError};

/// One scalar series: parallel step/value columns, in arrival order.
/// Non-finite values stay in place — a NaN is a gap, never interpolated.
#[derive(Default)]
pub struct ScalarSeries {
    pub steps: Vec<f64>,
    pub values: Vec<f64>,
}

/// Watches a logdir for scalar data matching a tag filter (substring match).
///
/// [`Watcher::poll`] rescans for new runs and files, drains every open
/// reader, and reports how many points arrived — the caller repaints only
/// when that is nonzero.
pub struct Watcher {
    root: PathBuf,
    filter: String,
    /// run name → tag → series, for every tag matching the filter.
    pub runs: BTreeMap<String, BTreeMap<String, ScalarSeries>>,
    /// Every scalar tag seen anywhere, matching or not — for "no such tag"
    /// suggestions.
    pub seen_tags: BTreeSet<String>,
    /// Records lost to corruption or undecodable bytes. Shown, not hidden.
    pub dropped_records: usize,
    /// Files whose framing died (bad length CRC) or that stopped reading on
    /// an I/O error. Their valid prefix is retained.
    pub dead_files: usize,
    readers: BTreeMap<PathBuf, OpenFile>,
}

struct OpenFile {
    run: String,
    reader: EventFileReader<File>,
    dead: bool,
}

impl Watcher {
    pub fn new(root: impl Into<PathBuf>, filter: impl Into<String>) -> Watcher {
        Watcher {
            root: root.into(),
            filter: filter.into(),
            runs: BTreeMap::new(),
            seen_tags: BTreeSet::new(),
            dropped_records: 0,
            dead_files: 0,
            readers: BTreeMap::new(),
        }
    }

    /// Rescans the root and drains all live readers. Returns the number of
    /// points appended. A missing root is fine (the trainer may not have
    /// started yet); it just yields nothing.
    pub fn poll(&mut self) -> io::Result<usize> {
        for (run, path) in discover(&self.root)? {
            if self.readers.contains_key(&path) {
                continue;
            }
            let file = match File::open(&path) {
                Ok(file) => file,
                // The file vanished between discovery and open; next poll
                // will see the current state.
                Err(err) if err.kind() == ErrorKind::NotFound => continue,
                Err(err) => return Err(err),
            };
            self.readers.insert(
                path,
                OpenFile {
                    run,
                    reader: EventFileReader::new(file),
                    dead: false,
                },
            );
        }

        let mut appended = 0;
        // BTreeMap order is lexicographic by path: files within a run drain
        // in filename order, matching tfevents' timestamped naming.
        for open in self.readers.values_mut() {
            if open.dead {
                continue;
            }
            loop {
                match open.reader.next_event() {
                    Ok(event) => {
                        let EventPayload::Summary(values) = &event.payload else {
                            continue;
                        };
                        for value in values {
                            let Some(scalar) = value.scalar() else {
                                continue;
                            };
                            self.seen_tags.insert(value.tag.clone());
                            if !value.tag.contains(self.filter.as_str()) {
                                continue;
                            }
                            let series = self
                                .runs
                                .entry(open.run.clone())
                                .or_default()
                                .entry(value.tag.clone())
                                .or_default();
                            series.steps.push(event.step as f64);
                            series.values.push(scalar);
                            appended += 1;
                        }
                    }
                    Err(ReadEventError::Truncated) => break,
                    Err(ReadEventError::Corrupt { .. } | ReadEventError::Malformed { .. }) => {
                        self.dropped_records += 1;
                    }
                    Err(ReadEventError::BadLengthCrc { .. } | ReadEventError::Io(_)) => {
                        open.dead = true;
                        self.dead_files += 1;
                        break;
                    }
                }
            }
        }
        Ok(appended)
    }
}

/// Walks `root` for tfevents files, returning `(run name, file path)` pairs.
fn discover(root: &Path) -> io::Result<Vec<(String, PathBuf)>> {
    let mut found = Vec::new();
    // Depth cap in place of true symlink-loop detection (Phase 2 grows the
    // real inventory); 32 levels is beyond any sane logdir.
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            // A missing root means the trainer hasn't started; a dir that
            // vanished mid-walk means a run was deleted. Both are states,
            // not errors.
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
        relative.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/")
    }
}
