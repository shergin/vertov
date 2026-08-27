//! The on-disk summary cache: instant tables over hundreds of runs without
//! re-reading gigabytes on startup.
//!
//! One small file per project under `$XDG_CACHE_HOME/vertov/` (or
//! `~/.cache/vertov/`) maps each event file's `(path, size, mtime)` to its
//! run's serialized summaries and the final committed byte offset. On a warm
//! start an unchanged logdir is a metadata walk; a grown file resumes from
//! its cached offset; a shrunk or replaced file invalidates its whole run.
//!
//! The cache lives outside every logdir and is disposable by design
//! (observer principle): the format is versioned by magic, any parse
//! surprise discards the whole file, and deleting it is always safe.
//! Hand-rolled binary, no serde — the field set is small and frozen here.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tfevents::HparamValue;

use crate::moments::Moments;
use crate::project::Run;
use crate::series::{PointStamp, SegmentSummary, Series, SeriesClass, SeriesSummary};

const MAGIC: &[u8; 8] = b"VTVCACH1";

/// A cached file's identity and resume state.
pub(crate) struct CachedFile {
    pub run: String,
    pub size: u64,
    pub mtime: SystemTime,
    pub offset: u64,
}

/// Everything a warm start needs.
pub(crate) struct CachedProject {
    pub runs: BTreeMap<String, Run>,
    pub files: BTreeMap<PathBuf, CachedFile>,
}

/// Where this project's cache file lives: inside `dir` when overridden,
/// else `$XDG_CACHE_HOME/vertov/` or `~/.cache/vertov/`. `None` when no
/// cache directory can be determined (no `$HOME`).
pub(crate) fn cache_path(root: &Path, dir: Option<&Path>) -> Option<PathBuf> {
    let base = match dir {
        Some(dir) => dir.to_path_buf(),
        None => std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?
            .join("vertov"),
    };
    let canonical = std::path::absolute(root).unwrap_or_else(|_| root.to_path_buf());
    Some(base.join(format!(
        "{:016x}.vcache",
        fnv1a(canonical.as_os_str().as_encoded_bytes())
    )))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Serializes and writes the cache for `root`. `files` carries
/// `(path, run name, committed offset)` for every live (non-dead) file;
/// sizes and mtimes are taken from the filesystem now.
pub(crate) fn save(
    root: &Path,
    dir: Option<&Path>,
    runs: &BTreeMap<String, Run>,
    files: &[(PathBuf, String, u64)],
) -> io::Result<()> {
    let Some(path) = cache_path(root, dir) else {
        return Ok(());
    };
    let mut by_run: BTreeMap<&String, Vec<(&PathBuf, u64)>> = BTreeMap::new();
    for (file, run, offset) in files {
        by_run.entry(run).or_default().push((file, *offset));
    }

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    let cached_runs: Vec<_> = runs
        .iter()
        .filter(|(name, _)| by_run.contains_key(name))
        .collect();
    put_u32(&mut out, cached_runs.len() as u32);
    for (name, run) in cached_runs {
        put_str(&mut out, name);
        put_str(&mut out, &run.dir.to_string_lossy());
        put_u32(&mut out, run.hparams.len() as u32);
        for (key, value) in &run.hparams {
            put_str(&mut out, key);
            match value {
                HparamValue::F64(v) => {
                    out.push(0);
                    put_f64(&mut out, *v);
                }
                HparamValue::String(v) => {
                    out.push(1);
                    put_str(&mut out, v);
                }
                HparamValue::Bool(v) => {
                    out.push(2);
                    out.push(u8::from(*v));
                }
            }
        }
        put_opt_f64(&mut out, run.first_wall);
        put_opt_f64(&mut out, run.last_wall);
        put_u64(&mut out, run.preemptions);

        let entries = &by_run[name];
        put_u32(&mut out, entries.len() as u32);
        for (file, offset) in entries {
            let metadata = std::fs::metadata(file)?;
            let mtime = metadata.modified()?;
            put_str(&mut out, &file.to_string_lossy());
            put_u64(&mut out, metadata.len());
            put_mtime(&mut out, mtime);
            put_u64(&mut out, *offset);
        }

        put_u32(&mut out, run.series.len() as u32);
        for (tag, series) in &run.series {
            put_str(&mut out, tag);
            out.push(class_code(series.class));
            match &series.plugin {
                Some(plugin) => {
                    out.push(1);
                    put_str(&mut out, plugin);
                }
                None => out.push(0),
            }
            put_u32(&mut out, series.summary.segments.len() as u32);
            for segment in &series.summary.segments {
                put_point(&mut out, segment.first);
                put_point(&mut out, segment.last);
                put_f64(&mut out, segment.min);
                put_f64(&mut out, segment.max);
                let (count, mean, m2) = segment.moments.raw();
                put_u64(&mut out, count);
                put_f64(&mut out, mean);
                put_f64(&mut out, m2);
                put_u64(&mut out, segment.count);
                put_u64(&mut out, segment.non_finite);
                match segment.preempted_at {
                    Some(step) => {
                        out.push(1);
                        put_u64(&mut out, step as u64);
                    }
                    None => out.push(0),
                }
            }
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Write-then-rename so a crash mid-save never leaves a torn cache.
    let staging = path.with_extension("vcache.tmp");
    std::fs::write(&staging, &out)?;
    std::fs::rename(&staging, &path)
}

/// Loads the cache for `root`. `None` on any surprise — a missing, stale-
/// versioned, or torn cache simply means a cold start.
pub(crate) fn load(root: &Path, dir: Option<&Path>) -> Option<CachedProject> {
    let bytes = std::fs::read(cache_path(root, dir)?).ok()?;
    let mut cursor = Cursor {
        buf: &bytes,
        pos: 0,
    };
    if cursor.take(8)? != MAGIC {
        return None;
    }
    let mut runs = BTreeMap::new();
    let mut files = BTreeMap::new();
    for _ in 0..cursor.u32()? {
        let name = cursor.string()?;
        let dir = PathBuf::from(cursor.string()?);
        let mut hparams = BTreeMap::new();
        for _ in 0..cursor.u32()? {
            let key = cursor.string()?;
            let value = match cursor.byte()? {
                0 => HparamValue::F64(cursor.f64()?),
                1 => HparamValue::String(cursor.string()?),
                2 => HparamValue::Bool(cursor.byte()? != 0),
                _ => return None,
            };
            hparams.insert(key, value);
        }
        let first_wall = cursor.opt_f64()?;
        let last_wall = cursor.opt_f64()?;
        let preemptions = cursor.u64()?;

        for _ in 0..cursor.u32()? {
            let path = PathBuf::from(cursor.string()?);
            let size = cursor.u64()?;
            let mtime = cursor.mtime()?;
            let offset = cursor.u64()?;
            files.insert(
                path,
                CachedFile {
                    run: name.clone(),
                    size,
                    mtime,
                    offset,
                },
            );
        }

        let mut series_map = BTreeMap::new();
        for _ in 0..cursor.u32()? {
            let tag = cursor.string()?;
            let class = class_from(cursor.byte()?)?;
            let plugin = match cursor.byte()? {
                0 => None,
                1 => Some(cursor.string()?),
                _ => return None,
            };
            let mut segments = Vec::new();
            for _ in 0..cursor.u32()? {
                let first = cursor.point()?;
                let last = cursor.point()?;
                let min = cursor.f64()?;
                let max = cursor.f64()?;
                let moments = Moments::from_raw(cursor.u64()?, cursor.f64()?, cursor.f64()?);
                let count = cursor.u64()?;
                let non_finite = cursor.u64()?;
                let preempted_at = match cursor.byte()? {
                    0 => None,
                    1 => Some(cursor.u64()? as i64),
                    _ => return None,
                };
                segments.push(SegmentSummary {
                    first,
                    last,
                    min,
                    max,
                    moments,
                    count,
                    non_finite,
                    preempted_at,
                });
            }
            series_map.insert(
                tag,
                Series {
                    class,
                    plugin,
                    summary: SeriesSummary { segments },
                },
            );
        }
        runs.insert(
            name.clone(),
            Run {
                // The cache stores tfevents state only; other backends
                // re-read cold (their files are small text).
                backend: crate::project::Backend::Tfevents,
                dir,
                hparams,
                series: series_map,
                first_wall,
                last_wall,
                last_write: None,
                preemptions,
            },
        );
    }
    if cursor.pos != bytes.len() {
        return None;
    }
    Some(CachedProject { runs, files })
}

fn class_code(class: SeriesClass) -> u8 {
    match class {
        SeriesClass::Scalar => 0,
        SeriesClass::Histogram => 1,
        SeriesClass::Image => 2,
        SeriesClass::Text => 3,
        _ => 4,
    }
}

fn class_from(code: u8) -> Option<SeriesClass> {
    Some(match code {
        0 => SeriesClass::Scalar,
        1 => SeriesClass::Histogram,
        2 => SeriesClass::Image,
        3 => SeriesClass::Text,
        4 => SeriesClass::Unknown,
        _ => return None,
    })
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn put_opt_f64(out: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(value) => {
            out.push(1);
            put_f64(out, value);
        }
        None => out.push(0),
    }
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn put_mtime(out: &mut Vec<u8>, mtime: SystemTime) {
    let since = mtime
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    put_u64(out, since.as_secs());
    put_u32(out, since.subsec_nanos());
}

fn put_point(out: &mut Vec<u8>, point: PointStamp) {
    put_u64(out, point.step as u64);
    put_f64(out, point.wall);
    put_f64(out, point.value);
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        let slice = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    fn byte(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(self.u64()?))
    }

    fn opt_f64(&mut self) -> Option<Option<f64>> {
        match self.byte()? {
            0 => Some(None),
            1 => Some(Some(self.f64()?)),
            _ => None,
        }
    }

    fn string(&mut self) -> Option<String> {
        let len = self.u32()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).ok()
    }

    fn mtime(&mut self) -> Option<SystemTime> {
        let secs = self.u64()?;
        let nanos = self.u32()?;
        UNIX_EPOCH.checked_add(Duration::new(secs, nanos))
    }

    fn point(&mut self) -> Option<PointStamp> {
        Some(PointStamp {
            step: self.u64()? as i64,
            wall: self.f64()?,
            value: self.f64()?,
        })
    }
}
