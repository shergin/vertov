//! The MLflow file store: `mlruns/<experiment>/<run>/` with
//! `metrics/<name>` as text lines of `timestamp value step` (milliseconds,
//! value, step), `params/<name>` as one-line values, and `meta.yaml`
//! carrying the run's name and lifecycle.

use crate::ParamValue;

/// One parsed metric line.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MetricLine {
    /// Global step.
    pub step: i64,
    /// Seconds since the Unix epoch.
    pub wall: f64,
    /// The value; non-finite parses pass through honestly.
    pub value: f64,
}

/// Parses one `timestamp value step` line. `None` for lines that do not
/// fit — counted by the caller as visible data loss.
pub fn parse_metric_line(line: &str) -> Option<MetricLine> {
    let mut fields = line.split_whitespace();
    let millis: f64 = fields.next()?.parse().ok()?;
    let value: f64 = fields.next()?.parse().ok()?;
    let step: i64 = fields.next()?.parse().ok()?;
    if fields.next().is_some() {
        return None;
    }
    Some(MetricLine {
        step,
        wall: millis / 1000.0,
        value,
    })
}

/// A `params/<name>` file: the whole (single-line) content is the value.
pub fn parse_param(content: &str) -> ParamValue {
    ParamValue::parse(content.trim())
}

/// What `meta.yaml` tells a viewer about the run.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct RunMeta {
    /// The human-readable run name, when recorded.
    pub run_name: Option<String>,
    /// The run id, which doubles as identity proof: a `meta.yaml` without
    /// one is an experiment's, not a run's.
    pub run_id: Option<String>,
    /// The lifecycle status word (`RUNNING`, `FINISHED`, `FAILED`, …) or
    /// numeric code, verbatim.
    pub status: Option<String>,
}

/// Reads the fields vertov uses from a run's `meta.yaml` (flat
/// `key: value` YAML — the file store writes nothing fancier).
pub fn parse_meta(text: &str) -> RunMeta {
    let mut meta = RunMeta::default();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "run_name" => meta.run_name = Some(value.to_owned()),
            "run_id" | "run_uuid" => meta.run_id = Some(value.to_owned()),
            "status" => meta.status = Some(value.to_owned()),
            _ => {}
        }
    }
    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_lines() {
        let line = parse_metric_line("1700000001500 0.75 12").unwrap();
        assert_eq!(line.step, 12);
        assert_eq!(line.wall, 1_700_000_001.5);
        assert_eq!(line.value, 0.75);
        assert!(parse_metric_line("1700000001500 nan 3").unwrap().value.is_nan());
        assert_eq!(parse_metric_line("only two"), None);
        assert_eq!(parse_metric_line("1 2 3 4"), None);
        assert_eq!(parse_metric_line(""), None);
    }

    #[test]
    fn meta_yaml() {
        let meta = parse_meta(
            "artifact_uri: file:///tmp/mlruns/0/abc/artifacts\nrun_id: abc123\nrun_name: brave-owl-7\nstatus: FINISHED\n",
        );
        assert_eq!(meta.run_name.as_deref(), Some("brave-owl-7"));
        assert_eq!(meta.run_id.as_deref(), Some("abc123"));
        assert_eq!(meta.status.as_deref(), Some("FINISHED"));
        // An experiment-level meta.yaml has no run id.
        assert_eq!(parse_meta("experiment_id: 0\nname: Default\n").run_id, None);
    }

    #[test]
    fn params() {
        assert_eq!(parse_param("0.01\n"), ParamValue::Number(0.01));
        assert_eq!(parse_param("adam"), ParamValue::Text("adam".into()));
    }
}
