//! dvclive: the friendliest format in the field. Metric history lives in
//! `dvclive/plots/metrics/<name>.tsv` — a header line then one row per
//! step, append-only and therefore tailable. Parameters live in
//! `dvclive/params.yaml` as a small YAML document.

use crate::ParamValue;

/// A TSV file's column layout, from its header line.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TsvSchema {
    /// Column index of the millisecond timestamp, when present.
    timestamp: Option<usize>,
    /// Column index of the step, when present (dvclive omits it for
    /// single-point metrics).
    step: Option<usize>,
    /// Column index of the value: the last column, whatever its name.
    value: usize,
}

/// One parsed metric row.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TsvRow {
    /// Global step (0 when the file has no step column).
    pub step: i64,
    /// Seconds since the Unix epoch, 0.0 when absent.
    pub wall: f64,
    /// The value; non-finite parses (`nan`, `inf`) pass through honestly.
    pub value: f64,
}

impl TsvSchema {
    /// Reads the layout from the header line: columns named `timestamp` and
    /// `step` are positional metadata, and the last column carries the
    /// value. `None` when the line cannot be a dvclive metrics header.
    pub fn from_header(header: &str) -> Option<TsvSchema> {
        let columns: Vec<&str> = header.trim_end().split('\t').collect();
        if columns.is_empty() || columns.iter().any(|column| column.is_empty()) {
            return None;
        }
        Some(TsvSchema {
            timestamp: columns.iter().position(|column| *column == "timestamp"),
            step: columns.iter().position(|column| *column == "step"),
            value: columns.len() - 1,
        })
    }

    /// Parses one data row. `None` for rows that do not fit the schema —
    /// the caller counts those as visible data loss, not silence.
    pub fn parse_row(&self, line: &str) -> Option<TsvRow> {
        let fields: Vec<&str> = line.trim_end().split('\t').collect();
        let value = fields.get(self.value)?.trim().parse::<f64>().ok()?;
        let step = match self.step {
            Some(index) => fields.get(index)?.trim().parse::<i64>().ok()?,
            None => 0,
        };
        let wall = match self.timestamp {
            Some(index) => {
                let millis = fields.get(index)?.trim().parse::<f64>().ok()?;
                millis / 1000.0
            }
            None => 0.0,
        };
        Some(TsvRow { step, wall, value })
    }
}

/// Parses `params.yaml` — the small subset dvclive and hand-edited files
/// actually use: top-level `key: value` scalars, plus one level of nesting
/// flattened to `parent.child`. Lists, deeper nesting, and YAML exotica are
/// skipped, not guessed at.
pub fn parse_params_yaml(text: &str) -> Vec<(String, ParamValue)> {
    let mut params = Vec::new();
    let mut parent: Option<String> = None;
    for line in text.lines() {
        let trimmed = line.trim_end();
        let stripped = trimmed.trim_start();
        if stripped.is_empty() || stripped.starts_with('#') || stripped.starts_with('-') {
            continue;
        }
        let indent = trimmed.len() - stripped.len();
        let Some((key, value)) = stripped.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || key.contains(' ') {
            continue;
        }
        if indent == 0 {
            if value.is_empty() {
                parent = Some(key.to_owned());
            } else {
                parent = None;
                params.push((key.to_owned(), ParamValue::parse(value)));
            }
        } else if let Some(parent) = &parent
            && !value.is_empty()
        {
            params.push((format!("{parent}.{key}"), ParamValue::parse(value)));
        }
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_with_timestamp_and_step() {
        let schema = TsvSchema::from_header("timestamp\tstep\tloss").unwrap();
        let row = schema.parse_row("1700000000500\t3\t0.25").unwrap();
        assert_eq!(row.step, 3);
        assert_eq!(row.wall, 1_700_000_000.5);
        assert_eq!(row.value, 0.25);
    }

    #[test]
    fn schema_without_timestamp() {
        let schema = TsvSchema::from_header("step\tacc").unwrap();
        let row = schema.parse_row("7\t0.9").unwrap();
        assert_eq!((row.step, row.wall, row.value), (7, 0.0, 0.9));
    }

    #[test]
    fn malformed_rows_are_none() {
        let schema = TsvSchema::from_header("timestamp\tstep\tloss").unwrap();
        assert_eq!(schema.parse_row("not\ta\tnumber"), None);
        assert_eq!(schema.parse_row(""), None);
        assert_eq!(schema.parse_row("12"), None);
    }

    #[test]
    fn nan_values_pass_through() {
        let schema = TsvSchema::from_header("step\tloss").unwrap();
        assert!(schema.parse_row("1\tnan").unwrap().value.is_nan());
    }

    #[test]
    fn params_yaml_subset() {
        let params = parse_params_yaml(
            "lr: 0.001\noptimizer: adam\nuse_ema: true\nmodel:\n  layers: 12\n  width: 768\n# comment\nseed: 42\n",
        );
        assert_eq!(
            params,
            vec![
                ("lr".to_owned(), ParamValue::Number(0.001)),
                ("optimizer".to_owned(), ParamValue::Text("adam".into())),
                ("use_ema".to_owned(), ParamValue::Bool(true)),
                ("model.layers".to_owned(), ParamValue::Number(12.0)),
                ("model.width".to_owned(), ParamValue::Number(768.0)),
                ("seed".to_owned(), ParamValue::Number(42.0)),
            ]
        );
    }
}
