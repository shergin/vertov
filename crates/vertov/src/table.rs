//! One row model, three encodings: aligned text for eyes, CSV and JSON for
//! pipes. Every view exports — escape hatches are load-bearing.
//!
//! Floats render via Rust's shortest-roundtrip `Display`: every digit shown
//! is real, nothing is rounded into a lie.

/// Output encoding, chosen by `--csv` / `--json`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Text,
    Csv,
    Json,
}

/// One cell of a table.
#[derive(Clone, Debug)]
pub enum Cell {
    Text(String),
    Int(i64),
    Float(f64),
    Empty,
}

impl Cell {
    fn text(&self) -> String {
        match self {
            Cell::Text(text) => text.clone(),
            Cell::Int(value) => value.to_string(),
            Cell::Float(value) => value.to_string(),
            Cell::Empty => String::new(),
        }
    }

    fn is_numeric(&self) -> bool {
        matches!(self, Cell::Int(_) | Cell::Float(_))
    }
}

/// A table: named columns, rows of cells.
pub struct Table {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Cell>>,
}

impl Table {
    pub fn render(&self, format: Format) -> String {
        match format {
            Format::Text => self.render_text(),
            Format::Csv => self.render_csv(),
            Format::Json => self.render_json(),
        }
    }

    fn render_text(&self) -> String {
        // Widths in characters, not bytes: non-ASCII run names would
        // otherwise blow the column out.
        let mut widths: Vec<usize> = self.columns.iter().map(|c| c.chars().count()).collect();
        let rendered: Vec<Vec<String>> = self
            .rows
            .iter()
            .map(|row| row.iter().map(Cell::text).collect())
            .collect();
        for row in &rendered {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(cell.chars().count());
            }
        }
        // Numeric columns right-align so magnitudes line up.
        let numeric: Vec<bool> = (0..self.columns.len())
            .map(|index| {
                self.rows
                    .iter()
                    .any(|row| row[index].is_numeric())
                    && self
                        .rows
                        .iter()
                        .all(|row| row[index].is_numeric() || matches!(row[index], Cell::Empty))
            })
            .collect();

        let mut out = String::new();
        for (index, column) in self.columns.iter().enumerate() {
            if index > 0 {
                out.push_str("  ");
            }
            pad(&mut out, column, widths[index], numeric[index]);
        }
        out.push('\n');
        for row in &rendered {
            for (index, cell) in row.iter().enumerate() {
                if index > 0 {
                    out.push_str("  ");
                }
                pad(&mut out, cell, widths[index], numeric[index]);
            }
            while out.ends_with(' ') {
                out.pop();
            }
            out.push('\n');
        }
        out
    }

    fn render_csv(&self) -> String {
        let mut out = String::new();
        for (index, column) in self.columns.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push_str(&csv_escape(column));
        }
        out.push('\n');
        for row in &self.rows {
            for (index, cell) in row.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&csv_escape(&cell.text()));
            }
            out.push('\n');
        }
        out
    }

    fn render_json(&self) -> String {
        let mut out = String::from("[\n");
        for (row_index, row) in self.rows.iter().enumerate() {
            out.push_str("  {");
            for (index, cell) in row.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&json_string(&self.columns[index]));
                out.push_str(": ");
                out.push_str(&match cell {
                    Cell::Text(text) => json_string(text),
                    Cell::Int(value) => value.to_string(),
                    // JSON has no NaN/Infinity; a non-finite value is a gap.
                    Cell::Float(value) if value.is_finite() => value.to_string(),
                    Cell::Float(_) | Cell::Empty => "null".to_owned(),
                });
            }
            out.push('}');
            if row_index + 1 < self.rows.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("]\n");
        out
    }
}

fn pad(out: &mut String, text: &str, width: usize, right: bool) {
    let fill = width.saturating_sub(text.chars().count());
    if right {
        out.extend(std::iter::repeat_n(' ', fill));
        out.push_str(text);
    } else {
        out.push_str(text);
        out.extend(std::iter::repeat_n(' ', fill));
    }
}

fn csv_escape(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_owned()
    }
}

fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// `92s` / `4m05s` / `2h11m` / `54d13h` — wall-clock spans, timezone-free.
pub fn fmt_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return String::new();
    }
    let total = seconds.round() as u64;
    if total < 120 {
        format!("{total}s")
    } else if total < 3600 {
        format!("{}m{:02}s", total / 60, total % 60)
    } else if total < 48 * 3600 {
        format!("{}h{:02}m", total / 3600, (total % 3600) / 60)
    } else {
        format!("{}d{:02}h", total / 86_400, (total % 86_400) / 3600)
    }
}

/// A float truncated to about `digits` significant digits — truncated,
/// never rounded: `0.9999999` must not become `1.000000`. Integer digits
/// are never cut (that would change the magnitude); only the fraction
/// shortens. For screen cells; data exports keep full precision.
pub fn fmt_sig(value: f64, digits: usize) -> String {
    let full = value.to_string();
    if !value.is_finite() {
        return full;
    }
    // Rust's f64 Display never emits scientific notation, so the string is
    // always sign + integer digits + optional fraction.
    truncate_fraction(&full, digits)
}

fn truncate_fraction(text: &str, digits: usize) -> String {
    let Some(dot) = text.find('.') else {
        return text.to_owned();
    };
    let integer = &text[..dot];
    let integer_significant = integer
        .chars()
        .filter(char::is_ascii_digit)
        .skip_while(|&ch| ch == '0')
        .count();
    let mut budget = digits.saturating_sub(integer_significant);
    if budget == 0 {
        return integer.to_owned();
    }
    let mut kept = String::from(integer);
    kept.push('.');
    let mut significant_started = integer_significant > 0;
    for ch in text[dot + 1..].chars() {
        kept.push(ch);
        if ch != '0' {
            significant_started = true;
        }
        if significant_started {
            budget -= 1;
            if budget == 0 {
                break;
            }
        }
    }
    if kept.ends_with('.') {
        kept.pop();
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Table {
        Table {
            columns: vec!["run".into(), "points".into(), "best".into()],
            rows: vec![
                vec![Cell::Text("adam".into()), Cell::Int(100), Cell::Float(0.25)],
                vec![Cell::Text("sgd, v2".into()), Cell::Int(90), Cell::Empty],
            ],
        }
    }

    #[test]
    fn text_aligns_numbers_right() {
        assert_eq!(
            sample().render(Format::Text),
            "run      points  best\n\
             adam        100  0.25\n\
             sgd, v2      90\n"
        );
    }

    #[test]
    fn csv_escapes_commas() {
        assert_eq!(
            sample().render(Format::Csv),
            "run,points,best\nadam,100,0.25\n\"sgd, v2\",90,\n"
        );
    }

    #[test]
    fn json_uses_null_for_gaps() {
        assert_eq!(
            sample().render(Format::Json),
            "[\n  {\"run\": \"adam\", \"points\": 100, \"best\": 0.25},\n  {\"run\": \"sgd, v2\", \"points\": 90, \"best\": null}\n]\n"
        );
    }

    #[test]
    fn durations() {
        assert_eq!(fmt_duration(45.0), "45s");
        assert_eq!(fmt_duration(245.0), "4m05s");
        assert_eq!(fmt_duration(7890.0), "2h11m");
        assert_eq!(fmt_duration(4_712_400.0), "54d13h");
    }

    #[test]
    fn text_widths_count_chars_not_bytes() {
        let table = Table {
            columns: vec!["run".into(), "n".into()],
            rows: vec![
                vec![Cell::Text("метрики".into()), Cell::Int(1)],
                vec![Cell::Text("ascii".into()), Cell::Int(2)],
            ],
        };
        assert_eq!(
            table.render(Format::Text),
            "run      n\nметрики  1\nascii    2\n"
        );
    }

    #[test]
    fn significant_truncation_never_rounds() {
        assert_eq!(fmt_sig(0.9999999, 5), "0.99999");
        assert_eq!(fmt_sig(0.916704475879, 5), "0.91670");
        assert_eq!(fmt_sig(487424.0, 5), "487424");
        // Integer digits are never cut — that would change the magnitude.
        assert_eq!(fmt_sig(487424.75, 5), "487424");
        assert_eq!(fmt_sig(-0.0001234567, 4), "-0.0001234");
        assert_eq!(fmt_sig(1.5, 5), "1.5");
        assert_eq!(fmt_sig(f64::NAN, 5), "NaN");
        assert_eq!(fmt_sig(1.23456789e-7, 4), "0.0000001234");
    }
}
