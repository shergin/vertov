//! Parsers for the non-tfevents formats vertov reads — the friendliest
//! local files trainers leave behind. Pure functions over lines and small
//! documents: all I/O, offsets, and resume state live in `vertov-model`,
//! which is what makes each parser independently testable.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dvclive;
pub mod mlflow;

/// A typed configuration value, as the simple formats can express one.
#[derive(Clone, PartialEq, Debug)]
pub enum ParamValue {
    /// A number.
    Number(f64),
    /// A boolean.
    Bool(bool),
    /// Anything else, verbatim.
    Text(String),
}

impl ParamValue {
    /// Types a raw scalar the way the source formats mean it: `true`/`false`
    /// as booleans, anything numeric as a number, the rest as text (quotes
    /// stripped).
    pub fn parse(raw: &str) -> ParamValue {
        let raw = raw.trim();
        match raw {
            "true" | "True" => return ParamValue::Bool(true),
            "false" | "False" => return ParamValue::Bool(false),
            _ => {}
        }
        if let Ok(number) = raw.parse::<f64>() {
            return ParamValue::Number(number);
        }
        let unquoted = raw
            .strip_prefix('\'')
            .and_then(|rest| rest.strip_suffix('\''))
            .or_else(|| {
                raw.strip_prefix('"')
                    .and_then(|rest| rest.strip_suffix('"'))
            })
            .unwrap_or(raw);
        ParamValue::Text(unquoted.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_values_are_typed() {
        assert_eq!(ParamValue::parse("0.001"), ParamValue::Number(0.001));
        assert_eq!(ParamValue::parse("1e-3"), ParamValue::Number(0.001));
        assert_eq!(ParamValue::parse("true"), ParamValue::Bool(true));
        assert_eq!(ParamValue::parse("False"), ParamValue::Bool(false));
        assert_eq!(ParamValue::parse("adam"), ParamValue::Text("adam".into()));
        assert_eq!(ParamValue::parse("'adam'"), ParamValue::Text("adam".into()));
        assert_eq!(ParamValue::parse("  8  "), ParamValue::Number(8.0));
    }
}
