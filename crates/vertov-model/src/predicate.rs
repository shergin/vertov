//! The predicate filter language: `lr > 1e-3 and status == active`.
//!
//! Small on purpose (comparisons, `and`/`or`/`not`, parentheses — no
//! functions, no arithmetic): it filters a runs table, it is not a query
//! engine. Fields resolve against a run in order: the built-ins `run`/`name`
//! and `status`, then hyperparameter keys, then scalar tags (their last
//! value) — by exact name first, then by unique `/`-suffix, so `loss < 0.5`
//! finds `train/loss` when nothing else ends in `/loss`.

use crate::project::Run;

/// A parsed predicate, evaluatable against runs.
#[derive(Clone, PartialEq, Debug)]
pub struct Predicate {
    root: Expr,
}

#[derive(Clone, PartialEq, Debug)]
enum Expr {
    Or(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Cmp {
        field: String,
        op: Op,
        value: Literal,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    /// `~`: substring on the field's text form.
    Contains,
}

#[derive(Clone, PartialEq, Debug)]
enum Literal {
    Number(f64),
    Text(String),
    Bool(bool),
}

/// A field's resolved value for one run.
enum FieldValue {
    Number(f64),
    Text(String),
    Bool(bool),
    /// Unknown field or the run has no such value. Every comparison on a
    /// missing value is false — absence never sneaks through a filter.
    Missing,
}

impl Predicate {
    /// Parses `input`. Errors are one human sentence.
    pub fn parse(input: &str) -> Result<Predicate, String> {
        let tokens = tokenize(input)?;
        let mut parser = Parser { tokens, pos: 0 };
        let root = parser.expr()?;
        if parser.pos != parser.tokens.len() {
            return Err(format!("unexpected `{}`", parser.tokens[parser.pos].text()));
        }
        Ok(Predicate { root })
    }

    /// Evaluates against one run. `status` is the caller's status word for
    /// the run (`active`/`idle`/`?` in vertov's UI).
    pub fn matches(&self, name: &str, status: &str, run: &Run) -> bool {
        eval(&self.root, name, status, run)
    }
}

fn eval(expr: &Expr, name: &str, status: &str, run: &Run) -> bool {
    match expr {
        Expr::Or(a, b) => eval(a, name, status, run) || eval(b, name, status, run),
        Expr::And(a, b) => eval(a, name, status, run) && eval(b, name, status, run),
        Expr::Not(inner) => !eval(inner, name, status, run),
        Expr::Cmp { field, op, value } => compare(&resolve(field, name, status, run), *op, value),
    }
}

fn resolve(field: &str, name: &str, status: &str, run: &Run) -> FieldValue {
    match field {
        "run" | "name" => return FieldValue::Text(name.to_owned()),
        "status" => return FieldValue::Text(status.to_owned()),
        _ => {}
    }
    if let Some(value) = run.hparams.get(field) {
        return match value {
            tfevents::HparamValue::F64(v) => FieldValue::Number(*v),
            tfevents::HparamValue::String(v) => FieldValue::Text(v.clone()),
            tfevents::HparamValue::Bool(v) => FieldValue::Bool(*v),
        };
    }
    if let Some(series) = run.series.get(field) {
        return last_value(series);
    }
    // Unique `/`-suffix match over scalar tags.
    let suffix = format!("/{field}");
    let mut matches = run
        .series
        .iter()
        .filter(|(tag, _)| tag.ends_with(&suffix));
    if let (Some((_, series)), None) = (matches.next(), matches.next()) {
        return last_value(series);
    }
    FieldValue::Missing
}

fn last_value(series: &crate::series::Series) -> FieldValue {
    match series.summary.last() {
        Some(point) if point.value.is_finite() => FieldValue::Number(point.value),
        _ => FieldValue::Missing,
    }
}

fn compare(field: &FieldValue, op: Op, literal: &Literal) -> bool {
    match op {
        Op::Contains => field_text(field).is_some_and(|text| match literal {
            Literal::Text(needle) => text.contains(needle),
            Literal::Number(needle) => text.contains(&needle.to_string()),
            Literal::Bool(needle) => text.contains(&needle.to_string()),
        }),
        Op::Eq | Op::Ne => {
            let equal = match (field, literal) {
                (FieldValue::Number(a), Literal::Number(b)) => a == b,
                (FieldValue::Bool(a), Literal::Bool(b)) => a == b,
                (FieldValue::Missing, _) => return false,
                (field, literal) => {
                    field_text(field).as_deref() == Some(literal_text(literal).as_str())
                }
            };
            (op == Op::Eq) == equal
        }
        Op::Gt | Op::Ge | Op::Lt | Op::Le => {
            let (FieldValue::Number(a), Literal::Number(b)) = (field, literal) else {
                return false;
            };
            match op {
                Op::Gt => a > b,
                Op::Ge => a >= b,
                Op::Lt => a < b,
                Op::Le => a <= b,
                _ => unreachable!(),
            }
        }
    }
}

fn field_text(field: &FieldValue) -> Option<String> {
    match field {
        FieldValue::Number(v) => Some(v.to_string()),
        FieldValue::Text(v) => Some(v.clone()),
        FieldValue::Bool(v) => Some(v.to_string()),
        FieldValue::Missing => None,
    }
}

fn literal_text(literal: &Literal) -> String {
    match literal {
        Literal::Number(v) => v.to_string(),
        Literal::Text(v) => v.clone(),
        Literal::Bool(v) => v.to_string(),
    }
}

#[derive(Clone, PartialEq, Debug)]
enum Token {
    Ident(String),
    Number(f64),
    Text(String),
    Op(Op),
    LeftParen,
    RightParen,
    And,
    Or,
    Not,
}

impl Token {
    fn text(&self) -> String {
        match self {
            Token::Ident(s) => s.clone(),
            Token::Number(n) => n.to_string(),
            Token::Text(s) => format!("'{s}'"),
            Token::Op(_) => "comparison".to_owned(),
            Token::LeftParen => "(".to_owned(),
            Token::RightParen => ")".to_owned(),
            Token::And => "and".to_owned(),
            Token::Or => "or".to_owned(),
            Token::Not => "not".to_owned(),
        }
    }
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' => {
                chars.next();
            }
            '(' => {
                chars.next();
                tokens.push(Token::LeftParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RightParen);
            }
            '~' => {
                chars.next();
                tokens.push(Token::Op(Op::Contains));
            }
            '=' => {
                chars.next();
                if chars.next_if_eq(&'=').is_none() {
                    return Err("use `==` for equality".to_owned());
                }
                tokens.push(Token::Op(Op::Eq));
            }
            '!' => {
                chars.next();
                if chars.next_if_eq(&'=').is_none() {
                    return Err("use `!=` for inequality".to_owned());
                }
                tokens.push(Token::Op(Op::Ne));
            }
            '>' => {
                chars.next();
                tokens.push(Token::Op(if chars.next_if_eq(&'=').is_some() {
                    Op::Ge
                } else {
                    Op::Gt
                }));
            }
            '<' => {
                chars.next();
                tokens.push(Token::Op(if chars.next_if_eq(&'=').is_some() {
                    Op::Le
                } else {
                    Op::Lt
                }));
            }
            '\'' | '"' => {
                let quote = ch;
                chars.next();
                let mut text = String::new();
                loop {
                    match chars.next() {
                        Some(ch) if ch == quote => break,
                        Some(ch) => text.push(ch),
                        None => return Err("unterminated string".to_owned()),
                    }
                }
                tokens.push(Token::Text(text));
            }
            _ => {
                let mut word = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || "/_.-+".contains(ch) {
                        word.push(ch);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if word.is_empty() {
                    return Err(format!("unexpected character `{ch}`"));
                }
                tokens.push(match word.as_str() {
                    "and" => Token::And,
                    "or" => Token::Or,
                    "not" => Token::Not,
                    "true" => Token::Ident("true".to_owned()),
                    "false" => Token::Ident("false".to_owned()),
                    _ => match word.parse::<f64>() {
                        Ok(number) => Token::Number(number),
                        Err(_) => Token::Ident(word),
                    },
                });
            }
        }
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn expr(&mut self) -> Result<Expr, String> {
        let mut left = self.and()?;
        while self.peek() == Some(&Token::Or) {
            self.pos += 1;
            left = Expr::Or(Box::new(left), Box::new(self.and()?));
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Expr, String> {
        let mut left = self.unary()?;
        while self.peek() == Some(&Token::And) {
            self.pos += 1;
            left = Expr::And(Box::new(left), Box::new(self.unary()?));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(Token::Not) => {
                self.pos += 1;
                Ok(Expr::Not(Box::new(self.unary()?)))
            }
            Some(Token::LeftParen) => {
                self.pos += 1;
                let inner = self.expr()?;
                if self.peek() != Some(&Token::RightParen) {
                    return Err("missing `)`".to_owned());
                }
                self.pos += 1;
                Ok(inner)
            }
            _ => self.comparison(),
        }
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        let field = match self.peek() {
            Some(Token::Ident(name)) => name.clone(),
            Some(other) => return Err(format!("expected a field name, got `{}`", other.text())),
            None => return Err("expected a field name".to_owned()),
        };
        self.pos += 1;
        let op = match self.peek() {
            Some(Token::Op(op)) => *op,
            _ => return Err(format!("expected a comparison after `{field}`")),
        };
        self.pos += 1;
        let value = match self.peek() {
            Some(Token::Number(number)) => Literal::Number(*number),
            Some(Token::Text(text)) => Literal::Text(text.clone()),
            Some(Token::Ident(word)) if word == "true" => Literal::Bool(true),
            Some(Token::Ident(word)) if word == "false" => Literal::Bool(false),
            Some(Token::Ident(word)) => Literal::Text(word.clone()),
            _ => return Err("expected a value to compare against".to_owned()),
        };
        self.pos += 1;
        Ok(Expr::Cmp { field, op, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::{PointStamp, SeriesClass, SeriesSummary};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn test_run() -> Run {
        let mut hparams = BTreeMap::new();
        hparams.insert("lr".to_owned(), tfevents::HparamValue::F64(0.003));
        hparams.insert(
            "optimizer".to_owned(),
            tfevents::HparamValue::String("adam".to_owned()),
        );
        hparams.insert("amsgrad".to_owned(), tfevents::HparamValue::Bool(true));
        let mut series = BTreeMap::new();
        let mut summary = SeriesSummary::default();
        summary.observe(PointStamp {
            step: 10,
            wall: 0.0,
            value: 0.25,
        });
        series.insert(
            "train/loss".to_owned(),
            crate::series::Series {
                class: SeriesClass::Scalar,
                plugin: None,
                summary,
            },
        );
        Run {
            dir: PathBuf::new(),
            hparams,
            series,
            first_wall: None,
            last_wall: None,
            last_write: None,
            preemptions: 0,
        }
    }

    fn check(input: &str) -> bool {
        Predicate::parse(input)
            .unwrap()
            .matches("exp/adam-1", "active", &test_run())
    }

    #[test]
    fn comparisons() {
        assert!(check("lr > 1e-3"));
        assert!(!check("lr > 0.01"));
        assert!(check("lr <= 0.003"));
        assert!(check("optimizer == adam"));
        assert!(check("optimizer != sgd"));
        assert!(check("amsgrad == true"));
        assert!(check("status == active"));
        assert!(check("run ~ adam"));
        assert!(check("name ~ 'exp/'"));
    }

    #[test]
    fn metric_fields_by_tag_and_suffix() {
        assert!(check("train/loss < 0.5"));
        assert!(check("loss < 0.5"));
        assert!(!check("loss > 0.5"));
    }

    #[test]
    fn boolean_combinators_and_precedence() {
        assert!(check("lr > 1e-3 and optimizer == adam"));
        assert!(check("lr > 1 or optimizer == adam"));
        assert!(!check("lr > 1 and optimizer == adam or optimizer == sgd"));
        assert!(check("not lr > 1"));
        assert!(check("(lr > 1 or lr < 0.01) and status == active"));
    }

    #[test]
    fn missing_fields_never_match() {
        assert!(!check("nonexistent > 1"));
        assert!(!check("nonexistent == 1"));
        assert!(!check("nonexistent != 1"));
        // ...but a `not` around them does, explicitly.
        assert!(check("not nonexistent == 1"));
    }

    #[test]
    fn parse_errors_are_sentences() {
        assert!(Predicate::parse("lr >").is_err());
        assert!(Predicate::parse("lr = 3").is_err());
        assert!(Predicate::parse("(lr > 1").is_err());
        assert!(Predicate::parse("lr > 1 banana").is_err());
        assert!(Predicate::parse("'unterminated").is_err());
    }
}
