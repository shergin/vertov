//! The TUI's visual voice: constructivist restraint. A grayscale ramp
//! carries the whole hierarchy — bold terminal-default for what matters
//! now, gray for what supports it, dark gray for chrome — and the one
//! accent, the kino-eye vermilion, is spent only on orientation: the
//! brand mark, the active tab's key, the focused frame, the selection
//! marker, a transient message. Keys in hint bars are bold, not colored;
//! the cursor row is a neutral band, not a tinted one. The terminal's
//! own background is always respected: nothing paints large fields.

use ratatui::style::{Color, Modifier, Style};

/// El Lissitzky's vermilion — the one loud color.
pub const ACCENT: Color = Color::Rgb(227, 66, 52);
/// A neutral graphite band for the cursor row.
pub const BAND: Color = Color::Rgb(52, 51, 49);
/// Quiet chrome: unfocused borders, separators.
pub const BORDER: Color = Color::Rgb(88, 86, 82);
/// Secondary text: hints, counts, durations.
pub const DIM: Color = Color::Rgb(148, 145, 138);
/// Run states — members of the chart palette (`crate::chart::SERIES`):
/// live wears the sage, stale the amber, done the sky, so a run's status
/// dot and its chart lines come from one set.
pub const LIVE: Color = Color::Rgb(139, 178, 91);
pub const STALE: Color = Color::Rgb(222, 168, 62);
pub const DONE: Color = Color::Rgb(108, 153, 212);

pub fn border() -> Style {
    Style::default().fg(BORDER)
}

pub fn border_focus() -> Style {
    Style::default().fg(ACCENT)
}

pub fn title() -> Style {
    Style::default().fg(DIM).add_modifier(Modifier::BOLD)
}

/// A focused panel's title: promoted to the primary tier, not colored —
/// the accent border already says where the keyboard is. The explicit
/// `Reset` matters: block titles inherit the border's color unless the
/// title states its own.
pub fn title_focus() -> Style {
    Style::default().fg(Color::Reset).add_modifier(Modifier::BOLD)
}

pub fn dim() -> Style {
    Style::default().fg(DIM)
}

pub fn accent() -> Style {
    Style::default().fg(ACCENT)
}

pub fn cursor_row() -> Style {
    Style::default().bg(BAND).add_modifier(Modifier::BOLD)
}

pub fn header() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

/// A key in a hint bar: bold, never colored — Grok's grammar, where the
/// verb is dim and the key carries the weight.
pub fn key() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

/// The `|` between hints: chrome-dark, quieter than the hints themselves.
pub fn separator() -> Style {
    Style::default().fg(BORDER)
}

/// The status word and its dot, by state.
pub fn status(word: &str) -> (&'static str, Style) {
    match word {
        "active" => ("●", Style::default().fg(LIVE)),
        "done" => ("✓", Style::default().fg(DONE)),
        "idle" => ("◌", Style::default().fg(DIM)),
        _ => ("?", Style::default().fg(DIM)),
    }
}
