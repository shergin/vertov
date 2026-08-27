//! The TUI's visual voice: constructivist restraint. One accent — the
//! kino-eye vermilion — spent only on orientation (active tab, focused
//! frame, cursor, sort mark); quiet graphite chrome around it; muted
//! semantic colors for run state. The terminal's own background is always
//! respected: nothing paints large fields.

use ratatui::style::{Color, Modifier, Style};

/// El Lissitzky's vermilion — the one loud color.
pub const ACCENT: Color = Color::Rgb(227, 66, 52);
/// The accent dimmed to a tint fit for a cursor-row background.
pub const ACCENT_DIM: Color = Color::Rgb(84, 32, 26);
/// Quiet chrome: unfocused borders, separators.
pub const BORDER: Color = Color::Rgb(88, 86, 82);
/// Secondary text: hints, counts, durations.
pub const DIM: Color = Color::Rgb(148, 145, 138);
/// Run states.
pub const LIVE: Color = Color::Rgb(120, 190, 100);
pub const STALE: Color = Color::Rgb(224, 164, 66);
pub const DONE: Color = Color::Rgb(122, 152, 189);

pub fn border() -> Style {
    Style::default().fg(BORDER)
}

pub fn border_focus() -> Style {
    Style::default().fg(ACCENT)
}

pub fn title() -> Style {
    Style::default().fg(DIM).add_modifier(Modifier::BOLD)
}

pub fn title_focus() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

pub fn dim() -> Style {
    Style::default().fg(DIM)
}

pub fn accent() -> Style {
    Style::default().fg(ACCENT)
}

pub fn cursor_row() -> Style {
    Style::default().bg(ACCENT_DIM).add_modifier(Modifier::BOLD)
}

pub fn header() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
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
