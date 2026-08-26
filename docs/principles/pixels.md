# Pixel-first, honest fallback

Where the terminal speaks a pixel protocol, chart panels are real
images inside the TUI. Layouts are designed assuming pixels; cell
glyphs are the honest fallback rung, never the target.

## Why

A terminal chart made of quadrant glyphs caps what a chart can
say: no antialiasing, coarse heatmaps, thick lines. Every
terminal competitor lives under that cap. Sixel, kitty graphics,
and iTerm2 images remove it — and they work over ssh precisely
because the escape sequences are decoded by the local emulator;
the bytes just ride the connection. Crisp device-pixel curves in
a TUI over ssh is the single most visible thing no other
experiment viewer has.

Designing pixel-*second* — bolting images onto a cell layout —
produces layouts that never quite use them. The commitment has to
come first.

## The idea

The integration is one trick, well-precedented: the chart widget
reserves its rectangle with skip-flagged cells so the cell diff
never paints there; after the frame flushes, the cursor jumps to
the panel and a single deterministic string — chrome as text,
plot rectangle as image, cursor walks column-absolute and
row-relative — lands the graphics.

Repaint on data change, not on a timer; a dashboard's few frames
a second is well within sixel-encoding budget, and kitty's
transmit-once placements make it near-free. Capability detection
probes once, before raw mode, and is cached. The fallback is a
runtime rung, never a compile-time fork: the same layout renders
through the cell widget, and that is also what lands in
scrollback on exit.

Terrain is engineered around, not against: tmux passthrough is
detected and degraded loudly; images ignore clipping, so pixel
panels live only in fixed, non-overlapping layout slots.

## Consequences

- The showcase is a gif no competitor can record.
- Piped, plain, and dumb-terminal output stay first-class — the
  cell ladder is malevich's home ground.
- Layout design carries a constraint (fixed slots) accepted
  knowingly.
- Emulator coverage (kitty, wezterm, iTerm2, foot, tmux) is a
  tested matrix, not a hope.

## Not this

- Requiring pixel support to use the TUI.
- X11 overlay hacks (ueberzug-style) — they die over ssh.
- Re-transmitting full sixel panels on every tick.
- Artifacts on unsupported terminals instead of loud fallback.

See [Vision](../vision.md) rules 4 and 6 — the fallback ladder is
an honesty property, and the integration seams belong upstream —
and [The stack is ours](upstream.md).

## Spelled today

malevich's `pixel` feature carries the ladder:
`render_pixels_at`, `Capabilities::detect_for`, the
sixel/kitty/iTerm2 encoders, the pixel benches. The skip-cell
mechanism and repaint policy are [plan.md](../plan.md) §5.6;
`ratatui-image` and yazi are the precedents. This section may
rot; the rest must not.
