# The stack is ours

malevich, kaz, and vertov are one stack under one owner. A
capability that belongs in the renderer is built in the renderer
— public, documented, tested — never worked around in the app.

## Why

kaz's design section makes a claim: the CLI contains zero
rendering logic, and that is the proof that a pure
string-renderer is enough. An app-local workaround — a hand-drawn
overlay here, a patched escape sequence there — quietly retracts
that claim. Worse, it forks the vocabulary: the workaround ships
once, in one app, while a library feature compounds across kaz,
the demos, vertov, and every other malevich user at once.

The direction of pressure is also the point. A library grows best
under a demanding real application; fred, sysmon, and learn each
pulled features into malevich that the library was better for.
vertov — the flagship application and the harshest test — is that
pressure at full strength. Treating library gaps as someone
else's roadmap would waste exactly the advantage of owning both
sides.

## The idea

When vertov needs something the renderer cannot say, the gap
flows upstream and is built there, under malevich's own bar: a
mark channel, a stat parameter, a scale option, or a theme entry
— or it does not ship. Ownership is not an exemption from the
budget discipline; it is being subject to it first. If a need
cannot be expressed at that altitude, the vertov feature gets
redesigned, not smuggled in as a special case.

Upstream features land whole — public API, docs, tests, a
changelog entry, semver honored — because they are malevich
features that vertov happens to have motivated, not vertov
internals hosted in the library. First candidates are already
visible: a pixel-aware ratatui integration, a row-anchored pixel
render.

The same ownership runs down-stack: a bug found through vertov is
fixed in malevich, with a regression test, even when an app-side
dodge would be quicker.

## Consequences

- vertov contains zero rendering logic — kaz's claim, extended to
  a full TUI.
- malevich grows only through proven need, its budget discipline
  intact.
- No vendored or forked renderer code, ever.
- Renderer gaps become issues, then features, then releases —
  never local patches.

## Not this

- An app-local chart type or drawing routine "for now".
- Copy-pasting renderer internals into vertov to move faster.
- Pushing a vertov-shaped special case into malevich that its
  budget would refuse from a stranger.
- Letting a vertov feature block on upstream forever — if it
  fails the library's bar, redesign the feature.

See [Vision](../vision.md) rule 6, and
[Pixel-first](pixels.md) for the first features this principle
will route upstream.

## Spelled today

malevich 1.17 already carries the vertov-shaped seams: M4,
`stat::ewma`, band axes, `stream::Ring`, `plot.widget()`,
`render_pixels_at`, `Capabilities::detect_for`. The upstream
candidates are named in [plan.md](../plan.md) §5.6. This section
may rot; the rest must not.
