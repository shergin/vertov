# One keystroke from data-out

Every view exports exactly what it shows. Every TUI view has a
headless twin. The interactive surface is never the only way to
an answer.

## Why

Guild and DVC both shipped CSV and plain-table modes, and users
leaned on them hard — for papers, for scripts, for CI, for the
tool they'd rather finish the analysis in. An interactive view
that traps its data forces re-derivation somewhere else; the
export is what makes the viewer a citizen of a pipeline instead
of a destination.

There is also an interop dividend: the flat runs × (params +
metrics) table is the shape every downstream tool speaks — it was
HiPlot's entire input format.

## The idea

Export is not a feature bolted onto views; it is the same pure
snapshot the view rendered, serialized instead of drawn. One
keystroke in the TUI writes the current view — its filter, its
selection, its columns — as CSV or JSON. One subcommand per view
prints the same thing headlessly, so anything the eye can check,
a script can check.

The bridge goes further in one direction: `--emit-code` prints
the equivalent malevich program with the data inlined, so a chart
that looks right in the viewer becomes source in a project.

## Consequences

- CI can chart a nightly run into a PR comment with the same tool
  a human uses interactively.
- The flat project table is a stable interchange format —
  parallel-coordinates tools and dataframes consume it as-is.
- What you exported is what you saw: filters and selections
  apply, provably, because both paths share the snapshot.
- The TUI can be skipped entirely and the tool remains whole.

## Not this

- Export-as-screenshot.
- Views whose underlying data has no serialized form.
- Interactive-only features with no CLI twin.
- An export that ignores the active filter and dumps everything.

See [Vision](../vision.md) rule 5 — shared pure snapshots are
what make the two paths provably agree — and
[Live by default](live-state.md) for the view state that exports
honor.

## Spelled today

The headless surface is [plan.md](../plan.md) §8 (`ls`, `show`,
`tail`, `export`, `summary`); kaz's conventions (data on stdout,
chart on stderr, `--emit-code`) are the model. This section may
rot; the rest must not.
