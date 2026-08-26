# Vision

vertov is a terminal viewer for training runs. Point it at the
files a trainer already writes and watch: live charts, a runs
table, comparisons — over ssh, in the terminal you are already
in. No server, no SDK, no browser. malevich renders; vertov
reads, models, and shows.

The goal is the window an on-call researcher actually stares at
during a run, built so that trusting it costs nothing: it asks
for no instrumentation, holds no copy of your data, and never
draws anything the data does not say.

That means three commitments:

- **Reads everything, requires nothing.** tfevents first, then
  the other local formats trainers already produce. The trainer's
  own output is the entire interface. Zero adoption cost is the
  feature every busy run selects for.
- **Honest at every altitude.** A rendered curve is
  pixel-identical to plotting every point; a summary is an exact
  accumulator, never a sample; a torn file, a gap, a stale view
  each look like what they are. A spike is never sampled away.
- **An application built as proof.** Every view is a pure
  function to a `String` — golden-tested, benchmarked against
  recorded budgets. vertov is malevich's flagship application and
  its harshest test; together with kaz it proves that a pure
  string-renderer is enough.

The name keeps the family line: Malevich painted the vocabulary,
kaz speaks it from the shell, Vertov — the kino-eye — points the
camera at life without staging it. A viewer that observes and
never interferes.

## The rules

Six rules, one axis each: what we read, what we may do, where
state lives, what is true, how it is proven, what we own.

1. **Read what training writes.** The standard formats — no
   logger, no SDK, no format of our own. If adopting the viewer
   requires touching training code, the viewer has failed.
2. **Observe; never stage.** vertov never writes into a run
   directory, never signals a training process, never mutates
   what it shows. Everything it persists is disposable cache
   outside the logdir.
3. **The files are the database.** No daemon, no ingest store.
   Exact summaries for every series; full points materialized
   transiently for what is on screen, re-readable at will. Open
   handles and byte offsets are the resume state.
4. **Show the data or say so.** M4 at render, exact accumulators
   in tables, smoothing as a labeled overlay. Truncation is a
   state, corruption is a visible loss, NaN is a gap, staleness
   is displayed. Restarts are segments, drawn as such.
5. **Views are pure strings.** (snapshot, view state, frame) →
   `String`, byte-for-byte testable; IO lives in a thin shell.
   Budgets are recorded and enforced, claims are program output.
6. **The stack is ours.** malevich, kaz, and vertov are one stack
   under one owner. A capability that belongs in the renderer is
   built in the renderer — public, documented, tested — never
   worked around in the app.

## Principles

Constraints the vision names without arguing. One file per
principle; the type names in each "Spelled today" section may
rot, the rest must not.

- [Observe; never stage](principles/observer.md)
- [The files are the database](principles/files-are-the-database.md)
- [Never hide a spike](principles/never-hide-a-spike.md)
- [Honest under failure](principles/honest-under-failure.md)
- [A run is its restarts](principles/restarts.md)
- [Live by default, losing nothing](principles/live-state.md)
- [Pixel-first, honest fallback](principles/pixels.md)
- [One keystroke from data-out](principles/escape-hatches.md)
- [The stack is ours](principles/upstream.md)

The operational elaboration — architecture, formats, roadmap —
is [plan.md](plan.md).
