# A run is its restarts

A modern run is a sequence of restart segments. Segments are
first-class: preemption is per-series, boundaries are drawn, and
the x-axis speaks tokens as well as steps.

## Why

Llama 3's 54-day pretraining logged 466 interruptions — roughly
one every three hours. Resuming from a checkpoint older than the
crash rewrites steps, so the same step number can carry two
values, one of them from an abandoned timeline. Every tracker
eventually grows a feature for this (rewind, fork, re-attach)
because users demand it after the fact; a viewer born in the LLM
era should treat it as the normal shape of a run.

And the axis matters: learning-rate schedules and anneals are
defined in tokens, so the plot a decision is made on must be
plottable in tokens.

## The idea

Within one series, a new point at a step at or below the tail
preempts: the overlapped tail is truncated and a segment boundary
is recorded. Preemption is strictly per-series — a step reset in
one tag never disturbs another — which is the semantics that
falls out of reading a run's files in order through per-tag
series, and the one that keeps unrelated metrics intact.

Truncated is not destroyed. The pre-restart tail is kept as a
ghost segment, renderable on demand, because the abandoned
timeline is evidence too — it is often *why* the run was
restarted.

Boundaries render as explicit marks, so a cliff in the loss curve
next to a restart marker explains itself. The x-axis cycles step,
wall time, relative time, and tokens — tokens via a designated
counter series, joined by step.

## Consequences

- Overlapping steps never double-plot; every rendered step has
  one value and a clear lineage.
- Comparisons can align on tokens when two runs' step counts
  diverge.
- "How many times did this run restart" is a first-class,
  visible fact.
- The ghost toggle turns a post-mortem from archaeology into a
  view.

## Not this

- Global purges across all tags on a step reset (old
  TensorBoard's behavior).
- Silently interleaving both timelines into one sawtooth curve.
- Discarding pre-restart data irrecoverably.
- Pretending step is the only x-axis.

See [Vision](../vision.md) rule 4, and
[Honest under failure](honest-under-failure.md) — a restart is a
lifecycle event, not a failure to hide.

## Spelled today

Preemption semantics follow RustBoard's `StageReservoir::preempt`
(per-tag tail truncation), extended with recorded boundaries and
ghost segments in [plan.md](../plan.md) §5.2; the tokens axis is
§3.6 and a Phase 5 deliverable. Segment marks render as malevich
`Rule` layers. This section may rot; the rest must not.
