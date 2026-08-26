# Live by default, losing nothing

Watching a running job is the headline use. Data refreshes
continuously; a refresh never costs the user their sort, cursor,
zoom, filter, or selection.

## Why

The tool exists for the on-call window: a training job on a
cluster, a researcher over ssh, a loss curve that needs watching.
Guild's TUI made refresh a keystroke that reset the sort order —
a paper cut so canonical it defines the anti-goal. A viewer that
punishes refreshing teaches users to stop refreshing, which
defeats a live viewer entirely.

The infrastructure reality shapes the mechanism: training logs
live on NFS and Lustre, where inotify is silent. Polling is not
the fallback; it is the design center.

## The idea

View state and data are separate values. The reload loop advances
a data snapshot; the view — sort, cursor, zoom, filters, the
keep/exclude working set — is state the snapshot never touches.
A tick re-renders the same view over newer data, and nothing else
moves. New runs append under the current sort; a sorted column
re-sorts stably.

Polling is primary (a few seconds, tunable); filesystem
notification only shortens the next tick when it happens to work.
Liveness is visible: a paused view says paused, a stale view says
stale, and the final frame survives in scrollback on exit rather
than vanishing with an alternate screen.

## Consequences

- Refresh invariance is testable: render(view, data₁) and
  render(view, data₂) differ only where the data does.
- Watching and interacting never conflict — no modal "reloading"
  moments.
- NFS-mounted logdirs are the well-supported case, not the
  degraded one.
- Pause is a state, not a stop: resume catches up from held
  offsets.

## Not this

- Refresh that rebuilds the table and drops sort, cursor, or
  selection.
- Repainting on a timer when no data changed.
- Requiring inotify to be live.
- Alt-screen exits that erase what the user was looking at.

See [Vision](../vision.md) rule 5 — view state separable from
data is a corollary of views being pure functions — and
[Honest under failure](honest-under-failure.md) for the staleness
contract.

## Spelled today

The reload loop is [plan.md](../plan.md) §5.3 (poll primary,
`notify` opportunistic, `PollWatcher` fallback); state
preservation is §3.8 and a Phase 4 exit criterion. malevich's
live-repaint discipline (in-place repaint, scrollback survives)
is the model for headless `tail`. This section may rot; the rest
must not.
