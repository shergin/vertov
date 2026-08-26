# The files are the database

No daemon, no ingest, no private store. Summaries for every
series; points only for what is being looked at, re-readable at
will.

## Why

TensorBoard's data server holds every run in memory, so it must
sample: 1000 reservoir points per scalar series, chosen at
ingest, before anyone asked a question. That is the correct
design for a server that answers arbitrary queries — and the
wrong one for a viewer, which knows exactly what is on screen.

A private store also creates the problems trackers are disliked
for: import time before first chart, migrations, a second copy of
the data that can drift from the first, and a daemon to babysit.
The event files are already durable, already append-only, already
the source of truth. Duplicating them buys nothing a viewer
needs.

## The idea

Two tiers, split by question. Every series, always: an exact
summary — count, first/last, min/max, mergeable moments — cheap
enough to keep for fifty thousand tags and sufficient for every
table. The series on screen: full points, materialized transiently
by reading the file, bounded by an LRU, droppable and re-readable
at any time.

Resume state is not bookkeeping but the reader itself: an open
handle whose offset, plus a partial-record buffer, marks exactly
where the file's valid prefix ends. Growing files continue from
there; nothing is ever re-parsed.

The only thing persisted is a summary cache — `(path, size,
mtime)` → summaries and final offset — so a cold start on an
unchanged logdir is a metadata walk. The cache holds conclusions,
never a second copy of the data, and deleting it is always safe.

## Consequences

- First chart appears while a large logdir is still streaming in;
  there is no import step to wait out.
- Memory scales with what is on screen, not with the logdir.
- `kill -9` loses nothing: all state reconstructs from the files.
- There is no migration problem, ever — the source of truth never
  moved.
- Zoom and overlay can always go back to the file for full
  fidelity.

## Not this

- Importing runs into a viewer-owned database.
- A background indexing daemon.
- Sampling at ingest to fit a memory budget.
- Caching decoded points on disk "to be fast" — the cache stores
  summaries and offsets only.

See [Vision](../vision.md) rule 3,
[Never hide a spike](never-hide-a-spike.md) for why ingest-time
sampling is also a fidelity failure, and
[Observe; never stage](observer.md) for where the cache may live.

## Spelled today

The model and tiers are [plan.md](../plan.md) §5.2, the loop
§5.3, the cache §5.4. Summaries use malevich's mergeable
accumulator law (`Moments`). RustBoard's open-handle resume and
truncation semantics are the adopted recipe. This section may
rot; the rest must not.
