# Observe; never stage

vertov reads run directories. It never writes into them, never
signals the training process, never mutates what it shows.

## Why

The viewer's one non-negotiable asset is that pointing it at the
only copy of a two-month run costs nothing. A viewer that writes
is a viewer you must audit first. Guild's TUI wrote `marked`
attributes back into run directories — a small convenience that
made every run dir partly the viewer's property. On shared
filesystems the stakes are higher: a stray lock file or an
opportunistic write on NFS can stall the trainer it was watching.

A tool with no side effects needs no permission, no rollout, no
trust negotiation. Anyone with read access can run it against
production logs today.

## The idea

Observation with zero back-pressure. Files open read-only;
SQLite backends open with read-only connections against WAL
databases built for concurrent readers; nothing in a logdir is
ever locked, touched, or reordered.

What vertov must persist — the summary cache, configuration —
lives outside the logdir, in the viewer's own space, and is
disposable by construction: deleting it is always safe, and the
next start rebuilds it from the files. If an annotation feature
ever exists (pin a run, note a spike), the annotation is vertov's
data in vertov's space, keyed to the run — never a write into it.

## Consequences

- Two viewers on one logdir cannot conflict, with each other or
  with the trainer.
- A viewer crash cannot corrupt a run. There is nothing to fsck.
- `rm -rf ~/.cache/vertov` is always a no-op semantically.
- Runs on read-only mounts and archived runs work identically to
  live ones.

## Not this

- Guild-style writeback of marks or attributes into run dirs.
- Lock files, temp files, or indexes placed inside the logdir.
- "Cleanup" or "delete run" features. vertov shows; the shell
  deletes.
- Opening a tracker's database read-write because it was
  convenient.

See [Vision](../vision.md) rule 2, and
[The files are the database](files-are-the-database.md) for where
vertov's own state lives instead.

## Spelled today

The cache location and disposability contract are
[plan.md](../plan.md) §5.4; the read path design is §5.3. Trackio
reads ride WAL read-only connections polled on
`max(mtime(db), mtime(db-wal))`. This section may rot; the rest
must not.
