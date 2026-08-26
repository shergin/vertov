# Honest under failure

Torn files, corrupt records, gaps, and stale views are shown as
what they are. The viewer never guesses.

## Why

Training logs are written by processes that get killed. A crashed
writer leaves a truncated record at the tail of the file as a
matter of routine, not exception; a network filesystem serves the
viewer an old view of a file that has already grown. A viewer
that treats these as errors refuses to show the 99% of the run
that is intact; a viewer that papers over them shows the on-call
researcher a fiction at exactly the moment fidelity matters.

The checksum lesson is also economic: TensorBoard's reloads were
famously CPU-bound on CRC verification of data that parses fine
without it.

## The idea

Truncation is a state, not an error. A short read at the tail
parks the reader — partial buffer kept, offset held — and the
valid prefix is served; if the file grows, the record completes
seamlessly on a later tick. A crashed run's torn tail is simply a
file that never grows again.

Corruption is a visible loss. The framing checksum (which makes
the rest of the file readable) is always verified; the payload
checksum is computed only when a record fails to parse, to report
corruption instead of a confusing decode error. A bad record
becomes a data-loss tombstone in the series, not a silent skip.

Absence and lag are rendered. NaN is a gap, never interpolated.
A series that stopped advancing while others continue looks
stopped. When the poll loop is behind or a file is dead, the view
says stale and says why. Run status ("running", "crashed") is
displayed with its provenance, because it is inferred from
different signals in different formats.

## Consequences

- The valid prefix of any file is always available, whatever
  happened to the tail.
- Reloads are parse-bound, not checksum-bound.
- Error states are ordinary renderable values — snapshot-tested
  like any view.
- The viewer never draws a value it did not read.

## Not this

- Declaring a file dead on a torn tail.
- Verifying every payload checksum on the hot path.
- Interpolating across gaps, or connecting a line through NaN.
- A spinner where a staleness label belongs.
- Treating EOF as end-of-run.

See [Vision](../vision.md) rule 4,
[A run is its restarts](restarts.md) for the failure mode that is
actually a lifecycle, and
[Never hide a spike](never-hide-a-spike.md) for honesty about
intact data.

## Spelled today

The semantics are RustBoard's, adopted in [plan.md](../plan.md)
§5.3: length-CRC always, data-CRC on parse failure, `Truncated`
as a wait-state, dead files keep their prefix. malevich already
renders NaN as an honest gap. This section may rot; the rest must
not.
