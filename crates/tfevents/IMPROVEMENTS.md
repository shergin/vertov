# tfevents: improvement audit

_Research and API review, 2026-08-27_

## Bottom line

tfevents already has an excellent core idea: a tiny, zero-dependency, safe
Rust reader designed specifically for live, append-only TensorBoard logs. Its
resumable framing and committed offsets are genuinely differentiated.

It is not yet “great” as a public crate. The main problems are not missing
features:

1. The advertised integrity guarantee is currently false.
2. The public API exposes TF1/TF2 protobuf representation details instead of
   TensorBoard concepts.
3. “No complete event available” is modeled as an error.
4. Hostile-input and compatibility guarantees exceed the current evidence.
5. The API surface is broader than the crate's intended product boundary.

The north-star contract should be:

> Yield every complete, valid, supported summary value exactly once; report
> every loss explicitly; return “not available yet” at the live tail; and
> provide an exact restart checkpoint.

## What the crate does today

It provides:

- TFRecord framing with hand-written masked CRC-32C.
- A resumable state machine that retains partial headers and payloads.
- Byte checkpoints via committed_offset.
- A minimal protobuf decoder for TensorBoard Event records.
- TF1 and TF2 scalar and histogram extraction.
- Raw-ish image, string tensor, metadata, and hparams decoding.
- A benchmark, three fuzz targets, synthetic tests, and tensorboardX fixtures.
- Zero normal dependencies and unsafe code forbidden.

The core is in [record.rs](src/record.rs), [reader.rs](src/reader.rs), and
[event.rs](src/event.rs).

The niche is good. The broader
[tfrecord crate](https://docs.rs/tfrecord/latest/tfrecord/) already covers
generated TensorFlow protos, writing, async, serde, ndarray, and similar
general-purpose needs. tfevents should own the narrower space: small, live,
semantic event reading.

## The critical corrections

### 1. Make integrity truthful

[EventFileReader::next_event](src/reader.rs) returns a successfully decoded
protobuf without checking its payload CRC. A changed float bit, tag byte, step,
or timestamp can remain valid protobuf and be silently accepted.

That directly contradicts “corruption is visible, never silent” in the
[README](README.md) and crate docs.

TensorBoard's own Rust reader verifies payload checksums by default and makes
unchecked operation an explicit opt-out:
[TensorBoard event_file.rs](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/data/server/event_file.rs).

Recommended API:

~~~rust
pub enum ChecksumPolicy {
    Always,
    OnDecodeFailure,
}

Reader::new(stream) // Always
Reader::with_checksum_policy(stream, ChecksumPolicy::OnDecodeFailure)
~~~

Use an enum, not a boolean. Vertov can explicitly select the faster policy if
benchmarks justify it, but the safe API must be the obvious API.

### 2. A live tail must not be an error

A completely clean EOF and a partial record both currently become
ReadEventError::Truncated. Even the zero-byte EOF case is called “truncated,”
despite the crate's central claim that this is normal state.

The primitive high-level operation should be:

~~~rust
pub fn read_event(&mut self) -> Result<Option<Event>, Error>;
~~~

- Some(event): one complete event.
- None: no complete event is available now; retry after the source grows.
- Err: actual corruption, unsupported data, or I/O failure.

Also provide:

~~~rust
pub fn available(&mut self) -> Available<'_, R>;
~~~

Each call drains everything currently available. This matches TensorBoard's
repeated Load() model:
[official event loader](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_file_loader.py).

If finished-file validation needs to distinguish a clean boundary from a torn
tail, expose a secondary tail_state() or finish() API. TensorFlow's lower-level
reader distinguishes clean EOF from truncated data:
[official RecordReader](https://github.com/tensorflow/tensorflow/blob/master/third_party/xla/xla/tsl/lib/io/record_reader.cc).

### 3. Harden the parser contract

These should be fixed before publication:

- [Tensor::scalar](src/event.rs) multiplies untrusted signed dimensions with
  product::<i64>(). This can overflow and panic in debug builds; negative
  shapes may also accidentally multiply to one. For the supported scalar
  convention, simply accept [] and deliberately supported [1].
- Record reservation is capped, but total record growth is not. Add a
  configurable maximum record size and RecordTooLarge. TensorBoard's Rust
  reader already has this error.
- String::from_utf8_lossy silently changes tags, plugin names, file versions,
  and hparam keys. Since the API promises String, invalid UTF-8 should be a
  decode error.
- TF2 histogram validation must ensure shape [k, 3], nonnegative k, and exactly
  k × 3 values.
- [Histogram::buckets](src/event.rs) does not discard empty exterior TF1
  buckets. TensorBoard deliberately trims those buckets before replacing
  sentinel edges with observed min/max:
  [official compatibility conversion](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/data_compat.py).
- Known-but-unsupported oneof fields must clear earlier alternatives. Currently
  graph/session/audio fields are ignored, so an earlier recognized payload can
  incorrectly survive even when a later oneof member should win.
- Unknown protobuf groups should be skipped, not reported as malformed, if
  forward compatibility is part of the contract.
- Reject at least NaN wall times, as TensorBoard's Rust reader does.
- Rename checksum fields from want/got to expected/actual.

### 4. Put a semantic layer above the wire model

The present model is halfway between raw protobuf and a user model:

- TF1 scalar is Simple.
- TF2 scalar is Tensor.
- TF1 image is Image.
- TF2 image is a generic string tensor.
- Text is another generic string tensor.
- Histogram access allocates tuples on every call.
- Consumers must understand plugin names and tensor shapes.

That logic has already leaked into Vertov's
[classification function](../vertov-model/src/project.rs).

The default API should normalize representation differences:

~~~rust
pub enum SummaryData<'a> {
    Scalar(f64),
    Histogram(HistogramView<'a>),
    Images(ImagesView<'a>),
    Text(TextView<'a>),
    Tensor(&'a RawTensor),
    Unsupported,
}

pub struct HistogramBucket {
    pub left: f64,
    pub right: f64,
    pub count: f64,
}
~~~

Users should not need to know whether a scalar was TF1 simple_value or a TF2
rank-zero tensor.

Keep the raw representation as an escape hatch, preferably under
tfevents::raw.

### 5. Resolve sparse metadata centrally

The official schema says metadata is commonly present only on the first value
for a tag:
[TensorFlow summary.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto).
TensorBoard's loader therefore remembers initial metadata by tag before
classifying values.

tfevents currently exposes each value independently, forcing every consumer to
recreate that cache or guess from tensor shape. The latter cannot reliably
distinguish TF2 text, images, and audio.

The semantic reader should remember metadata by tag and expose resolved
metadata. If the state must span rotated files, make the metadata catalog
transferable as an advanced API.

Also decode the missing standard metadata fields:

- display_name
- summary_description

These are part of the actual schema and useful to viewers.

## Proposed public API shape

A simple strict path should look approximately like this:

~~~rust
use tfevents::{Reader, SummaryData};

let mut reader = Reader::new(file); // verifies checksums

while let Some(event) = reader.read_event()? {
    for value in event.values() {
        match value.data() {
            SummaryData::Scalar(x) => {
                println!("{}\t{}\t{x}", event.step(), value.tag());
            }
            SummaryData::Histogram(histogram) => {
                for bucket in histogram.buckets() {
                    // ...
                }
            }
            _ => {}
        }
    }
}

let checkpoint = reader.checkpoint();
~~~

Design choices:

- Rename EventFileReader to Reader or EventReader; it accepts any Read, not
  only files.
- Add Event::values() so the common path does not pattern-match EventPayload.
- Separate errors structurally into record-local and stream-terminal failures.
- Make Checkpoint a small newtype around the byte offset.
- Replace the into_inner footgun with into_parts(), returning the stream,
  checkpoint, and pending-tail state.
- Change hparams from the nested Option<Result<...>> to:

~~~rust
pub fn hparams(&self) -> Result<Option<Hparams>, HparamsError>;
~~~

- Make public structs non-exhaustive or fields private before consumers depend
  on struct literals.
- Move CRC/framing types into raw.
- Remove the public [writer module](src/writer.rs), or expose it only through a
  test-util feature. A half-supported writer conflicts with the crate's
  reader-only boundary.

## Prioritized roadmap

| Priority | Work |
| --- | --- |
| P0 — truthful correctness | Safe checksum default; scalar-shape panic; record limit; UTF-8 validation; histogram conversion; oneof semantics; histogram size validation; NaN handling. |
| P1 — API reset | Result<Option<Event>>; available(); semantic TF1/TF2 normalization; resolved metadata; record-local versus terminal errors; cleaner checkpoint/parts API. |
| P2 — compatibility proof | Real fixtures from distinct encoders, differential testing against TensorBoard/prost, adversarial tests, full accessor fuzzing, publishable package. |
| P3 — measured performance | Benchmark verified and fast modes separately; optimize CRC and allocations only after profiling. |
| P4 — selective breadth | Semantic TF2 images/text, then audio. Add other Event payloads only when a viewer has a concrete use. |

Performance improvements worth measuring include slicing-by-8 CRC, an inline
record header, and payload-buffer reuse. The current
[benchmark](benches/cold_load.rs) only measures the unchecked event path, so it
cannot yet justify making that path the default.

## Compatibility and release gates

The current foundation is healthy: 40 tests pass, clippy and docs pass, Rust
1.88 works, and there are zero normal dependencies.

But the evidence is narrower than the README implies:

- The only recorded producer is tensorboardX; TF2 coverage is mostly
  synthetic.
- Prefer fixtures from genuinely distinct encoders: TensorFlow 2/Keras,
  PyTorch, tensorboardX, and JAX. Lightning and Hugging Face often delegate to
  another writer, so prioritize encoding diversity over brand count.
- Generate a semantic digest with TensorBoard's official loader and compare
  Rust output against it.
- Add cases for every byte truncation boundary, valid-protobuf CRC corruption,
  invalid UTF-8, giant lengths, negative/overflowing shapes,
  reordered/duplicate oneofs, sparse metadata, and empty exterior histogram
  buckets.
- The fuzz target claims to exercise every accessor but omits histogram
  normalization: [decode_event.rs](fuzz/fuzz_targets/decode_event.rs).
- cargo package --list omits the external fixtures referenced by
  [tests/fixtures.rs](tests/fixtures.rs), as well as the workspace license
  files. The published source package would not have a self-contained test
  suite.
- There are currently no doctests. The primary API example should compile in
  CI.
- Pin the formatting toolchain; the current formatter reports diffs.

## What I would deliberately not add

- No async API yet.
- No general TensorFlow protobuf surface.
- No logdir discovery, aggregation, downsampling, or chart concerns.
- No production writer unless the crate explicitly expands its mission.
- No dependency-heavy abstraction framework.
- No unchecked integrity default in pursuit of a benchmark number.

The winning product is not “TensorBoard in Rust.” It is:

> The smallest dependable way to consume a live TensorBoard event stream as
> semantic Rust values.
