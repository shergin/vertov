# tfevents

Read TensorBoard `tfevents` files: TFRecord framing and minimal Event
decoding. Zero dependencies, hand-rolled varint/CRC32C, `#![forbid(unsafe_code)]`.

tfevents is the de-facto interchange format of ML training logs — PyTorch
`SummaryWriter`, Lightning, HF Trainer, MMEngine, and TF/Keras all write it.
This crate is a minimal, maintained reader for it, built for the files
trainers actually leave behind:

- **A truncated tail is a state, not an error.** A live writer (or one that
  crashed mid-record) leaves a torn last record; the reader serves the valid
  prefix and resumes exactly where it stopped once the file grows.
- **Corruption is visible, never silent.** Payload checksums are skipped on
  the hot path (framing is guarded by the length CRC) and computed only when
  a record fails to decode — distinguishing corrupt bytes from an
  unsupported proto, per record, without giving up on the file.
- **Readers are resumable.** The committed byte offset is exposed and a
  reader can be reconstructed from it, so tailing a growing file is an
  incremental read and a warm start needs no re-parse.

What it decodes: TF1 and TF2 scalars (`simple_value` and rank-0 tensors),
histograms (both `HistogramProto` and `[k,3]` tensor form), images (TF1
message and TF2 blob form), text, summary metadata (plugin name, data
class), and typed hyperparameters from the hparams plugin. Everything else
is skipped at the wire level — unknown fields and payloads are forward
compatibility, not failures.

```rust
use std::fs::File;
use tfevents::{EventFileReader, EventPayload, ReadEventError};

let file = File::open("events.out.tfevents.1234.host")?;
let mut reader = EventFileReader::new(file);
loop {
    match reader.next_event() {
        Ok(event) => {
            if let EventPayload::Summary(values) = &event.payload {
                for value in values {
                    if let Some(scalar) = value.scalar() {
                        println!("{}\t{}\t{}", event.step, value.tag, scalar);
                    }
                }
            }
        }
        // End of what's on disk — a live writer may append more; keep the
        // reader and call again later.
        Err(ReadEventError::Truncated) => break,
        // Corrupt/Malformed spoil one record; the stream continues.
        Err(ReadEventError::Corrupt { .. } | ReadEventError::Malformed { .. }) => continue,
        // BadLengthCrc/Io end the file; its valid prefix stands.
        Err(err) => return Err(err.into()),
    }
}
```

Tests run against both hand-encoded wire bytes and recorded files written by
real writers (checked into the repository's `fixtures/` with the scripts
that made them).

Part of [vertov](https://github.com/shergin/vertov), a terminal viewer for
training runs. Field constants follow the frozen TensorFlow protos, checked
against TensorBoard's Rust data server.

## License

MIT or Apache-2.0, at your option.
