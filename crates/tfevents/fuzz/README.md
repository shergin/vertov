# tfevents fuzzing

Three cargo-fuzz targets, one per parsing layer, so coverage guidance
reaches each decoder without tunneling through the layer above (framing's
CRC gate in particular):

- `read_events` — arbitrary bytes as a whole file: framing + decode +
  accessors, the full `EventFileReader` loop.
- `decode_event` — arbitrary bytes as one record payload: the proto layer
  without the CRC gate.
- `decode_hparams` — arbitrary bytes as hparams plugin content: two message
  layers further down.

The contract: no panic, no hang, no unbounded allocation — hostile input
must surface as the error taxonomy, never as a crash.

Run (nightly; `cargo install cargo-fuzz` once), from `crates/tfevents/`:

```sh
mkdir -p fuzz/corpus/{read_events,decode_event,decode_hparams}
cargo +nightly fuzz run read_events fuzz/corpus/read_events fuzz/seeds/read_events
cargo +nightly fuzz run decode_event fuzz/corpus/decode_event fuzz/seeds/decode_event
cargo +nightly fuzz run decode_hparams
```

PATH gotcha on this machine: Homebrew's stable cargo shadows rustup's shim,
and cargo-fuzz's inner `cargo build` resolves through PATH — if the build
fails with "1 nightly option were parsed", prefix the command with
`PATH="$HOME/.cargo/bin:$PATH"`.

`fuzz/seeds/` holds checked-in starting inputs cut from the recorded
fixtures (`fixtures/generate/fuzz_seeds.py` regenerates them); the growing
corpus and any crash artifacts stay untracked. The Phase 1 exit bar is a
24-hour soak across the targets.
