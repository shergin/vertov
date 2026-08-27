# vertov

**TensorBoard for people who live in ssh.** A read-only terminal viewer for ML
training runs: point it at a logdir and get live charts, a runs table, and
comparison views — rendered by [malevich](https://github.com/shergin/malevich),
reading the files that trainers already write. No server, no SDK, no browser.

```sh
vertov runs/                  # TUI: runs table, scalars, live tail
vertov show runs/ -t loss     # headless: chart to the terminal, kaz-style
vertov export runs/ --csv     # escape hatch: flat runs × (params + metrics)
```

Early days. The vision and its rules live in [docs/vision.md](docs/vision.md);
each principle is argued in [docs/principles/](docs/principles/).

## Workspace

- [`crates/tfevents`](crates/tfevents) — standalone TFRecord + Event decoding:
  zero dependencies, hand-rolled varint/CRC32C, truncation-tolerant, resumable,
  fuzzed.
- [`crates/vertov-model`](crates/vertov-model) — the unified data model:
  catalog, exact mergeable summaries for every series, restart segments with
  RustBoard's preemption semantics (ghost tails kept), the reload loop, and an
  on-disk summary cache for warm starts.
- [`crates/vertov`](crates/vertov) — the binary. `vertov <logdir>` opens the
  TUI: a sortable runs table with a predicate filter bar
  (`lr > 1e-3 and status == active`) and HiPlot-style keep/exclude
  refinement; scalars with smoothing, log-y, ghost (pre-restart) tails, and
  an x axis switchable between step, wall, relative, and tokens (mapped
  through the run's token-counter series); a compare grid of small
  multiples on a shared domain; the flat hparams × metrics table; and
  histogram distributions as ridgelines. Live polling never loses
  cursor/selection/filter state, every view exports to CSV on one
  keystroke, and chart panels draw as real antialiased images where the
  terminal speaks sixel, kitty, or iTerm2 graphics (cell glyphs as the
  honest fallback). Headless: `show`, `tail`, `ls`, `summary`, `export`
  in text, CSV, or JSON. Build it with `cargo run -p vertov`.
- [`fixtures/`](fixtures) — recorded logs from real writers, checked in with
  the scripts that made them.

## License

MIT or Apache-2.0, at your option.
