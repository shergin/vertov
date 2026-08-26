# Benchmarks

Recorded results for the criterion suites, against dated hardware. Budgets
are commitments: a change that regresses one materially needs a reason in
its commit message. Run with `cargo bench -p tfevents`.

## tfevents cold load (`crates/tfevents/benches/cold_load.rs`)

One million TF1 scalar events (~47.6 MB) from memory — parser cost only;
real cold starts add I/O.

| bench | what | time | throughput |
|---|---|---|---|
| `frames_1m` | TFRecord framing only | 68 ms | ~700 MiB/s |
| `events_1m` | framing + Event decode + scalar extraction | 175 ms | ~273 MiB/s |

Recorded 2026-08-26, Apple M1 Pro, 32 GB, macOS 15 (Darwin 25.5), rustc
1.98.0, `--release` defaults.

Budget: a 1M-point series cold-loads in under 250 ms of parse time on this
class of hardware.

## vertov-model catalog scan (`crates/vertov-model/benches/scan.rs`)

A generated on-disk logdir: 1000 runs × 5 series × 100 points (500k points),
so numbers include real file I/O through the page cache.

| bench | what | time |
|---|---|---|
| `cold_1000_runs` | discover + ingest everything into summaries | 602 ms |
| `quiet_tick_1000_runs` | one refresh with nothing changed (walk + drain + stat) | 24 ms |
| `warm_1000_runs` | load summary cache + one verifying refresh, nothing re-read | 48 ms |

Recorded 2026-08-26, same hardware as above.

Budgets: a 1000-run cold scan under 1 s; a quiet poll tick under 100 ms —
comfortably inside a 5 s poll interval; a cache-hit warm start under 100 ms.
