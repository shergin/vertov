#!/usr/bin/env python3
"""Extract fuzz seed inputs from the recorded fixtures into
`crates/tfevents/fuzz/seeds/`. Dependency-free.

- read_events/: whole small files and a record-aligned prefix of the big one
  (libFuzzer's default max_len is 4096, so seeds stay under that).
- decode_event/: individual record payloads (framing stripped), one per
  payload kind the fixtures contain.

Usage: python fuzz_seeds.py
"""

import shutil
import struct
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
FIXTURES = ROOT / "fixtures"
SEEDS = ROOT / "crates" / "tfevents" / "fuzz" / "seeds"


def records(data: bytes):
    """Yields (offset, payload) for each complete TFRecord."""
    offset = 0
    while offset + 12 <= len(data):
        (length,) = struct.unpack("<Q", data[offset : offset + 8])
        end = offset + 12 + length + 4
        if end > len(data):
            return
        yield offset, data[offset + 12 : offset + 12 + length]
        offset = end


def events_file(dirname: str) -> Path:
    matches = [
        path
        for path in (FIXTURES / dirname).iterdir()
        if path.is_file() and "tfevents" in path.name
    ]
    assert len(matches) == 1, matches
    return matches[0]


def main() -> None:
    if SEEDS.exists():
        shutil.rmtree(SEEDS)
    main_file = events_file("tensorboardx").read_bytes()
    hparams_file = events_file("tensorboardx-hparams/hparam-session").read_bytes()

    read_events = SEEDS / "read_events"
    read_events.mkdir(parents=True)
    (read_events / "hparam-session").write_bytes(hparams_file)
    prefix_end = 0
    for offset, payload in records(main_file):
        if offset + 12 + len(payload) + 4 > 4000:
            break
        prefix_end = offset + 12 + len(payload) + 4
    (read_events / "scalars-prefix").write_bytes(main_file[:prefix_end])

    decode_event = SEEDS / "decode_event"
    decode_event.mkdir(parents=True)
    for source, name in ((main_file, "main"), (hparams_file, "hparams")):
        for index, (_, payload) in enumerate(records(source)):
            if index >= 8:
                break
            (decode_event / f"{name}-{index:02}").write_bytes(payload)

    for file in sorted(SEEDS.rglob("*")):
        if file.is_file():
            print(f"{file.relative_to(SEEDS)}: {file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
