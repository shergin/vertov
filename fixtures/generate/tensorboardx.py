#!/usr/bin/env python3
"""Record the `fixtures/tensorboardx/` fixture: a small real tfevents file
written by tensorboardX's SummaryWriter (TF1-style summaries: simple_value
scalars, HistogramProto histograms, Summary.Image images).

Values follow fixed formulas so tests can assert them exactly; wall times are
whatever the recording machine's clock said — fixtures are recorded, never
synthesized.

Usage: python tensorboardx.py [outdir]   (default: fixtures/tensorboardx)
Requires: pip install tensorboardX numpy pillow
"""

import math
import shutil
import sys
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter

STEPS = 20
SPIKE_STEP = 13
SPIKE_VALUE = 25.0


def loss(step: int) -> float:
    if step == SPIKE_STEP:
        return SPIKE_VALUE  # the spike no viewer may hide
    return 4.0 * math.exp(-0.25 * step) + 0.5


def accuracy(step: int) -> float:
    return 1.0 - math.exp(-0.2 * step) * 0.9


def record_hparams(outdir: Path) -> None:
    """A separate fixture dir: add_hparams writes the hparams-plugin markers
    (experiment, session_start_info with typed values, session_end_info) into
    a sub-run named `hparam-session`."""
    if outdir.exists():
        shutil.rmtree(outdir)
    writer = SummaryWriter(logdir=str(outdir), flush_secs=10**6)
    writer.add_hparams(
        {"lr": 0.001, "optimizer": "adam", "amsgrad": True, "layers": 4},
        {"metrics/final_loss": 0.75},
        name="hparam-session",
    )
    writer.close()
    for file in sorted(outdir.rglob("*")):
        if file.is_file():
            print(f"{file.relative_to(outdir)}: {file.stat().st_size} bytes")


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "tensorboardx")
    record_hparams(outdir.parent / "tensorboardx-hparams")
    if outdir.exists():
        shutil.rmtree(outdir)
    writer = SummaryWriter(logdir=str(outdir), flush_secs=10**6)
    for step in range(STEPS):
        writer.add_scalar("train/loss", loss(step), step)
        writer.add_scalar("train/accuracy", accuracy(step), step)
        if step % 5 == 0:
            values = np.linspace(-1.0, 1.0, 101) * (1.0 + step / 10.0)
            writer.add_histogram("params/weights", values, step)
    writer.add_text("notes", "fixture recorded by tensorboardx.py", 0)
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    image[0, :, :] = 255  # solid red 4x4
    writer.add_image("samples/red", image, 0)
    writer.close()
    for file in sorted(outdir.iterdir()):
        print(f"{file.name}: {file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
