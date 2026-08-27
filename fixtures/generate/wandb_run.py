#!/usr/bin/env python3
"""Record the `fixtures/wandb/` fixture: a real wandb offline run — the
`offline-run-*/run-*.wandb` transaction log plus its sidecar files.

Values follow fixed formulas so tests can assert them exactly.

Usage: python wandb_run.py [outdir]   (default: fixtures/wandb)
Requires: pip install wandb
"""

import math
import os
import shutil
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "offline"
os.environ.setdefault("WANDB_CONSOLE", "off")
import wandb  # noqa: E402

STEPS = 10


def main() -> None:
    outdir = Path(
        sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "wandb"
    ).resolve()
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    run = wandb.init(
        mode="offline",
        dir=str(outdir),
        project="vertov-fixture",
        config={"lr": 0.003, "optimizer": "adam", "amsgrad": True},
    )
    for step in range(STEPS):
        run.log(
            {
                "train/loss": 5.0 * math.exp(-0.35 * step) + 0.2,
                "train/accuracy": 1.0 - 0.7 * math.exp(-0.25 * step),
            },
            step=step,
        )
    run.finish()
    # Keep only the transaction log and the config sidecar: debug logs, the
    # latest-run symlink, and files/wandb-metadata.json carry host/user
    # details that do not belong in a fixture (and the reader ignores them).
    run_dirs = list((outdir / "wandb").glob("offline-run-*"))
    assert len(run_dirs) == 1, run_dirs
    keep = {"run-" + run_dirs[0].name.rsplit("-", 1)[1] + ".wandb"}
    for entry in sorted(outdir.rglob("*"), reverse=True):
        relative = entry.relative_to(outdir)
        if entry.is_file() or entry.is_symlink():
            if entry.name not in keep and relative.as_posix() != f"{run_dirs[0].relative_to(outdir).as_posix()}/files/config.yaml":
                entry.unlink()
        elif not any(entry.rglob("*")):
            entry.rmdir()
    for file in sorted(outdir.rglob("*")):
        if file.is_file():
            print(f"{file.relative_to(outdir)}: {file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
