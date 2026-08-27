#!/usr/bin/env python3
"""Record the `fixtures/dvclive/` fixture: a real dvclive run — TSV metric
history under `plots/metrics/`, `params.yaml`, `metrics.json`.

Values follow fixed formulas so tests can assert them exactly.

Usage: python dvclive_run.py [outdir]   (default: fixtures/dvclive)
Requires: pip install dvclive
"""

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dvclive import Live

STEPS = 12


def main() -> None:
    outdir = Path(
        sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "dvclive"
    ).resolve()
    if outdir.exists():
        shutil.rmtree(outdir)
    # Record in a scratch directory far outside any git repository — DVC
    # walks up from the cwd and would otherwise scaffold `.dvc/` into the
    # enclosing repo — then copy the run into the fixture tree.
    scratch = Path(tempfile.mkdtemp(prefix="vertov-dvclive-record-"))
    exp = scratch / "exp"
    exp.mkdir(parents=True)
    os.chdir(exp)
    with Live(dir="dvclive", save_dvc_exp=False, report=None) as live:
        live.log_param("lr", 0.005)
        live.log_param("optimizer", "adamw")
        for step in range(STEPS):
            live.log_metric("train/loss", 6.0 * math.exp(-0.4 * step) + 0.25)
            live.log_metric("train/accuracy", 1.0 - 0.8 * math.exp(-0.3 * step))
            live.next_step()
    # Keep the run itself (dvclive/ and dvc.yaml); DVC's repo scaffolding
    # (.dvc/, .git droppings) stays behind in the scratch dir.
    (outdir / "exp").mkdir(parents=True)
    shutil.copytree(exp / "dvclive", outdir / "exp" / "dvclive")
    if (exp / "dvc.yaml").exists():
        shutil.copy(exp / "dvc.yaml", outdir / "exp" / "dvc.yaml")
    os.chdir(Path(__file__).parent)
    shutil.rmtree(scratch)
    for file in sorted(outdir.rglob("*")):
        if file.is_file():
            print(f"{file.relative_to(outdir)}: {file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
