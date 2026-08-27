#!/usr/bin/env python3
"""Record the `fixtures/mlflow/` fixture: a real MLflow file-store run —
`mlruns/<exp>/<run>/` with metric line files, param files, and meta.yaml.

Values follow fixed formulas so tests can assert them exactly.

Usage: python mlflow_run.py [outdir]   (default: fixtures/mlflow)
Requires: pip install mlflow-skinny

MLflow 3.15 put the filesystem backend in maintenance mode behind
MLFLOW_ALLOW_FILE_STORE=true; earlier versions (the bulk of installs)
write it by default. The on-disk layout is frozen either way — which is
exactly why a viewer can rely on it.
"""

import math
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow  # noqa: E402

STEPS = 10


def main() -> None:
    outdir = Path(
        sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "mlflow"
    ).resolve()
    if outdir.exists():
        shutil.rmtree(outdir)
    mlflow.set_tracking_uri(f"file://{outdir}/mlruns")
    with mlflow.start_run(run_name="warm-start-7"):
        mlflow.log_param("lr", 0.02)
        mlflow.log_param("optimizer", "sgd")
        for step in range(STEPS):
            mlflow.log_metric("loss", 3.0 * math.exp(-0.5 * step) + 0.1, step=step)
            mlflow.log_metric("val/accuracy", 1.0 - math.exp(-0.4 * step), step=step)
    for file in sorted(outdir.rglob("*")):
        if file.is_file():
            print(f"{file.relative_to(outdir)}: {file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
