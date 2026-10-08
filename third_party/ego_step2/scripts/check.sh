#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m compileall -q src tests
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m hand_detection_sam3_mask --help >/dev/null
"$PYTHON_BIN" -m hand_detection_sam3_mask video --help >/dev/null
"$PYTHON_BIN" -m hand_detection_sam3_mask.download_models --help >/dev/null
for script in scripts/*.sh slurm/*.sbatch; do
  bash -n "$script"
done
