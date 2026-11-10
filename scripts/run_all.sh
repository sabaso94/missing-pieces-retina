#!/usr/bin/env bash
# Reproduce every result and figure: download + cache -> tests -> cohort report -> train -> predict
# -> E1 -> E2 -> E3 -> E5 -> E4 -> figures. Checkpoints that already exist are loaded instead of retrained.
# Usage: bash scripts/run_all.sh   (set DEVICE to choose the GPU; default cuda)
set -euo pipefail

cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-.venv/bin/python}"

bash scripts/download_data.sh
"$PYTHON" -m pytest -q
"$PYTHON" scripts/cohort_report.py
"$PYTHON" scripts/train.py --device "$DEVICE"
"$PYTHON" scripts/predict.py --device "$DEVICE"
"$PYTHON" scripts/run_e1.py
"$PYTHON" scripts/run_e2.py --device "$DEVICE"
"$PYTHON" scripts/run_e3.py
"$PYTHON" scripts/run_e5.py
"$PYTHON" scripts/run_e4.py --device "$DEVICE"
"$PYTHON" scripts/make_figures.py
