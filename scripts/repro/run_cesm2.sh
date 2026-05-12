#!/usr/bin/env bash
# Train CaDRe on the CESM2 Pacific SST dataset, then evaluate against the
# wind-field surrogate. Reproduces the CESM2 rows of Tables 5 and 6.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"
: "${DATA_DIR:?Set DATA_DIR to your data root (must contain CESM2/)}"
: "${CKPT_DIR:=$PROJECT_ROOT/checkpoints}"
: "${LOG_DIR:=$PROJECT_ROOT/logs}"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

echo ">>> Training CaDRe on CESM2 Pacific SST"
python SSM/train_CESM2.py ${EXTRA_ARGS:-}

echo ">>> Evaluating against WeatherBench wind-field surrogate"
python SSM/test_CESM2.py ${EXTRA_ARGS:-}
