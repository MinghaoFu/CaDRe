#!/usr/bin/env bash
# Train CaDRe on ERSST v5 SST anomalies (1880-present, monthly).
# Reproduces ERSST rows of Table 6.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"
: "${DATA_DIR:?Set DATA_DIR to your data root (must contain ERSST/)}"

# ERSST is loaded by the same downscale path as CESM2.
python SSM/test_ds_CESM2.py --dataset ERSST ${EXTRA_ARGS:-}
