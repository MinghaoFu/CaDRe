#!/usr/bin/env bash
# Train / evaluate on ERSST v5 (§5.2, Table 6).
#
# ERSST is loaded by the downscale-CESM2 path; place ERSST_v5_*.nc files
# under $DATA_DIR/ERSST/ and adjust the data-path constants at the top of
# SSM/test_ds_CESM2.py before running.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATA_DIR:?Set DATA_DIR (must contain ERSST/)}"

python SSM/test_ds_CESM2.py
