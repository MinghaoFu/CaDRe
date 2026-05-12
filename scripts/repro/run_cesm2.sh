#!/usr/bin/env bash
# Train CaDRe on CESM2 Pacific SST and evaluate against the wind-field
# surrogate (§5.2, Tables 5 and 6).
#
# Requires $DATA_DIR/CESM2/CESM2_pacific_SST.pkl and the grouped/downscaled
# derivatives (produced by dataset/CESM2_analysis.ipynb).

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATA_DIR:?Set DATA_DIR (must contain CESM2/)}"

EXP_TAG="${EXP_TAG:-cadre_cesm2}"
SEED="${SEED:-1}"

echo ">>> Training CaDRe on CESM2 Pacific SST"
python SSM/train_CESM2.py -e "$EXP_TAG" -s "$SEED" ${EXTRA_ARGS:-}

echo ">>> Note: SSM/test_CESM2.py runs top-to-bottom (no CLI flags)."
echo ">>> Edit the CKP_PATH / RESULTS_SAVE_DIR constants at the top of that"
echo ">>> file to point at the checkpoint produced by the training step."
