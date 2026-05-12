#!/usr/bin/env bash
# Synthetic identifiability sweep, paper §5.1 / Tables 11-15 / Fig. 4.
# Trains CaDRe on simulated nonparametric SEM time series and reports
# SHD, TPR, Precision, MCC(z_t), MCC(s_t), R^2.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"
: "${DATA_DIR:?Set DATA_DIR to your data root}"
: "${LOG_DIR:=$PROJECT_ROOT/logs}"

mkdir -p "$LOG_DIR"

# Sweep observation dimension d_x (Table 12)
for DX in 3 6 8 10; do
  echo ">>> Synthetic: d_z=3 d_x=$DX"
  python General/train_syn.py \
    --config General/nonparam.yaml \
    --dx "$DX" --dz 3 --seed 1 \
    --log-dir "$LOG_DIR/synthetic_dx${DX}" \
    ${EXTRA_ARGS:-}
done

# Sparsity regimes (Table 3)
for REGIME in independent sparse dense; do
  echo ">>> Synthetic: regime=$REGIME"
  python General/train_syn.py \
    --config General/nonparam.yaml \
    --dx 10 --dz 3 --regime "$REGIME" --seed 1 \
    --log-dir "$LOG_DIR/synthetic_${REGIME}" \
    ${EXTRA_ARGS:-}
done
