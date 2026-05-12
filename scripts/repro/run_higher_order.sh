#!/usr/bin/env bash
# Higher-order Markov latent dynamics (App. D, Table 16).
#
# Set `LAG: 2` (or 3, 5) in `General/nonparam.yaml` before running.
# This wrapper sweeps five seeds.

set -euo pipefail
cd "$(dirname "$0")/../.."
EXP_TAG="${EXP_TAG:-cadre_higher_order_L2}"

for SEED in 1 2 3 4 5; do
  echo ">>> Higher-order: tag=$EXP_TAG seed=$SEED"
  python General/train_syn.py -e "$EXP_TAG" -s "$SEED" ${EXTRA_ARGS:-}
done
