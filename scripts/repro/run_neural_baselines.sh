#!/usr/bin/env bash
# Neural causal-discovery baselines (DYNOTEARS, CUTS, Rhino, Jacobian-CD)
# vs CaDRe at d_z=3, d_x in {3,6,8,10}. Reproduces Fig. 8 in the appendix.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"

# Baselines: install via the bundled subrepos (see baseline_exp/ in the
# rebuttal companion repo if these are missing locally).
echo ">>> Running DYNOTEARS / CUTS / Rhino / Jacobian-CD baselines"
python -m baseline_exp.run_all_v2 --dz 3 --dx-grid 3,6,8,10 --seeds 5 \
  ${EXTRA_ARGS:-}

# CaDRe at the same grid (uses the same synthetic generator)
echo ">>> Running CaDRe at the same grid"
for DX in 3 6 8 10; do
  python General/train_syn.py \
    --config General/nonparam.yaml \
    --dx "$DX" --dz 3 --seed 1 \
    ${EXTRA_ARGS:-}
done
