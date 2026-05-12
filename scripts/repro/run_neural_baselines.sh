#!/usr/bin/env bash
# Neural causal-discovery baselines vs CaDRe at d_z=3, d_x in {3,6,8,10}.
# Reproduces Fig. 8 in the appendix.
#
# These baselines were run from the rebuttal companion (not in this
# repository by default). The driver script lives at:
#   baseline_exp/run_all_v2.py
# in our internal rebuttal directory; copy it into this repo's
# `baseline_exp/` to enable. The CaDRe side is identical to run_synthetic.sh.

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -f baseline_exp/run_all_v2.py ]; then
  echo ">>> Running DYNOTEARS / CUTS / Rhino / Jacobian-CD baselines"
  python baseline_exp/run_all_v2.py ${EXTRA_ARGS:-}
else
  echo "[skip] baseline_exp/run_all_v2.py not present in this checkout."
  echo "       See the rebuttal-companion repository for the baseline driver."
fi

echo ">>> Running CaDRe at the same grid"
./scripts/repro/run_synthetic.sh
