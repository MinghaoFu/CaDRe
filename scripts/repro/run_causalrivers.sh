#!/usr/bin/env bash
# External validation on the CausalRivers benchmark (App. D, Fig. 9(b)).
#
# The benchmark driver lives in causalrivers_exp/causalrivers/run_baselines.py
# (vendored from the upstream CausalRivers repo, see
# https://github.com/causalriversbenchmark/causalrivers ).
# Expected data layout: $DATA_DIR/CausalRivers/

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATA_DIR:?Set DATA_DIR}"

if [ -f causalrivers_exp/causalrivers/run_baselines.py ]; then
  python causalrivers_exp/causalrivers/run_baselines.py ${EXTRA_ARGS:-}
else
  echo "[skip] causalrivers_exp/ not present in this checkout."
  echo "       Clone https://github.com/causalriversbenchmark/causalrivers"
  echo "       into causalrivers_exp/causalrivers/ to enable."
fi
