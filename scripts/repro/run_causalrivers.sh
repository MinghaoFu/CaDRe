#!/usr/bin/env bash
# External validation on the CausalRivers benchmark (Stein et al., 2025).
# Reproduces Fig. 9(b) in the appendix.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"
: "${DATA_DIR:?Set DATA_DIR to your data root}"

# The CausalRivers benchmark expects its own data layout under
# $DATA_DIR/CausalRivers/. The submodule in causalrivers_exp/ contains the
# benchmark utilities.
python -m causalrivers_exp.causalrivers.run_baselines \
  --data-root "$DATA_DIR/CausalRivers" \
  --methods dynotears cuts rhino jacobian-cd var pcmci cadre \
  ${EXTRA_ARGS:-}
