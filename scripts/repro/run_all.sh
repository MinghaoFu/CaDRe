#!/usr/bin/env bash
# Sequential full reproduction. Expect this to take many GPU-hours.

set -euo pipefail
HERE="$(dirname "$0")"

"$HERE/run_synthetic.sh"
"$HERE/run_neural_baselines.sh"
"$HERE/run_higher_order.sh"
"$HERE/run_causalrivers.sh"
"$HERE/run_cesm2.sh"
"$HERE/run_weatherbench.sh"
"$HERE/run_ersst.sh"
"$HERE/run_forecasting.sh"
"$HERE/run_visualize.sh"
