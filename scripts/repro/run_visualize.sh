#!/usr/bin/env bash
# Climate visualizations and qualitative analyses. Produces Figs. 5, 10, 11.
# These are notebook-driven; we shell out to nbconvert so they can be run
# headlessly from this script.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"
: "${DATA_DIR:?Set DATA_DIR}"

for NB in \
  analyze/WeatherBench_wind.ipynb \
  analyze/init_downscale_CESM2.ipynb \
  analyze/draw_comp.ipynb \
; do
  if [ -f "$NB" ]; then
    echo ">>> Running $NB"
    jupyter nbconvert --to notebook --execute "$NB" --output "$(basename "$NB" .ipynb)_executed.ipynb"
  fi
done
