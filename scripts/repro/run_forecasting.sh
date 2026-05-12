#!/usr/bin/env bash
# Forecasting comparison against time-series baselines
# (Autoformer / TimesNet / TimeMixer / TimeXer / iTransformer / FITS / CARD /
#  MICN / Timer-XL / xLSTM-Mixer). Reproduces Tables 5, 18, 19, 20.
#
# The forecasting evaluation is notebook-driven:
#   forecasting/CESM2Forecast.ipynb
#   forecasting/DownscaledCESM2Forecast.ipynb
# Run them with jupyter nbconvert --execute.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATA_DIR:?Set DATA_DIR}"

for NB in \
  forecasting/CESM2Forecast.ipynb \
  forecasting/DownscaledCESM2Forecast.ipynb \
; do
  if [ -f "$NB" ]; then
    echo ">>> Executing $NB"
    jupyter nbconvert --to notebook --execute "$NB" \
      --output "$(basename "$NB" .ipynb)_executed.ipynb"
  fi
done
