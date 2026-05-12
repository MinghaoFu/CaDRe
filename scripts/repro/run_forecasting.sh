#!/usr/bin/env bash
# Forecasting comparison against time-series baselines (Autoformer,
# TimesNet, TimeMixer, TimeXer, iTransformer, FITS, CARD, MICN, Timer-XL,
# xLSTM-Mixer). Reproduces Tables 5, 18, 19, 20.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"
: "${DATA_DIR:?Set DATA_DIR}"

for DS in CESM2 ERSST Weather; do
  for HORIZON in 96 192 336; do
    echo ">>> Forecasting: dataset=$DS horizon=$HORIZON"
    python -m forecasting.read_data --dataset "$DS" --horizon "$HORIZON" \
      ${EXTRA_ARGS:-}
  done
done
