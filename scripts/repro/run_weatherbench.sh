#!/usr/bin/env bash
# Train + evaluate on WeatherBench. Reproduces WeatherBench rows of Table 5
# and the wind-pattern visualizations in §5.2.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"
: "${DATA_DIR:?Set DATA_DIR to your data root (must contain WeatherBench/)}"

python SSM/test_WB.py ${EXTRA_ARGS:-}
