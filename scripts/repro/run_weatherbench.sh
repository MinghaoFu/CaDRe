#!/usr/bin/env bash
# Train / evaluate on WeatherBench (§5.2, Tables 5 and 6).
# Note: SSM/test_WB.py runs as a script, not via CLI flags; edit the
# constants at the top of that file to choose dataset paths and checkpoint.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATA_DIR:?Set DATA_DIR (must contain WeatherBench/)}"

python SSM/test_WB.py
