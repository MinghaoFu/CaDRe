#!/usr/bin/env bash
# Synthetic identifiability experiments (§5.1, Tables 11-15, Fig. 4).
#
# The synthetic trainer is `General/train_syn.py`. It is config-driven:
# the dimension (d_x, d_z), sparsity regime, and noise schedule live in
# `General/nonparam.yaml`. To reproduce a specific row of the paper:
#
#   1. Edit `General/nonparam.yaml` to set d_x, d_z, and the regime.
#   2. Pick an experiment tag for the log directory (`-e <tag>`).
#   3. Pick a seed (`-s <seed>`). The paper reports 5-seed averages.
#
# This wrapper sweeps five seeds and keeps the YAML at its default.
# Change the YAML and re-run for each configuration.

set -euo pipefail
cd "$(dirname "$0")/../.."
EXP_TAG="${EXP_TAG:-cadre_synthetic}"

for SEED in 1 2 3 4 5; do
  echo ">>> Synthetic: tag=$EXP_TAG seed=$SEED"
  python General/train_syn.py -e "$EXP_TAG" -s "$SEED" ${EXTRA_ARGS:-}
done
