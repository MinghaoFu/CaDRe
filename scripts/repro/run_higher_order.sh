#!/usr/bin/env bash
# Higher-order Markov latent dynamics (L=2). Reproduces Table 16.

set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PROJECT_ROOT:=$PWD}"

for DX in 3 6 8 10; do
  echo ">>> Higher-order Markov: L=2 d_z=3 d_x=$DX"
  python General/train_syn.py \
    --config General/nonparam.yaml \
    --dx "$DX" --dz 3 --markov-order 2 --seed 1 \
    ${EXTRA_ARGS:-}
done
