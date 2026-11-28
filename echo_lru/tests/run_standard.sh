#!/usr/bin/env bash
# Golden check and benchmark of the built plan operators.
#
#   bash run_standard.sh                 # device 2, standard matrix, all candidates
#   bash run_standard.sh 3               # device 3
#   bash run_standard.sh 3 --methods plan
#   bash run_standard.sh 3 --samples 5 --profile-steps 1
#
# Before running: `npu-smi info` and pick an idle card. The ordered-input
# operators need the ordered QLI package: export KV_ORDER_ENV=<its set_env.bash>.
# The remaining arguments are passed to bench_unified.py.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

DEVICE="${1:-2}"
shift || true

python3 bench_unified.py --orchestrate --device "${DEVICE}" \
  --datasets glm,constructed \
  --batches 8,16 --seqs 102400,358400 --mtps 0,1,3 \
  --scenarios cold_reset,full_cache_90pct \
  --samples 10 --warmup 3 --profile-steps 2 \
  "$@"
