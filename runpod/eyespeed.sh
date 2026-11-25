#!/usr/bin/env bash
# Speed test for the pixel model (about 5 minutes; run it when nothing else uses the GPU):
#   bash runpod/eyespeed.sh
# Tries batch sizes and compile modes, prints examples per second for each. The fastest line wins.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
N="${N:-40}"; VIEW="${VIEW:-2}"
run() {  # batch compile
  python3 -m ballroom.eye --bench "$N" --rooms 20000 --batch "$1" ${2:+--compile "$2"} --view "${3:-$VIEW}" ${4:-} --out /tmp/eyespeed 2>&1 \
    | grep -E "BENCH|Error|error" | tail -1 || true
}
echo "compare 'moments learned/s' (higher = faster learning)"
echo "speed test, view $VIEW, $N steps each (only the second half is timed)"
# run 1024 "" 0 (no view: 1.81 steps/s on the 4090)
# run 1024 "" (room view, plain: 1.10 steps/s on the 4090)
run 1024 default              # room view, compiled (fastest so far: 1.68 steps/s on the 4090)
run 1024 reduce-overhead      # room view, compiled + CUDA graphs
run 1024 "" "" "--fast 4 --fast-batch 4096"          # parallel training: short chains side by side
run 1024 "" "" "--fast 4 --fast-batch 8192"
run 1024 default "" "--fast 4 --fast-batch 4096"
run 1024 reduce-overhead "" "--fast 4 --fast-batch 4096"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
