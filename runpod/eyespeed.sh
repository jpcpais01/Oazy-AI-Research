#!/usr/bin/env bash
# Speed test for the pixel model (about 5 minutes; run it when nothing else uses the GPU):
#   bash runpod/eyespeed.sh
# Tries batch sizes and compile modes, prints examples per second for each. The fastest line wins.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
N="${N:-40}"; VIEW="${VIEW:-2}"
run() {  # batch compile
  python3 -m ballroom.eye --bench "$N" --rooms 20000 --batch "$1" ${2:+--compile "$2"} --view "${3:-$VIEW}" --out /tmp/eyespeed 2>&1 \
    | grep -E "BENCH|Error|error" | tail -1 || true
}
echo "speed test, view $VIEW, $N steps each (only the second half is timed)"
run 1024 "" 0                 # without the room view, for reference
run 1024 ""                   # room view, plain
run 1024 default              # room view, compiled
run 1024 reduce-overhead      # room view, compiled + CUDA graphs
run 2048 reduce-overhead
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
