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
run 1024 default                                     # room view, compiled: 1.68 steps/s on the 4090 (the current setup)
run 1024 default "" "--bf16 1"                       # + the imagining in bf16
run 1024 default "" "--glances 1"                    # + the eye checks only the last window
run 1024 default "" "--bf16 1 --glances 1"           # both
run 1024 default "" "--bf16 1 --glances 1 --room-patch 8 --walls 1"   # both + the room read in 8x8 squares, floor drawn
run 1024 default "" "--bf16 1 --glances 1 --res 64 --patch 8"    # + 64x64 pictures, same 64 patches
run 1024 default "" "--bf16 1 --glances 1 --res 96 --patch 16"   # + 96x96 pictures, 36 patches
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
