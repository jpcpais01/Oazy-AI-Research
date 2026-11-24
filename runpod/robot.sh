#!/usr/bin/env bash
# Ballroom Robot: a cart with a cannon learns, from pixels, to throw the ball so it rests on a target.
#   bash runpod/robot.sh
#   MIN=20 NOISE=0.0001 THROWS=400000 WORLD=/path/eye.pt bash runpod/robot.sh
# Trains on rooms with 2-3 platforms, then fires on fresh rooms with 2-3 and 4-5, at 0.01% and 1% body noise.
# Imagines with the exact simulator, and with the pixel world model from runpod/eye.sh when there is one.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS="${RUNS:-$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))" 2>/dev/null || echo /workspace/oazy_runs)}/ballroom"
mkdir -p "$RUNS/robot"
WORLD="${WORLD:-$RUNS/eye/eye.pt}"
python3 -m ballroom.robot --minutes "${MIN:-20}" --throws "${THROWS:-400000}" --plat "${PLAT:-2,3}" --noise "${NOISE:-0.0001}" \
    --world "$WORLD" --out "$RUNS/robot" 2>&1 | tee "$RUNS/robot/robot.log"
