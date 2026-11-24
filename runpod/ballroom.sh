#!/usr/bin/env bash
# Ballroom: train the ball-in-a-room models, score them, open the room editor.
#   bash runpod/ballroom.sh
#   STEP_MIN=10 SHOT_MIN=3 ROOMS=100000 PLAT=2,3 D=128 LAYERS=3 PORT=8000 bash runpod/ballroom.sh
# Trains on rooms with 2-3 platforms, tests on fresh rooms with 2-3 and with 4-5.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS="${RUNS:-$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))" 2>/dev/null || echo /workspace/oazy_runs)}/ballroom"
mkdir -p "$RUNS"
COMMON=(--rooms "${ROOMS:-100000}" --plat "${PLAT:-2,3}" --batch "${BATCH:-4096}" --d "${D:-128}" --layers "${LAYERS:-3}" --out "$RUNS")
python3 -m ballroom.train --what step --minutes "${STEP_MIN:-10}" --refresh 1 "${COMMON[@]}" 2>&1 | tee "$RUNS/step.log"
python3 -m ballroom.train --what shot --minutes "${SHOT_MIN:-3}" "${COMMON[@]}" 2>&1 | tee "$RUNS/shot.log"
python3 -m ballroom.test --runs "$RUNS" 2>&1 | tee "$RUNS/test.log"
python3 -m ballroom.export --runs "$RUNS"
echo
echo "Room editor: open the pod's port ${PORT:-8000} (Ctrl+C stops it; start again with: python3 -m ballroom.editor --port ${PORT:-8000})"
python3 -m ballroom.editor --port "${PORT:-8000}"
