#!/usr/bin/env bash
# Ballroom Eye: the ball-in-a-room task from pixels, stepping in the model's own inner space.
#   bash runpod/eye.sh
#   MIN=20 RES=128 PATCH=16 D=192 UNROLL=24 WINDOWS=4 BATCH=1024 INIT=/path/eye.pt COMPILE=reduce-overhead bash runpod/eye.sh
# Trains on rooms with 2-3 platforms, then scores on the same fresh rooms as ballroom.sh (2-3 and 4-5 platforms).
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS="${RUNS:-$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))" 2>/dev/null || echo /workspace/oazy_runs)}/ballroom/eye"
mkdir -p "$RUNS"
python3 -m ballroom.eye --minutes "${MIN:-20}" --rooms "${ROOMS:-100000}" --plat "${PLAT:-2,3}" --res "${RES:-128}" --patch "${PATCH:-16}" \
    --d "${D:-192}" --enc "${ENC:-4}" --step "${STEP:-3}" --unroll "${UNROLL:-24}" --windows "${WINDOWS:-4}" --batch "${BATCH:-1024}" ${INIT:+--init "$INIT"} ${COMPILE:+--compile "$COMPILE"} --out "$RUNS" 2>&1 | tee "$RUNS/eye.log"
if [ -f "$RUNS/../test.log" ]; then echo; echo "For comparison, the object models on the same rooms:"; cat "$RUNS/../test.log"; fi
