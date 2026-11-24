#!/usr/bin/env bash
# Ballroom: train the ball-in-a-room models, score them, open the room editor.
#   bash runpod/ballroom.sh
#   MIN=10 ARENA=65536 SHOT=1 PLAT=2,3 PORT=8000 ARMS="sharp:3:128:3 big:3:256:4" bash runpod/ballroom.sh
# Each arm is name:features:width:layers, and all arms train at the same time on the one GPU.
# Trains on rooms with 2-3 platforms, tests on fresh rooms with 2-3 and with 4-5. The best arm goes to the editor.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS="${RUNS:-$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))" 2>/dev/null || echo /workspace/oazy_runs)}/ballroom"
mkdir -p "$RUNS"
ARMS="${ARMS:-sharp:3:128:3 big:3:256:4}"
PLAT="${PLAT:-2,3}"
pids=()
for arm in $ARMS; do
    IFS=: read -r name feat d layers <<< "$arm"
    mkdir -p "$RUNS/$name"; rm -f "$RUNS/$name/step.pt"
    arena="${ARENA:-65536}"; [ "$d" -ge 256 ] && arena=$((arena / 2))   # wide arms take half the balls: memory
    python3 -m ballroom.train --what step --minutes "${MIN:-10}" --rooms "${ROOMS:-100000}" --plat "$PLAT" \
        --arena "$arena" --feat "$feat" --d "$d" --layers "$layers" --out "$RUNS/$name" > "$RUNS/$name/step.log" 2>&1 &
    pids+=($!)
done
# the one-shot model is trained once and kept; SHOT=1 trains it again
if [ "${SHOT:-0}" = 1 ] || [ ! -f "$RUNS/shot.pt" ]; then
    python3 -m ballroom.train --what shot --minutes "${MIN:-10}" --rooms "${SHOT_ROOMS:-300000}" --plat "$PLAT" --batch 4096 \
        --feat 3 --d 256 --layers 4 --out "$RUNS" > "$RUNS/shot.log" 2>&1 &
    pids+=($!)
else
    echo "saved (kept from an earlier run)" > "$RUNS/shot.log"
fi
echo "training $ARMS at once for ${MIN:-10} minutes; one line per arm every 30 s"
while [ -n "$(jobs -r)" ]; do
    sleep 30
    for arm in $ARMS; do name="${arm%%:*}"; printf '%-6s %s\n' "$name" "$(grep '^it ' "$RUNS/$name/step.log" | tail -1)"; done
    echo
done
wait || true
for arm in $ARMS; do
    name="${arm%%:*}"
    if grep -q '^saved ' "$RUNS/$name/step.log"; then tail -1 "$RUNS/$name/step.log"
    else echo "ARM $name DIED, last lines of its log:"; tail -4 "$RUNS/$name/step.log"; rm -f "$RUNS/$name/step.pt"; fi
done
grep -q 'saved ' "$RUNS/shot.log" || { echo "ONE-SHOT MODEL DIED, last lines of its log:"; tail -4 "$RUNS/shot.log"; }
python3 -m ballroom.test --runs "$RUNS" 2>&1 | tee "$RUNS/test.log"
python3 -m ballroom.export --runs "$RUNS"
echo
echo "Room editor: open the pod's port ${PORT:-8000} (Ctrl+C stops it; start again with: python3 -m ballroom.editor --port ${PORT:-8000})"
python3 -m ballroom.editor --port "${PORT:-8000}"
