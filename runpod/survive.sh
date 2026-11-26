#!/usr/bin/env bash
# The creature (clean v2 recipe). Trains from scratch, saves a mind every 15 min, quick tests every 15 min.
#   MIN=300 RES=64 OUT=clean bash runpod/survive.sh
#   bench a mind:  python3 -m survive.bench /workspace/oazy_runs/survive/clean/mind_60min.pt
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=/workspace/oazy_runs/survive/${OUT:-clean}
mkdir -p "$RUNS"
python3 -m survive.train --minutes "${MIN:-300}" --res "${RES:-64}" --envs "${ENVS:-256}" --batch "${BATCH:-64}" \
  --seq "${SEQ:-32}" --replay-gb "${REPLAY_GB:-8}" --every "${EVERY:-1}" --seed "${SEED:-0}" --out "$RUNS" 2>&1 | tee -a "$RUNS/train.log"
