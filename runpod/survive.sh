#!/usr/bin/env bash
# The creature (clean v2 recipe). Trains from scratch, saves a mind every 15 min, quick tests every 15 min.
#   MIN=300 RES=64 OUT=clean bash runpod/survive.sh
#   bench a mind:  python3 -m survive.bench /workspace/oazy_runs/survive/clean/mind_60min.pt
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=/workspace/oazy_runs/survive/${OUT:-clean}
mkdir -p "$RUNS"
python3 -m survive.train --minutes "${MIN:-300}" --res "${RES:-64}" --fovea "${FOVEA:-0}" --gaze "${GAZE:-0}" --eye-sep "${EYE_SEP:-0.08}" --light "${LIGHT:-0}" --horizon "${HORIZON:-15}" --intent "${INTENT:-0}" --start-stride "${START_STRIDE:-1}" --stamina "${STAMINA:-0}" --metab "${METAB:-1}" --life-cap "${LIFE_CAP:-100}" --g-slots "${G_SLOTS:-1}" --g-cls "${G_CLS:-16}" --back-speed "${BACK_SPEED:-1}" --heat "${HEAT:-1}" --drive "${DRIVE:-linear}" --matter "${MATTER:-0}" --ent-target "${ENT_TARGET:-nan}" --ent-per "${ENT_PER:-0}" --body-w "${BODY_W:-1}" --world-v "${WORLD_V:-1}" --long-fresh "${LONG_FRESH:-0}" --jumps "${JUMPS:-}" --jump-w "${JUMP_W:-0.5}" --recon "${RECON:-0}" --smell "${SMELL:-0}" --init "${INIT:-}" --envs "${ENVS:-512}" --batch "${BATCH:-256}" \
  --seq "${SEQ:-32}" --replay-gb "${REPLAY_GB:-8}" --every "${EVERY:-1}" --emotion "${EMOTION:-0}" --gamma "${GAMMA:-0.997}" --compile "${COMPILE:-1}" --seed "${SEED:-0}" --out "$RUNS" 2>&1 | tee -a "$RUNS/train.log"
# the yardstick: 1024 fresh lives in unseen, hard and training-like worlds, saved next to the mind
LAST=$(ls -t "$RUNS"/mind_*min.pt | head -1)
python3 -m survive.bench "$LAST" --lives "${LIVES:-1024}" 2>&1 | tee -a "$RUNS/train.log"
