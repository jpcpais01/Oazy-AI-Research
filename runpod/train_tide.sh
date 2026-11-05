#!/usr/bin/env bash
# Start (or resume) Tide training in one tmux session, next to the Engram experiment:
#   window "tide"   - training Tide (same FineWeb-Edu data, order and schedule as Engram and the GPT)
#   window "gpt"    - the GPT-2 baseline, only if it has not been trained in this runs folder yet
#   window "dash"   - the live dashboard on port 8000, only if nothing is serving it already
#
#   bash runpod/train_tide.sh                    # extra arguments go to the trainer(s)
#   bash runpod/train_tide.sh --tokens 2.5e9     #   e.g. the budget suggested by --bench
#
# Data and runs are shared with Engram (/workspace/engram_data, /workspace/engram_runs), so the
# dashboard shows Tide, Engram and the GPT side by side. Needs  bash runpod/setup_engram.sh  once.
# With 2+ GPUs Tide gets GPU 0 and the GPT GPU 1 (set TIDE_GPU=1 to move Tide while Engram trains on 0).
# Re-running after a stop or a pod restart resumes from the checkpoints.
# Attach with:  tmux attach -t tide      (detach again with Ctrl-b then d)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
DATA="${ENGRAM_DATA:-/workspace/engram_data}"
RUNS="${ENGRAM_RUNS:-/workspace/engram_runs}"
EXTRA="$*"
mkdir -p "$RUNS"
if [ ! -f "$DATA/train.bin" ]; then
    echo "no training data in $DATA yet: run  bash runpod/setup_engram.sh  first"
    exit 1
fi
NGPU=$( (nvidia-smi -L 2>/dev/null || true) | grep -c "^GPU" || true)
GPU_T="${TIDE_GPU:-0}"
if [ "${NGPU:-1}" -ge 2 ]; then GPU_G=$(( (GPU_T + 1) % NGPU )); else GPU_G=0; fi
echo "GPUs found: ${NGPU}. Tide on GPU ${GPU_T}."

if tmux has-session -t tide 2>/dev/null; then
    echo "a 'tide' tmux session is already running; attach with: tmux attach -t tide"
    exit 0
fi
tmux new-session -d -s tide -n tide -c "$REPO"
tmux send-keys -t tide:tide "CUDA_VISIBLE_DEVICES=$GPU_T python3 -m engram.train --model tide --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/tide.log" C-m
if [ ! -f "$RUNS/gpt/ckpt.pt" ] && ! tmux has-session -t engram 2>/dev/null; then
    echo "no GPT baseline in $RUNS yet: training it too (GPU ${GPU_G})"
    tmux new-window -t tide: -n gpt -c "$REPO"
    tmux send-keys -t tide:gpt "CUDA_VISIBLE_DEVICES=$GPU_G python3 -m engram.train --model gpt --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/gpt.log" C-m
fi
if ! (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    tmux new-window -t tide: -n dash -c "$REPO"
    tmux send-keys -t tide:dash "python3 -m engram.dashboard --runs $RUNS --port 8000" C-m
fi
tmux select-window -t tide:tide
echo "started. dashboard: port 8000.  attach with: tmux attach -t tide"
