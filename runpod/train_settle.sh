#!/usr/bin/env bash
# Start (or resume) the whole Settle experiment in one tmux session:
#   window "dash"   - the live training dashboard on port 8000
#   window "settle" - training Settle
#   window "gpt"    - training the matched GPT baseline (same size, same data)
#
#   bash runpod/train_settle.sh            # Settle + GPT baseline; extra arguments go to the trainers, e.g. --tokens 1.5e8
#   bash runpod/train_settle.sh --solo     # only Settle, no baseline
#
# Re-running it after a stop or a pod restart resumes both runs from their checkpoints.
# Attach with:  tmux attach -t settle      (detach again with Ctrl-b then d)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
DATA="${SETTLE_DATA:-/workspace/settle_data}"
RUNS="${SETTLE_RUNS:-/workspace/settle_runs}"
SOLO=0
if [ "${1:-}" = "--solo" ]; then SOLO=1; shift; fi
EXTRA="$*"
mkdir -p "$RUNS"

# with 2 or more GPUs, Settle gets GPU 0 and the GPT baseline GPU 1; with one GPU they share it
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
if [ "${NGPU:-1}" -ge 2 ]; then GPU_S=0; GPU_G=1; else GPU_S=0; GPU_G=0; fi
echo "GPUs found: ${NGPU}. Settle on GPU ${GPU_S}, GPT on GPU ${GPU_G}."

if tmux has-session -t settle 2>/dev/null; then
    echo "a 'settle' tmux session is already running; attach with: tmux attach -t settle"
    exit 0
fi

tmux new-session -d -s settle -n dash -c "$REPO"
tmux send-keys -t settle:dash "python3 -m settle.dashboard --runs $RUNS --port 8000" C-m
tmux new-window -t settle -n settle -c "$REPO"
tmux send-keys -t settle:settle "CUDA_VISIBLE_DEVICES=$GPU_S python3 -m settle.train --model settle --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/settle.log" C-m
if [ "$SOLO" = 0 ]; then
    tmux new-window -t settle -n gpt -c "$REPO"
    tmux send-keys -t settle:gpt "CUDA_VISIBLE_DEVICES=$GPU_G python3 -m settle.train --model gpt --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/gpt.log" C-m
else
    echo "solo mode: training only Settle (no GPT baseline)"
fi
tmux select-window -t settle:settle
echo "started. dashboard: port 8000.  attach with: tmux attach -t settle"
