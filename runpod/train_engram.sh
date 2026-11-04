#!/usr/bin/env bash
# Start (or resume) the whole Engram experiment in one tmux session:
#   window "dash"   - the live training dashboard on port 8000
#   window "engram" - training Engram
#   window "gpt"    - training the GPT-2 baseline (same data, same order, same schedule)
#   window "ref"    - measures OpenAI's released GPT-2 124M on the same tests, once (a few minutes)
#
#   bash runpod/train_engram.sh                     # both models; extra arguments go to both trainers
#   bash runpod/train_engram.sh --tokens 2.5e9      #   e.g. the budget suggested by --bench
#   bash runpod/train_engram.sh --solo              # only Engram
#   bash runpod/train_engram.sh --preset big        # 5B tokens (prepare 8B tokens of data first)
#
# With 2+ GPUs, Engram gets GPU 0 and the GPT GPU 1; with one GPU they share it.
# Re-running after a stop or a pod restart resumes both runs from their checkpoints.
# Attach with:  tmux attach -t engram      (detach again with Ctrl-b then d)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
DATA="${ENGRAM_DATA:-/workspace/engram_data}"
RUNS="${ENGRAM_RUNS:-/workspace/engram_runs}"
SOLO=0
if [ "${1:-}" = "--solo" ]; then SOLO=1; shift; fi
EXTRA="$*"
mkdir -p "$RUNS"

NGPU=$( (nvidia-smi -L 2>/dev/null || true) | grep -c "^GPU" || true)
if [ "${NGPU:-1}" -ge 2 ]; then GPU_E=0; GPU_G=1; else GPU_E=0; GPU_G=0; fi
echo "GPUs found: ${NGPU}. Engram on GPU ${GPU_E}, GPT on GPU ${GPU_G}."

if tmux has-session -t engram 2>/dev/null; then
    echo "an 'engram' tmux session is already running; attach with: tmux attach -t engram"
    exit 0
fi

tmux new-session -d -s engram -n dash -c "$REPO"
tmux send-keys -t engram:dash "python3 -m engram.dashboard --runs $RUNS --port 8000" C-m
tmux new-window -t engram: -n engram -c "$REPO"
tmux send-keys -t engram:engram "CUDA_VISIBLE_DEVICES=$GPU_E python3 -m engram.train --model engram --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/engram.log" C-m
if [ "$SOLO" = 0 ]; then
    tmux new-window -t engram: -n gpt -c "$REPO"
    tmux send-keys -t engram:gpt "CUDA_VISIBLE_DEVICES=$GPU_G python3 -m engram.train --model gpt --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/gpt.log" C-m
fi
if [ ! -f "$RUNS/openai-gpt2/status.json" ]; then
    tmux new-window -t engram: -n ref -c "$REPO"
    tmux send-keys -t engram:ref "CUDA_VISIBLE_DEVICES=$GPU_G python3 -m engram.compare --data $DATA --runs $RUNS 2>&1 | tee -a $RUNS/openai-gpt2.log" C-m
fi
tmux select-window -t engram:engram
echo "started. dashboard: port 8000.  attach with: tmux attach -t engram"
