#!/usr/bin/env bash
# Start (or resume) the whole Settle experiment in one tmux session:
#   window "dash"   - the live training dashboard on port 8000
#   window "settle" - training Settle
#   window "gpt"    - training the matched GPT baseline (same size, same data)
#
#   bash runpod/train_settle.sh            # extra arguments go to both trainers, e.g. --tokens 1.5e8
#
# Re-running it after a stop or a pod restart resumes both runs from their checkpoints.
# Attach with:  tmux attach -t settle      (detach again with Ctrl-b then d)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
DATA="${SETTLE_DATA:-/workspace/settle_data}"
RUNS="${SETTLE_RUNS:-/workspace/settle_runs}"
EXTRA="$*"
mkdir -p "$RUNS"

if tmux has-session -t settle 2>/dev/null; then
    echo "a 'settle' tmux session is already running; attach with: tmux attach -t settle"
    exit 0
fi

tmux new-session -d -s settle -n dash -c "$REPO"
tmux send-keys -t settle:dash "python3 -m settle.dashboard --runs $RUNS --port 8000" C-m
tmux new-window -t settle -n settle -c "$REPO"
tmux send-keys -t settle:settle "python3 -m settle.train --model settle --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/settle.log" C-m
tmux new-window -t settle -n gpt -c "$REPO"
tmux send-keys -t settle:gpt "python3 -m settle.train --model gpt --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/gpt.log" C-m
tmux select-window -t settle:settle
echo "started. dashboard: port 8000.  attach with: tmux attach -t settle"
