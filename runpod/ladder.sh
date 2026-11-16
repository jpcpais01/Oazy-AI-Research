#!/usr/bin/env bash
# Ladder: how much bigger must a normal pretrained model be to reason as deep as the fed model? (keep/rival.py)
# SmolLM2-135M, 360M and 1.7B are fine-tuned one after the other on the same scenes as the fed model's run (by default
# h4 of round r18: the family tales, trained to depth 3) and sit the same exam; then ONE table. Needs nobody. Runs in
# the tmux window "ladder" of the session "oazy". A size that is already done is not trained again.
#
#   bash runpod/ladder.sh                   # the three rivals, then the ladder against the run h4
#   bash runpod/ladder.sh h4 --families tales
#   cat <runs>/ladder.md                    # the table (written when the three are done)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
OURS="${1:-h4}"
shift || true
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
WIN=ladder
tmux has-session -t oazy 2>/dev/null || tmux new-session -d -s oazy -n app -c "$REPO"
win() { tmux list-windows -t oazy -F '#W' | grep -qx "$1"; }
if win "$WIN"; then
    case "$(tmux display-message -p -t "oazy:$WIN" '#{pane_current_command}' 2>/dev/null)" in
        bash|zsh|sh|fish) ;;
        *) echo "the ladder is already running (tmux attach -t oazy). Log: tail $RUNS/ladder.log"; exit 0 ;;
    esac
else
    tmux new-window -t oazy: -n "$WIN" -c "$REPO"
fi
tmux send-keys -t "oazy:$WIN" "cd $REPO && python3 -m pip install -q transformers tokenizers && for s in 135m 360m 1.7b; do CUDA_VISIBLE_DEVICES=${OAZY_GPU:-0} python3 -m keep.rival --size \$s --runs $RUNS $* 2>&1 | tee -a $RUNS/ladder.log; done; python3 -m keep.rival --table --ours $OURS --runs $RUNS 2>&1 | tee -a $RUNS/ladder.log" C-m
echo "the ladder started in tmux window '$WIN'. It needs nobody. Progress: tail $RUNS/ladder.log   The table at the end: cat $RUNS/ladder.md"
