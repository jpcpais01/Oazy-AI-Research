#!/usr/bin/env bash
# Keep: train the model that keeps what it reads and its plain twin on the same streams, examine both at growing
# gaps, write the report. One GPU. Runs in the tmux window "keep" of the session "oazy".
#
#   OAZY_DATA=/workspace/data OAZY_RUNS=/workspace/runs bash runpod/keep.sh          # the real run (base: ~50M each, 400M tokens each)
#   bash runpod/keep.sh --preset big --name keep-big     # ~120M each, 1B tokens each; extra options go to python3 -m keep.run
#   bash runpod/keep.sh --only report                    # examine the saved models again and rewrite the report
# Needs Opal's text on the data folder (<data>/opal). Results: <runs>/keep/report.md and results.json.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
mkdir -p "$RUNS"
NAME=keep
PREV=""
for a in "$@"; do [ "$PREV" = "--name" ] && NAME="$a"; PREV="$a"; done
tmux has-session -t oazy 2>/dev/null || tmux new-session -d -s oazy -n app -c "$REPO"
win() { tmux list-windows -t oazy -F '#W' | grep -qx "$1"; }
if win "$NAME"; then
    case "$(tmux display-message -p -t "oazy:$NAME" '#{pane_current_command}' 2>/dev/null)" in
        bash|zsh|sh|fish) ;;
        *) echo "window '$NAME' is busy: not starting twice (tmux attach -t oazy)"; exit 0 ;;
    esac
else
    tmux new-window -t oazy: -n "$NAME" -c "$REPO"
fi
tmux send-keys -t "oazy:$NAME" "cd $REPO && CUDA_VISIBLE_DEVICES=${OAZY_GPU:-0} python3 -m keep.run --data $DATA --runs $RUNS $* 2>&1 | tee -a $RUNS/$NAME.log" C-m
echo "started in tmux window '$NAME'. Watch: tmux attach -t oazy. Report: $RUNS/$NAME/report.md"
