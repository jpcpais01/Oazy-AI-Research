#!/usr/bin/env bash
# World-first, stage 1: train the Laws model and its rival on the world, examine both, write the report.
# One GPU (or none), no dataset needed. Runs in the tmux window "world" of the session "oazy".
#
#   bash runpod/world.sh                         # the real run (base: ~15M each, 20,000 steps each)
#   bash runpod/world.sh --steps 40000           # extra options go to python3 -m worldfirst.run
#   bash runpod/world.sh --only report           # examine the saved models again and rewrite the report
# Results: <runs>/worldfirst/report.md and results.json; the dashboard (port 8000) shows them as a card.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
NAME=worldfirst
PREV=""
for a in "$@"; do [ "$PREV" = "--name" ] && NAME="$a"; PREV="$a"; done
tmux has-session -t oazy 2>/dev/null || tmux new-session -d -s oazy -n app -c "$REPO"
win() { tmux list-windows -t oazy -F '#W' | grep -qx "$1"; }
if ! (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    win app || tmux new-window -t oazy: -n app -c "$REPO"
    tmux send-keys -t oazy:app "python3 -m oazy.dashboard --runs $RUNS --port 8000" C-m
fi
W="world"
[ "$NAME" != worldfirst ] && W="$NAME"
if win "$W"; then
    case "$(tmux display-message -p -t "oazy:$W" '#{pane_current_command}' 2>/dev/null)" in
        bash|zsh|sh|fish) ;;
        *) echo "window '$W' is busy: not starting twice (tmux attach -t oazy)"; exit 0 ;;
    esac
else
    tmux new-window -t oazy: -n "$W" -c "$REPO"
fi
tmux send-keys -t "oazy:$W" "cd $REPO && CUDA_VISIBLE_DEVICES=${OAZY_GPU:-0} python3 -m worldfirst.run --runs $RUNS $* 2>&1 | tee -a $RUNS/$NAME.log" C-m
echo "started in tmux window '$W'. Watch: tmux attach -t oazy. Report: $RUNS/$NAME/report.md"
