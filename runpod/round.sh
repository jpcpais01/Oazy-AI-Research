#!/usr/bin/env bash
# Round: a set of think runs trained one after the other without anyone watching, and ONE table (keep/round.py).
# The runs of a round are listed in runpod/rounds/<round>.txt. They start two at a time, a new one only while the GPU
# has memory to spare, in the tmux window "round-<round>" of the session "oazy".
#
#   bash runpod/round.sh r2                 # start (or go on with) round r2
#   cat <runs>/round-r2.md                  # the table: rewritten whenever a run starts or ends
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
R="${1:?which round? e.g. bash runpod/round.sh r2}"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
WIN="round-$R"
tmux has-session -t oazy 2>/dev/null || tmux new-session -d -s oazy -n app -c "$REPO"
win() { tmux list-windows -t oazy -F '#W' | grep -qx "$1"; }
if win "$WIN"; then
    case "$(tmux display-message -p -t "oazy:$WIN" '#{pane_current_command}' 2>/dev/null)" in
        bash|zsh|sh|fish) ;;
        *) echo "round '$R' is already running (tmux attach -t oazy). Table: cat $RUNS/round-$R.md"; exit 0 ;;
    esac
else
    tmux new-window -t oazy: -n "$WIN" -c "$REPO"
fi
tmux send-keys -t "oazy:$WIN" "cd $REPO && CUDA_VISIBLE_DEVICES=${OAZY_GPU:-0} python3 -m keep.round --runs $RUNS $* 2>&1 | tee -a $RUNS/$WIN.log" C-m
echo "round '$R' started in tmux window '$WIN'. It needs nobody. The table: cat $RUNS/round-$R.md"
