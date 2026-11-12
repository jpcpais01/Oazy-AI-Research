#!/usr/bin/env bash
# Think: is reasoning depth bought with weights, or with reading time? Three models of one size (Keep with its stores
# written from the top, Keep as it was, the plain twin) are trained on Field1 scenes told 1 to 4 times in a row and
# examined by the depth of the question and by the number of tellings. One GPU, no data folder needed (the scenes are
# made as they are read). Runs in the tmux window "think" of the session "oazy".
#
#   bash runpod/think.sh                                  # the real run (base: ~50M each, 300M tokens each)
#   bash runpod/think.sh --only feed --name think         # one model (feed | keep | full): they can train side by side
#   bash runpod/think.sh --families switches --name sw    # one family; extra options go to python3 -m keep.think
#   bash runpod/think.sh --only report                    # examine the saved models again and rewrite the report
# Results: <runs>/think/think.md and think.json.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
NAME=think
WIN=think
PREV=""
for a in "$@"; do
    [ "$PREV" = "--name" ] && NAME="$a"
    [ "$PREV" = "--only" ] && WIN="think-$a"
    PREV="$a"
done
[ "$NAME" != "think" ] && WIN="$WIN-$NAME"
tmux has-session -t oazy 2>/dev/null || tmux new-session -d -s oazy -n app -c "$REPO"
win() { tmux list-windows -t oazy -F '#W' | grep -qx "$1"; }
if win "$WIN"; then
    case "$(tmux display-message -p -t "oazy:$WIN" '#{pane_current_command}' 2>/dev/null)" in
        bash|zsh|sh|fish) ;;
        *) echo "window '$WIN' is busy: not starting twice (tmux attach -t oazy)"; exit 0 ;;
    esac
else
    tmux new-window -t oazy: -n "$WIN" -c "$REPO"
fi
tmux send-keys -t "oazy:$WIN" "cd $REPO && CUDA_VISIBLE_DEVICES=${OAZY_GPU:-0} python3 -m keep.think --runs $RUNS $* 2>&1 | tee -a $RUNS/$WIN.log" C-m
echo "started in tmux window '$WIN'. Watch: tmux attach -t oazy. Report: $RUNS/$NAME/think.md"
