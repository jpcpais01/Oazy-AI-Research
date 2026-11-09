#!/usr/bin/env bash
# Bee6 against Bee5 (bee6/README.md), one after the other on one GPU, then one table:
#   1. Bee5, 500M tokens of Sado            (bee5-race; kept if it already finished)
#   2. Bee6, the same 500M tokens           (bee6-race: equal tokens, so quality per word)
#   3. Bee6 for exactly Bee5's training time (bee6-hours: equal hours, so quality per GPU-hour)
#   4. python3 -m bee6.race                 (held-out loss, HellaSwag, LogT, speed, side by side)
#
#   tmux new-window -t oazy -n race6 "OAZY_DATA=/root/data bash runpod/bee6_race.sh; bash"
#   tmux new-window -t oazy -n race6b "OAZY_DATA=/root/data bash runpod/bee6_race.sh bloom; bash"   # Bee6 with bloom on
#
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TAG=bee6
SET=()
if [ "${1:-}" = bloom ]; then TAG=bee6bloom; SET=(--set bloom=on); fi
COMMON=(--preset big --micro 32 --data-sub sado --data "$DATA" --runs "$RUNS")
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
run() {     # run <name> <model> <trainer options...>
    local name="$1" model="$2"
    shift 2
    if finished "$name"; then echo "$name already finished: kept"; return; fi
    echo "== $name =="
    python3 -m oazy.train --model "$model" "${COMMON[@]}" --name "$name" "$@" 2>&1 | tee -a "$RUNS/$name.log"
}
run bee5-race bee5 --tokens 0.5e9
run "$TAG-race" bee6 --tokens 0.5e9 "${SET[@]}"
MIN="$(python3 -c "import json; print(round(json.load(open('$RUNS/bee5-race/status.json'))['elapsed'] / 60, 2))")"
echo "Bee5 trained for $MIN minutes: Bee6 gets the same"
run "$TAG-hours" bee6 --minutes "$MIN" "${SET[@]}"
python3 -m bee6.race --runs "$RUNS" --bee6 "$TAG"
