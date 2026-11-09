#!/usr/bin/env bash
# Does world practice make a normal next-word model better? (oazy/worldmix.py; João, Nov 9)
# Bee6 at 500M tokens of Sado, one run after the other on one GPU, then one table:
#   1. bee6-race      plain Bee6                                    (kept if it already finished: the same settings)
#   2. bee6-world10   the same, with 10% of the tokens world episodes (30% for the first 20%, then 5%)
#   3. bee6-world5    half strength, 5% (15% then 2.5%)              (only with: bash runpod/bee6_world.sh half)
#   4. the world's secret exams asked as plain text, for every run   (python3 -m oazy.worldmix exam)
#   5. python3 -m bee6.world_race                                    (held-out Sado loss, HellaSwag, LogT, world exams)
#
#   tmux new-window -t oazy -n world6 "OAZY_DATA=/root/data bash runpod/bee6_world.sh; bash"
#   tmux new-window -t oazy -n world6 "OAZY_DATA=/root/data bash runpod/bee6_world.sh half; bash"   # + the 5% run
#
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
COMMON=(--preset big --micro 32 --data-sub sado --data "$DATA" --runs "$RUNS" --tokens 0.5e9)   # = runpod/bee6_race.sh's
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
run() {     # run <name> <trainer options...>
    local name="$1"
    shift
    if finished "$name"; then echo "$name already finished: kept"; return; fi
    echo "== $name =="
    python3 -m oazy.train --model bee6 "${COMMON[@]}" --name "$name" "$@" 2>&1 | tee -a "$RUNS/$name.log"
}
run bee6-race
run bee6-world10 --world 0.10
ARMS=(bee6-race bee6-world10)
if [ "${1:-}" = half ]; then
    run bee6-world5 --world 0.05
    ARMS+=(bee6-world5)
fi
python3 -m oazy.worldmix exam --runs "$RUNS" --run "${ARMS[@]}"
python3 -m bee6.world_race --runs "$RUNS" --arms "${ARMS[@]}"
