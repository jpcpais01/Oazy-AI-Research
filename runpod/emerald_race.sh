#!/usr/bin/env bash
# The Emerald race (João, Nov 9): does reading the world make a normal next-word model better, for real?
# Bee6, the same recipe and the same tokens (2B by default), one run after the other on one GPU, then one table:
#   1. Emerald's world text is built (World 2) on the CPU, in the background, while the plain run trains
#   2. bee6-sado-<T>      plain Bee6 on Sado                       (kept if it already finished)
#   3. bee6-emerald-<T>   the same on Emerald: 10% world (30% for the first 20%, then 5%)
#   4. the world's secret exams as plain text for both, then the table: held-out Sado loss, HellaSwag, ARC, PIQA,
#      LogT and the world exams (python3 -m bee6.world_race)  ->  <runs>/emerald-race-<T>.md
#
#   tmux new-window -t oazy -n emerald "bash runpod/emerald_race.sh; bash"           # 2B tokens each
#   tmux new-window -t oazy -n emerald "bash runpod/emerald_race.sh 0.5e9; bash"     # the quick filter
#
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
TOK="${1:-2e9}"
TAG="$(python3 -c "t=float('$TOK'); print(f'{t/1e9:g}b')")"
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
COMMON=(--preset big --micro 32 --data "$DATA" --runs "$RUNS" --tokens "$TOK")
PLAIN="bee6-sado-$TAG"
EMER="bee6-emerald-$TAG"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
run() {     # run <name> <trainer options...>
    local name="$1"
    shift
    if finished "$name"; then echo "$name already finished: kept"; return; fi
    echo "== $name =="
    python3 -m oazy.train --model bee6 "${COMMON[@]}" --name "$name" "$@" 2>&1 | tee -a "$RUNS/$name.log"
}
echo "building Emerald's world text in the background (CPU; log: $RUNS/emerald-build.log)"
nice -n 10 python3 -m oazy.emerald build --data "$DATA" > "$RUNS/emerald-build.log" 2>&1 &
BUILD=$!
run "$PLAIN" --data-sub sado
echo "waiting for Emerald's build to finish ..."
wait "$BUILD" || { echo "the Emerald build failed: see $RUNS/emerald-build.log"; exit 1; }
tail -n 1 "$RUNS/emerald-build.log"
run "$EMER" --data-sub emerald
python3 -m oazy.worldmix exam --runs "$RUNS" --run "$PLAIN" "$EMER"
python3 -m bee6.world_race --runs "$RUNS" --arms "$PLAIN" "$EMER" --out "emerald-race-$TAG.md"
