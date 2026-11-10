#!/usr/bin/env bash
# The Logos race (João, Nov 9): Logos1-200M against Bee6, both on Emerald (Sado + 10% World 2), the same recipe and the
# same tokens (2B by default), one run after the other on one GPU, then one table:
#   1. Emerald's world text (World 2) and its marks are built on the CPU, if they are not there yet
#   2. logos-emerald-<T>  Logos1-200M on Emerald, first (João: our model first); the marks grade its linker on the world's episodes only
#   3. bee6-teach-emerald-<T>  Bee6 on Emerald, with the same answer key Logos1 gets: a link head on its stream after
#                              layer 4, graded by the marks on the world's episodes (as the rival in world-link)
#   4. the world's secret exams as plain text for both, then the table: held-out Sado loss, HellaSwag, ARC, PIQA,
#      LogT and the world exams (python3 -m bee6.world_race)  ->  <runs>/logos-race-<T>.md
#
#   tmux new-window -t oazy -n logos "bash runpod/logos_race.sh; bash"             # 2B tokens each
#   tmux new-window -t oazy -n logos "bash runpod/logos_race.sh 0.5e9; bash"       # the quick filter
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
COMMON=(--preset big --micro 32 --data "$DATA" --runs "$RUNS" --tokens "$TOK" --data-sub emerald)
BEE="bee6-teach-emerald-$TAG"
LOGOS="logos-emerald-$TAG"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
run() {     # run <model> <name> <more trainer options...>
    if finished "$2"; then echo "$2 already finished: kept"; return; fi
    echo "== $2 =="
    local m="$1" n="$2"
    shift 2
    python3 -m oazy.train --model "$m" "${COMMON[@]}" --name "$n" "$@" 2>&1 | tee -a "$RUNS/$n.log"
}
echo "the tests' files (HellaSwag, ARC, PIQA, LAMBADA), fetched once if they are not in $DATA"
python3 -m oazy.data --hellaswag-only --out "$DATA"
python3 -m oazy.evals --prepare --data "$DATA"     # fetches only the tests not there yet
echo "Emerald's world and its marks (built once; a world built before the marks only gets the marks added)"
nice -n 10 python3 -m oazy.emerald build --data "$DATA" 2>&1 | tee -a "$RUNS/emerald-build.log"
run logos "$LOGOS"           # João, Nov 9: our model first
run bee6 "$BEE" --set teach=1
python3 -m oazy.worldmix exam --runs "$RUNS" --run "$BEE" "$LOGOS"
python3 -m bee6.world_race --runs "$RUNS" --arms "$BEE" "$LOGOS" --out "logos-race-$TAG.md"
