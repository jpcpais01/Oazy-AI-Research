#!/usr/bin/env bash
# The Logos1.1 race (João, Nov 9 night): after Logos1-200M lost to Bee6 on Emerald, the core gets its own job, the job of
# a law: from the things as they are before each stretch, guess how each will be spoken of in that stretch (graded on
# all text, no marks). Three arms on Emerald at 0.5B tokens each (the quick filter), one after the other, then one table:
#   1. logos11-emerald-<T>   Logos1.1: the law's guess (--set predict=1)
#   2. logos11f-emerald-<T>  Logos1.1 + focus: the core also sees the latest words (--set predict=1 focus=1)
#   3. bee6-teach-emerald-<T>  Bee6 with the same answer key (as in the 2B race)
#   then the world's secret exams as plain text and the table (python3 -m bee6.world_race) -> <runs>/logos11-race-<T>.md
#   Its [core] line shows the law's cosine against plain copying: the law has to beat copying to be learning anything.
#
#   tmux new-window -t oazy -n logos11 "OAZY_DATA=/root/data OAZY_RUNS=/workspace/runs bash runpod/logos11_race.sh; bash"
#
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
TOK="${1:-0.5e9}"
TAG="$(python3 -c "t=float('$TOK'); print(f'{t/1e9:g}b')")"
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
COMMON=(--preset big --micro 32 --data "$DATA" --runs "$RUNS" --tokens "$TOK" --data-sub emerald)
BEE="bee6-teach-emerald-$TAG"
L11="logos11-emerald-$TAG"
L11F="logos11f-emerald-$TAG"
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
[ -d "$DATA/evals" ] || python3 -m oazy.evals --prepare --data "$DATA"
echo "Emerald's world and its marks (built once; a world built before the marks only gets the marks added)"
nice -n 10 python3 -m oazy.emerald build --data "$DATA" 2>&1 | tee -a "$RUNS/emerald-build.log"
run logos "$L11" --set predict=1
run logos "$L11F" --set predict=1 focus=1
run bee6 "$BEE" --set teach=1
python3 -m oazy.worldmix exam --runs "$RUNS" --run "$BEE" "$L11" "$L11F"
python3 -m bee6.world_race --runs "$RUNS" --arms "$BEE" "$L11" "$L11F" --out "logos11-race-$TAG.md"
