#!/usr/bin/env bash
# The world-mix race (João, Nov 10): how much world text, and when, and does the SmolLM2-360M teacher help or pull Bee6
# toward ordinary-LM thinking? Five arms of plain dense Bee6 on Emerald, 2B tokens each, one after the other:
#   A bee6-mix-std-teach-2b  today's schedule (30% world until 20%, then 5%) WITH the teacher
#   B bee6-mix-std-2b        today's schedule, no teacher
#   C bee6-mix-flat10-2b     10% world the whole run, no teacher
#   D bee6-mix-flat20-2b     20% world the whole run, no teacher
#   E bee6-mix-30to15-2b     30% for the first 10%, then 15%, no teacher
# then the world exams and one table (python3 -m bee6.mix_race) -> <runs>/world-mix-race-2b.md
#
#   tmux new-window -t oazy -n mix "OAZY_DATA=/root/data OAZY_RUNS=/workspace/runs bash runpod/world_mix_race.sh; bash"
#
# B200/B300: ~1.5 h per arm. Re-running carries on (finished arms kept, a stopped one resumes). On a B300, export
# TRITON_PTXAS_PATH first (see docs). Optional first argument: tokens per arm (default 2e9).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
TOK="${1:-2e9}"
TAG="$(python3 -c "t=float('$TOK'); print(f'{t/1e9:g}b')")"
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
COMMON=(--model bee6 --preset big --micro 32 --data "$DATA" --runs "$RUNS" --tokens "$TOK")
A="bee6-mix-std-teach-$TAG"; B="bee6-mix-std-$TAG"; C="bee6-mix-flat10-$TAG"; D="bee6-mix-flat20-$TAG"; E="bee6-mix-30to15-$TAG"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
run() {     # run <name> <more trainer options...>
    if finished "$1"; then echo "$1 already finished: kept"; return; fi
    echo "== $1 =="
    local n="$1"
    shift
    python3 -m oazy.train "${COMMON[@]}" --name "$n" "$@" 2>&1 | tee -a "$RUNS/$n.log"
}
echo "the tests' files (HellaSwag, ARC, PIQA, LAMBADA), fetched once if they are not in $DATA"
python3 -m oazy.data --hellaswag-only --out "$DATA"
python3 -m oazy.evals --prepare --data "$DATA"
echo "Emerald's world (built once)"
nice -n 10 python3 -m oazy.emerald build --data "$DATA" 2>&1 | tee -a "$RUNS/emerald-build.log"
run "$A" --emerald-plan 0.30:0.2,0.05
run "$B" --emerald-plan 0.30:0.2,0.05 --mentor none
run "$C" --emerald-plan flat:0.10 --mentor none
run "$D" --emerald-plan flat:0.20 --mentor none
run "$E" --emerald-plan 0.30:0.1,0.15 --mentor none
python3 -m oazy.worldmix exam --runs "$RUNS" --run "$A" "$B" "$C" "$D" "$E"
python3 -m bee6.mix_race --runs "$RUNS" --arms "$A" "$B" "$C" "$D" "$E" --out "world-mix-race-$TAG.md"
