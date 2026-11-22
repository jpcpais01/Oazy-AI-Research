#!/usr/bin/env bash
# Loop-135 against Bee-135 (loopab/ab.py), both NEXT-TOKEN, equal weights (132M) and equal compute:
#   plain   loopab-plain-135   Bee-135, 24 layers once each, reads 1.433x the words
#   loop    loopab-loop-135    the same 24 layers: reader 16 once, loop 4 in rounds (2-6 in training, 4 reading), mouth 4
# Same text: Opal3 (Field1 off) + 5% wordings (oazy/wordings.py). Same seed, no mentor. Plain first, then loop.
#
#   tmux new -d -s oazy 2>/dev/null; tmux new-window -t oazy -n loop "cd /workspace/oazy && bash runpod/loopab.sh; bash"
#   TOKENS=1e9 bash runpod/loopab.sh       # the loop arm's words (default 2e9; plain then reads 1.433x that)
#   SUB=opal bash runpod/loopab.sh         # the first Opal (Field1 off) if Opal3 is not on this pod
#   ARMS=loop bash runpod/loopab.sh        # one arm alone
#   TUNE=0 bash runpod/loopab.sh           # no kernel tuning (starts in seconds, a few % slower)
#   MICRO=16 bash runpod/loopab.sh         # if the loop arm runs out of GPU memory (it holds ~1.5x Bee-135's activations)
#
# First, a minute on the GPU: python3 -m gnosis2.loop_tests --device cuda   (checks + both arms' speed)
# Rule fixed before the run (loopab/ab.py): kept only if the held-out rule test's hard parts (new wordings, deeper
# chains) are 5 points or more above plain, reasoning (ARC-E, ARC-C, CSQA, Winogrande) no more than 0.5 under, and
# HellaSwag no more than 2 under. At the end: python3 -m loopab.ab --rounds  (the rule test by rounds: thinking longer).
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-2e9}"
ARMS="${ARMS:-plain loop}"
SUB="${SUB:-opal3}"
if [ "$SUB" = "opal" ]; then TEXT=(--opal-plan flat:0); else TEXT=(--data-sub "$SUB"); fi
[ -f "$DATA/hellaswag_val.jsonl" ] || python3 -m oazy.data --hellaswag-only --out "$DATA" || echo "HellaSwag could not be fetched: the run goes on without it"
[ -f "$DATA/evals/arc_challenge.jsonl" ] || python3 -m oazy.evals --prepare --data "$DATA" || echo "ARC and the others could not be fetched: the run goes on without them"
nice -n 10 python3 -m oazy.wordings build --data "$DATA"        # the wordings text (a few minutes, once) and the rule test
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
for arm in $ARMS; do
    name="$(python3 -m loopab.ab --name "$arm")"
    if finished "$name"; then echo "$name already finished: kept"; continue; fi
    echo "== $name =="
    sets=($(python3 -m loopab.ab --set "$arm" $([ "${TUNE:-1}" = "1" ] && echo --tune)))
    python3 -m oazy.train --model gnosis2 --preset big --micro "${MICRO:-32}" --data "$DATA" --runs "$RUNS" "${TEXT[@]}" --mentor none \
        --wordings "$(python3 -m loopab.ab --wordings)" --tokens "$(python3 -m loopab.ab --tokens "$arm" "$TOKENS")" \
        --name "$name" --seed 0 --set "${sets[@]}" 2>&1 | tee -a "$RUNS/$name.log"
done
python3 -m loopab.ab --runs "$RUNS"
[ -d "$RUNS/loopab-loop-135" ] && python3 -m loopab.ab --runs "$RUNS" --rounds || true
