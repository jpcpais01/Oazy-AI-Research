#!/usr/bin/env bash
# Think-135 against Bee-135 (gnosis2/thinkab.py): the same text, tokens and seed, no mentor.
#   plain   judgeab-plain-135   Bee-135 (kept if it has finished already: the run on the volume)
#   think   thinkab-think-135   Bee-135 whole + a thought per sentence + a diffusion mind that makes the next thought
#
#   ARMS=think SUB=opal bash runpod/thinkab.sh    # Think-135 against the Bee-135 run already on the volume
#   TOKENS=1e9 ...                                # half the price (default 2e9)
#   TUNE=1 ...                                    # torch's kernel tuning on (a compile of minutes)
#
# First, a minute on the GPU: python3 -m gnosis2.think_tests --device cuda
# At the end the run is scored BY MEANING (1, 4, 16, 64 thoughts) and the verdict printed by the rule fixed before the
# run (gnosis2/thinkab.py). Re-running carries on where it stopped.
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-2e9}"
ARMS="${ARMS:-plain think}"
SUB="${SUB:-opal3}"
if [ "$SUB" = "opal" ]; then TEXT=(--opal-plan flat:0); else TEXT=(--data-sub "$SUB"); fi
[ -f "$DATA/hellaswag_val.jsonl" ] || python3 -m oazy.data --hellaswag-only --out "$DATA" || echo "HellaSwag could not be fetched: the run goes on without it"
[ -f "$DATA/evals/arc_challenge.jsonl" ] || python3 -m oazy.evals --prepare --data "$DATA" || echo "ARC and the others could not be fetched: the run goes on without them"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
for arm in $ARMS; do
    name="$(python3 -m gnosis2.thinkab --name "$arm")"
    if finished "$name"; then echo "$name already finished: kept"; continue; fi
    echo "== $name =="
    sets=($(python3 -m gnosis2.thinkab --set "$arm" ${TUNE:+--tune}))
    python3 -m oazy.train --model gnosis2 --preset big --micro 32 --data "$DATA" --runs "$RUNS" \
        "${TEXT[@]}" --mentor none --tokens "$TOKENS" --name "$name" --seed 0 \
        --set "${sets[@]}" 2>&1 | tee -a "$RUNS/$name.log"
done
if finished "$(python3 -m gnosis2.thinkab --name think)"; then
    python3 -m gnosis2.thinkab --runs "$RUNS" --data "$DATA" --score
else
    python3 -m gnosis2.thinkab --runs "$RUNS" --data "$DATA"
fi
