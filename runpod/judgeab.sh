#!/usr/bin/env bash
# Judge first, write second (gnosis2/judgeab.py): two short runs of Bee-135 (lean + cuDNN attention), the same text
# (Opal3 with Field1 off), the same tokens, the same seed, no mentor, one after the other:
#   plain   judgeab-plain-135   the next-word lesson alone
#   judge   judgeab-judge-135   + the judge's lesson: sentences and sentence endings of each row change places and a
#                               yes/no head answers "does this belong here?" on every piece (loss = next word + 10 x judge)
#
#   tmux new-window -t oazy -n judge "bash runpod/judgeab.sh; bash"
#   TOKENS=1e9 bash runpod/judgeab.sh        # half the price (default 2e9 each)
#   SUB=opal bash runpod/judgeab.sh          # the first Opal (Field1 off) if Opal3 is not on this pod (default opal3)
#   ARMS=judge bash runpod/judgeab.sh        # one arm alone
#   TUNE=1 bash runpod/judgeab.sh            # with torch's kernel tuning on (a compile of minutes per run)
#
# Rule fixed before the run: the judge is kept only if HellaSwag scored by the judge is 5 points or more above the
# plain arm's HellaSwag, and ARC (Easy and Challenge, their mean) scored by the judge is no more than 1 point under it.
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-2e9}"
ARMS="${ARMS:-plain judge}"
SUB="${SUB:-opal3}"
if [ "$SUB" = "opal" ]; then TEXT=(--opal-plan flat:0); else TEXT=(--data-sub "$SUB"); fi
# the tests the rule reads, fetched once if they are not on this pod
[ -f "$DATA/hellaswag_val.jsonl" ] || python3 -m oazy.data --hellaswag-only --out "$DATA" || echo "HellaSwag could not be fetched: the run goes on without it"
[ -f "$DATA/evals/arc_challenge.jsonl" ] || python3 -m oazy.evals --prepare --data "$DATA" || echo "ARC and the others could not be fetched: the run goes on without them"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
for arm in $ARMS; do
    name="$(python3 -m gnosis2.judgeab --name "$arm")"
    if finished "$name"; then echo "$name already finished: kept"; continue; fi
    echo "== $name =="
    python3 -m oazy.train --model gnosis2 --preset big --micro 32 --data "$DATA" --runs "$RUNS" "${TEXT[@]}" --mentor none \
        --tokens "$TOKENS" --name "$name" --seed "$(python3 -m gnosis2.judgeab --seed "$arm")" \
        --set $(python3 -m gnosis2.judgeab --set "$arm" ${TUNE:+--tune}) 2>&1 | tee -a "$RUNS/$name.log"
done
python3 -m gnosis2.judgeab --runs "$RUNS"
