#!/usr/bin/env bash
# Canvas against Bee-135 (canvas/ab.py): the same text (Opal3 with Field1 off), the same tokens, the same seed, no
# mentor, one after the other:
#   plain    judgeab-plain-135     Bee-135, the next-word lesson alone (the judge A/B's plain arm: if it has finished
#                                  already, it is kept and not trained again)
#   canvas   canvasab-canvas-135   Canvas-135: thinks in meanings of 16-piece chunks, the next 4 refined together, words last
#   canvas3  canvas3ab-135         Canvas-3: meanings of SENTENCES, the next 6 refined together, sureness by content,
#                                  slots that find themselves wrong open again, a mouth that sees the whole text
#
#   ARMS=canvas3 SUB=opal bash runpod/canvasab.sh     # Canvas-3 against the Bee-135 run already on the volume
#
#   tmux new-window -t oazy -n canvas "bash runpod/canvasab.sh; bash"
#   TOKENS=1e9 bash runpod/canvasab.sh       # half the price (default 2e9 each)
#   SUB=opal bash runpod/canvasab.sh         # the first Opal (Field1 off) if Opal3 is not on this pod (default opal3)
#   ARMS=canvas bash runpod/canvasab.sh      # one arm alone
#   TUNE=1 bash runpod/canvasab.sh           # with torch's kernel tuning on (a compile of minutes per run)
#
# First, a minute on the GPU: python3 -m canvas.tests --device cuda
# Rule fixed before the run (canvas/ab.py): kept only if reasoning (ARC-E, ARC-C, CSQA, Winogrande) is 1.5 points or
# more above plain, HellaSwag no more than 3 points under, and the held-out loss rises 0.05 or more with the canvas off.
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-2e9}"
ARMS="${ARMS:-plain canvas}"
SUB="${SUB:-opal3}"
if [ "$SUB" = "opal" ]; then TEXT=(--opal-plan flat:0); else TEXT=(--data-sub "$SUB"); fi
[ -f "$DATA/hellaswag_val.jsonl" ] || python3 -m oazy.data --hellaswag-only --out "$DATA" || echo "HellaSwag could not be fetched: the run goes on without it"
[ -f "$DATA/evals/arc_challenge.jsonl" ] || python3 -m oazy.evals --prepare --data "$DATA" || echo "ARC and the others could not be fetched: the run goes on without them"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
for arm in $ARMS; do
    name="$(python3 -m canvas.ab --name "$arm")"
    if finished "$name"; then echo "$name already finished: kept"; continue; fi
    echo "== $name =="
    sets=($(python3 -m canvas.ab --set "$arm" ${TUNE:+--tune}))
    python3 -m oazy.train --model "$(python3 -m canvas.ab --model "$arm")" --preset big --micro 32 --data "$DATA" --runs "$RUNS" \
        "${TEXT[@]}" --mentor none --tokens "$TOKENS" --name "$name" --seed "$(python3 -m canvas.ab --seed "$arm")" \
        ${sets[@]+--set "${sets[@]}"} 2>&1 | tee -a "$RUNS/$name.log"
done
python3 -m canvas.ab --runs "$RUNS"
