#!/usr/bin/env bash
# Think-135 v2 (gnosis2/think.py: thought_tie=1 thought_min=8) against Bee-135 and the first Think-135.
#   SUB=opal bash runpod/think2.sh      # trains thinkab-think2-135 (2B), scores it by meaning, prints the verdict
#   TOKENS=1e9 ...   TUNE=1 ...
# Re-running carries on where it stopped.
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-2e9}"
SUB="${SUB:-opal3}"
if [ "$SUB" = "opal" ]; then TEXT=(--opal-plan flat:0); else TEXT=(--data-sub "$SUB"); fi
name="$(python3 -m gnosis2.thinkab --name think2)"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
if finished "$name"; then echo "$name already finished: kept"; else
    sets=($(python3 -m gnosis2.thinkab --set think2 ${TUNE:+--tune}))
    python3 -m oazy.train --model gnosis2 --preset big --micro 32 --data "$DATA" --runs "$RUNS" \
        "${TEXT[@]}" --mentor none --tokens "$TOKENS" --name "$name" --seed 0 \
        --set "${sets[@]}" 2>&1 | tee -a "$RUNS/$name.log"
fi
python3 -m gnosis2.thinkab --runs "$RUNS" --data "$DATA" --arm think2 --score
