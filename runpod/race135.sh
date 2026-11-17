#!/usr/bin/env bash
# The 135M race (gnosis2/race135.py): three NEXT-WORD models of SmolLM2-135M's size, the same text (Opal with Field1
# off), the same number of tokens, no mentor, one after the other on one GPU, then one table with the verdicts:
#   A  smol-race-135   SmolLM2-135M's body and their recipe, from random weights
#   B  bee-race-135    Bee-135: the Bee6 / Gnosis2 body at 24 layers of 576, our recipe
#   C  phi-race-135    Phi-135: Bee-135 with the phrase hint in place of the two value tables (the same size)
#
#   tmux new-window -t oazy -n race135 "bash runpod/race135.sh; bash"
#   TOKENS=1e9 bash runpod/race135.sh        # a shorter race (default 4e9 each)
#
# Rules fixed before the race: B stays only if it reaches A's final held-out loss on 1.3x fewer tokens;
# C stays only if it beats B by 0.02 held-out loss at equal tokens and is under 10% slower.
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-4e9}"
COMMON=(--preset big --micro 32 --data "$DATA" --runs "$RUNS" --opal-plan flat:0 --mentor none --tokens "$TOKENS")
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
run() {     # run <name> <model> <trainer options...>
    local name="$1" model="$2"
    shift 2
    if finished "$name"; then echo "$name already finished: kept"; return; fi
    echo "== $name =="
    python3 -m oazy.train --model "$model" "${COMMON[@]}" --name "$name" "$@" 2>&1 | tee -a "$RUNS/$name.log"
}
# the bodies are checked before any GPU hour is spent (SKIP_CHECKS=1 leaves the checks out: they run on the CPU and
# the toy training among them is slow on a many-core pod)
[ -n "${SKIP_CHECKS:-}" ] || python3 -m gnosis2.hint_tests
run smol-race-135 smol
run bee-race-135 gnosis2 --set $(python3 -m gnosis2.race135 --set b)
run phi-race-135 gnosis2 --set $(python3 -m gnosis2.race135 --set c)
python3 -m gnosis2.race135 --runs "$RUNS"
