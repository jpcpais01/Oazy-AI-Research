#!/usr/bin/env bash
# The memory table's test (oazy/book.py), all of it in one go:
#   1. count Opal's text into a 2 GB table of phrases and what came next (CPUs, minutes; no GPU)
#   2. time Bee-135 with and without the table on this GPU (a few minutes; also the first real GPU pass with it)
#   3. train Bee-135 + the table (NEXT-WORD, the 135M race's arm B and its text, tokens and recipe) as bee-book-race-135
#   4. print the table against Bee-135 (2.869 held-out loss, 33.4 HellaSwag at 4,000M) with the verdict
#
#   tmux new-window -t oazy -n book "bash runpod/book.sh; bash"
#   BUILD_ONLY=1 bash runpod/book.sh         # only step 1
#   GB=4 bash runpod/book.sh                 # a bigger table
#
# Rule fixed before the run: the table is kept only if the held-out loss is 0.02 or more better than Bee-135's at equal
# tokens and the pass is under 10% slower. Re-running carries on where it stopped.
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-4e9}"
GB="${GB:-2}"
NAME=bee-book-race-135
BOOK="$(python3 -m oazy.book path --data "$DATA")"
nice -n 10 python3 -m oazy.book build --data "$DATA" --gb "$GB"
python3 -m oazy.book show --data "$DATA" || true
[ -n "${BUILD_ONLY:-}" ] && exit 0
python3 -m gnosis2.speed135 --book "$BOOK" --runs "$RUNS" 2>&1 | tee "$RUNS/speed135-book.log" || echo "the speed check did not finish; the run goes on"
if python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$NAME/status.json')).get('state') == 'done' else 1)" 2>/dev/null; then
    echo "$NAME already finished: kept"
else
    echo "== $NAME =="
    python3 -m oazy.train --model gnosis2 --preset big --micro 32 --data "$DATA" --runs "$RUNS" --opal-plan flat:0 --mentor none \
        --tokens "$TOKENS" --name "$NAME" --set $(python3 -m gnosis2.race135 --set b) "book=$BOOK" 2>&1 | tee -a "$RUNS/$NAME.log"
fi
python3 -m oazy.book verdict --runs "$RUNS"
