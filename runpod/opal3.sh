#!/usr/bin/env bash
# Opal3's test (oazy/opal3.py), all of it in one go:
#   1. build Opal3's six new parts on the CPUs (minutes; a training run on the GPU is not disturbed)
#   2. wait until no training run is using the GPU
#   3. train Bee-135 (the 135M race's arm B, NEXT-WORD) on Opal3: the same tokens, Field1 off, no mentor
#   4. print the table against bee-race-135 (the same model on Opal) with the verdict
#
#   tmux new-window -t oazy -n opal3 "bash runpod/opal3.sh; bash"
#   BUILD_ONLY=1 bash runpod/opal3.sh        # only step 1
#   TOKENS=1e9 bash runpod/opal3.sh          # a shorter run (the race's arm must be as long for a verdict)
#
# Rule fixed before the run: Opal3 is kept only if the average of HellaSwag, PIQA, CommonsenseQA and ARC-Easy is 1.0
# point or more above bee-race-135's at equal tokens, and the held-out loss is no more than 0.03 worse.
# Re-running carries on where it stopped (built parts are kept, a stopped run resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-4e9}"
NAME=bee-opal3-race-135
python3 -m pip install -q pyarrow huggingface_hub >/dev/null 2>&1 || true
nice -n 10 python3 -m oazy.opal3 build --data "$DATA" --run-tokens "$TOKENS"      # stops here if a part could not be built
[ -n "${BUILD_ONLY:-}" ] && exit 0
busy() { pgrep -f "oazy\.train --model" >/dev/null; }
quiet=0
while [ "$quiet" -lt 2 ]; do                   # free twice, a minute apart: the gap between two runs of a race is not "free"
    if busy; then quiet=0; echo "$(date +%H:%M) the GPU is training something else; waiting"; sleep 300; else quiet=$((quiet + 1)); sleep 60; fi
done
python3 -m oazy.opal3 check --data "$DATA" --run-tokens "$TOKENS" >/dev/null
if python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$NAME/status.json')).get('state') == 'done' else 1)" 2>/dev/null; then
    echo "$NAME already finished: kept"
else
    echo "== $NAME =="
    python3 -m oazy.train --model gnosis2 --preset big --micro 32 --data "$DATA" --runs "$RUNS" --data-sub opal3 --mentor none \
        --tokens "$TOKENS" --name "$NAME" --set $(python3 -m gnosis2.race135 --set b) 2>&1 | tee -a "$RUNS/$NAME.log"
fi
python3 -m oazy.opal3 verdict --runs "$RUNS"
