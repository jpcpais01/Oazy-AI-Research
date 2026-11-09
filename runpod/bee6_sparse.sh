#!/usr/bin/env bash
# Bee6 against Bee6 with the sparse MLP (bee6/sparse.py: each word runs only the 10% of each MLP its picker chooses),
# one after the other on one GPU, then one table:
#   0. python3 -m bee6.speedcheck            (the speed rows, incl. sparse with / without Mixture of Depths; skipped if saved)
#   1. plain Bee6, 500M tokens of Sado       (bee6-race: kept if it already finished ON THIS GPU, else a new baseline)
#   2. Bee6 + sparse MLP, the same tokens    (bee6sparse-race: quality per word)
#   3. Bee6 + sparse MLP for plain Bee6's training time (bee6sparse-hours: quality per GPU-hour)
#   4. with "nomod": Bee6 + sparse MLP without Mixture of Depths, 500M tokens (bee6sparse-nomod-race)
#   5. python3 -m bee6.sparse_race           (held-out loss, HellaSwag, LogT, speed, picker recall) -> <runs>/bee6-sparse-race.md
#
#   tmux new-window -t oazy -n sparse "OAZY_DATA=/root/data bash runpod/bee6_sparse.sh; bash"
#   tmux new-window -t oazy -n sparse "OAZY_DATA=/root/data bash runpod/bee6_sparse.sh nomod; bash"
#
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
COMMON=(--preset big --micro 32 --data-sub sado --data "$DATA" --runs "$RUNS")
SPARSE=(--set sparse_mlp=0.10)
GPU="$(python3 -c "import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')")"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
same_gpu() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/config.json')).get('gpu') == '$GPU' else 1)" 2>/dev/null; }
run() {     # run <name> <trainer options...>
    local name="$1"
    shift
    if finished "$name"; then echo "$name already finished: kept"; return; fi
    echo "== $name =="
    python3 -m oazy.train --model bee6 "${COMMON[@]}" --name "$name" "$@" 2>&1 | tee -a "$RUNS/$name.log"
}
if [ ! -f "$RUNS/speedcheck-bee6-sparse.log" ]; then
    echo "== speed check (the GPU must be free) =="
    python3 -m bee6.speedcheck --runs "$RUNS" 2>&1 | tee "$RUNS/speedcheck-bee6-sparse.log.part" \
        && mv "$RUNS/speedcheck-bee6-sparse.log.part" "$RUNS/speedcheck-bee6-sparse.log"
fi
BASE=bee6-race
if finished "$BASE" && ! same_gpu "$BASE"; then
    BASE="bee6-race-$(echo "$GPU" | tr -cs 'A-Za-z0-9' '-' | tr 'A-Z' 'a-z' | sed 's/-$//')"
    echo "bee6-race was trained on another GPU: a new plain baseline on this one, $BASE (so equal hours is fair)"
fi
run "$BASE" --tokens 0.5e9
run bee6sparse-race --tokens 0.5e9 "${SPARSE[@]}"
MIN="$(python3 -c "import json; print(round(json.load(open('$RUNS/$BASE/status.json'))['elapsed'] / 60, 2))")"
echo "plain Bee6 trained for $MIN minutes: Bee6 + sparse MLP gets the same"
run bee6sparse-hours --minutes "$MIN" "${SPARSE[@]}"
ARMS=(bee6sparse-race bee6sparse-hours)
if [ "${1:-}" = nomod ]; then
    run bee6sparse-nomod-race --tokens 0.5e9 --set sparse_mlp=0.10 mod_layers=
    ARMS+=(bee6sparse-nomod-race)
fi
python3 -m bee6.sparse_race --runs "$RUNS" --base "$BASE" --arms "${ARMS[@]}"
