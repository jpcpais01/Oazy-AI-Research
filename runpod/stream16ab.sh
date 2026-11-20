#!/usr/bin/env bash
# The bf16 stream's A/B (gnosis2/stream16ab.py): does `stream16=1` cost any loss? Three short NEXT-WORD runs of Bee-135
# (lean + cuDNN attention), the same text (Opal with Field1 off), the same tokens, no mentor, one after the other:
#   A   f32-stream-135    the stream in 32-bit
#   B   bf16-stream-135   the stream in bf16 (stream16=1), the same seed
#   A2  f32b-stream-135   A again with another seed (how far two runs of the same setting end apart)
#
#   tmux new-window -t oazy -n stream16 "bash runpod/stream16ab.sh; bash"
#   TOKENS=5e8 bash runpod/stream16ab.sh     # half the price (default 1e9 each; under that the warm-up is most of the run)
#   ARMS="a b" bash runpod/stream16ab.sh     # without the second seed (the rule then falls back to 0.01)
#   TUNE=1 bash runpod/stream16ab.sh         # with torch's kernel tuning on (a compile of minutes per run)
#
# Rule fixed before the run: the bf16 stream is kept if B's final held-out loss is no more than
# max(0.01, the gap between A and A2) above A's.
# Re-running carries on where it stopped (finished runs are kept, a stopped one resumes from its checkpoint).
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-1e9}"
ARMS="${ARMS:-a b a2}"
finished() { python3 -c "import json, sys; sys.exit(0 if json.load(open('$RUNS/$1/status.json')).get('state') == 'done' else 1)" 2>/dev/null; }
for arm in $ARMS; do
    name="$(python3 -m gnosis2.stream16ab --name "$arm")"
    if finished "$name"; then echo "$name already finished: kept"; continue; fi
    echo "== $name =="
    python3 -m oazy.train --model gnosis2 --preset big --micro 32 --data "$DATA" --runs "$RUNS" --opal-plan flat:0 --mentor none \
        --tokens "$TOKENS" --name "$name" --seed "$(python3 -m gnosis2.stream16ab --seed "$arm")" \
        --set $(python3 -m gnosis2.stream16ab --set "$arm" ${TUNE:+--tune}) 2>&1 | tee -a "$RUNS/$name.log"
done
python3 -m gnosis2.stream16ab --runs "$RUNS"
