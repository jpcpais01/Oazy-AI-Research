#!/usr/bin/env bash
# Canvas-2 on 10B tokens (canvas/model.py with window=0): the word layers see the whole text like Bee-135, the canvas
# (the next 4 chunks' meanings, refined together) adds the thought on top. Bee-135's speed switches carried over:
# lean kernels, cuDNN attention, torch's full kernel tuning (minutes of compiling at the start of each window length).
#
#   tmux new -d -s oazy 2>/dev/null; tmux new-window -t oazy -n canvas10b "cd /workspace/oazy && bash runpod/canvas10b.sh; bash"
#   SUB=opal3 bash runpod/canvas10b.sh       # Opal3 instead of the first Opal (only if Opal3 is on the volume)
#   TOKENS=5e9 bash runpod/canvas10b.sh      # a shorter run
#   TUNE=0 bash runpod/canvas10b.sh          # no kernel tuning (starts in seconds, ~5% slower)
#
# First, a minute on the GPU: python3 -m canvas.tests --device cuda   (prints Canvas-2's speed against Bee-135)
# Re-running carries on from the last checkpoint. Compare along the way with Bee-135 on the same text
# (judgeab-plain-135, 2B tokens): the dashboard puts both on one chart.
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"
cd "$(dirname "$0")/.."
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
TOKENS="${TOKENS:-10e9}"
SUB="${SUB:-opal}"
NAME="${NAME:-canvas2-10b-135}"
if [ "$SUB" = "opal" ]; then TEXT=(--opal-plan flat:0); else TEXT=(--data-sub "$SUB"); fi
SETS=(window=0 lean=1 attn_backend=cudnn)
[ "${TUNE:-1}" = "1" ] && SETS+=(compile_mode=max-autotune-no-cudagraphs)
[ -f "$DATA/hellaswag_val.jsonl" ] || python3 -m oazy.data --hellaswag-only --out "$DATA" || echo "HellaSwag could not be fetched: the run goes on without it"
[ -f "$DATA/evals/arc_challenge.jsonl" ] || python3 -m oazy.evals --prepare --data "$DATA" || echo "ARC and the others could not be fetched: the run goes on without them"
python3 -m oazy.train --model canvas --preset big --micro 32 --data "$DATA" --runs "$RUNS" "${TEXT[@]}" --mentor none \
    --tokens "$TOKENS" --name "$NAME" --seed 0 --set "${SETS[@]}" 2>&1 | tee -a "$RUNS/$NAME.log"
