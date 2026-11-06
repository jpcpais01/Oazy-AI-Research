#!/usr/bin/env bash
# Train a model of the lab (and keep the app running) in one tmux session called "oazy".
#
#   bash runpod/train.sh tide                       # train Tide
#   bash runpod/train.sh fern --tokens 2e9          # train Fern (writes passages coarse to fine)
#   bash runpod/train.sh memtrix --tokens 2e9        # extra arguments go to the trainer
#   bash runpod/train.sh tide --gpt                 # also train the GPT-2 baseline next to it
#   bash runpod/train.sh ember --tokens 2e9         # Ember (grows, learns from GPT-2 medium, Muon)
#   bash runpod/train.sh fern2 --tokens 2e9         # Fern2 (Fern, smaller, raised the Ember way)
#   bash runpod/train.sh fern3 --tokens 2e9         # Fern3 (Fern + remembers old passages by their gist)
#   bash runpod/train.sh bee --tokens 5e9           # Bee0.2B, the main model (needs: bash runpod/setup.sh bee)
#   bash runpod/train.sh bee2 --preset big          # Bee_v2_0.2B, SmolLM2-360M mentor (needs: bash runpod/setup.sh bee2)
#   bash runpod/train.sh bee3 --preset big --tokens 2.5e9   # Bee3 on Tagus, our dataset (needs: python3 -m oazy.tagus)
#
# Each model gets its own tmux window; the app (dashboard, port 8000) gets one too, and OpenAI's
# GPT-2 is measured once as the reference line. Running it again for another model adds that model.
# Re-running after a stop or a pod restart resumes from the checkpoint.
#   OAZY_GPU=1 bash runpod/train.sh memtrix          # put the model on GPU 1
# Watch:  tmux attach -t oazy     (Ctrl-b then a window number to switch, Ctrl-b then d to leave)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
MODEL="${1:-}"
[ "$MODEL" = engram ] && MODEL=memtrix          # Memtrix's old name
case "$MODEL" in memtrix|tide|fern|ember|fern2|fern3|bee|bee2|bee3|gpt) shift ;; *) echo "usage: bash runpod/train.sh memtrix|tide|fern|ember|fern2|fern3|bee|bee2|bee3|gpt [--gpt] [trainer options]"; exit 1 ;; esac
WITH_GPT=0
ARGS=()
for a in "$@"; do if [ "$a" = "--gpt" ]; then WITH_GPT=1; else ARGS+=("$a"); fi; done
EXTRA="${ARGS[*]:-}"
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
if [ ! -f "$DATA/train.bin" ]; then
    echo "no training data in $DATA yet: run  bash runpod/setup.sh  first"
    exit 1
fi
NGPU=$( (nvidia-smi -L 2>/dev/null || true) | grep -c "^GPU" || true)
GPU_M="${OAZY_GPU:-0}"
if [ "${NGPU:-1}" -ge 2 ]; then GPU_G=$(( (GPU_M + 1) % NGPU )); else GPU_G=0; fi
echo "data $DATA, runs $RUNS. GPUs: ${NGPU}. $MODEL on GPU ${GPU_M}."
if [ "$MODEL" = bee ] && { [ ! -f "$DATA/dclm/train.bin" ] || [ ! -f "$DATA/chat/train.bin" ]; }; then
    echo "note: Bee's DCLM or chat text is missing, so it will train on what is there. Prepare it with: bash runpod/setup.sh bee"
fi
if [ "$MODEL" = bee2 ] && [ ! -f "$DATA/smol/train.bin" ]; then
    echo "Bee_v2_0.2B needs the text re-cut with SmolLM2's tokenizer first: bash runpod/setup.sh bee2"
    exit 1
fi
if [ "$MODEL" = bee3 ] && [ ! -f "$DATA/smol/cosmo/train.bin" ]; then
    echo "Bee3 reads Tagus, our dataset; prepare it first: python3 -m oazy.tagus --run-tokens 2.5e9"
    exit 1
fi

tmux has-session -t oazy 2>/dev/null || tmux new-session -d -s oazy -n app -c "$REPO"
win() { tmux list-windows -t oazy -F '#W' | grep -qx "$1"; }
start() {   # start <window> <gpu> <command>
    if win "$1"; then echo "window '$1' already exists: not starting it twice"; return; fi
    tmux new-window -t oazy: -n "$1" -c "$REPO"
    tmux send-keys -t "oazy:$1" "CUDA_VISIBLE_DEVICES=$2 $3" C-m
}
if ! (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    win app || tmux new-window -t oazy: -n app -c "$REPO"
    tmux send-keys -t oazy:app "python3 -m oazy.dashboard --runs $RUNS --port 8000" C-m
else
    echo "something already serves port 8000 (the app); leaving it"
fi
start "$MODEL" "$GPU_M" "python3 -m oazy.train --model $MODEL --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/$MODEL.log"
if [ "$WITH_GPT" = 1 ] && [ "$MODEL" != gpt ]; then
    start gpt "$GPU_G" "python3 -m oazy.train --model gpt --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/gpt.log"
fi
if [ ! -f "$RUNS/openai-gpt2/status.json" ]; then
    start ref "$GPU_G" "python3 -m oazy.compare --data $DATA --runs $RUNS 2>&1 | tee -a $RUNS/openai-gpt2.log"
fi
tmux select-window -t "oazy:$MODEL"
echo "started. app: port 8000.  watch with: tmux attach -t oazy"
