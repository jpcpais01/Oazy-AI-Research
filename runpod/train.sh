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
#   bash runpod/train.sh frog --preset big --tokens 4e9     # Frog, Bee3 at 406M (needs: python3 -m oazy.tagus --run-tokens 4e9)
#   bash runpod/train.sh coral --preset big --tokens 20e9 --micro 32   # Coral, 1B experts (needs: python3 -m oazy.tagus --run-tokens 20e9 --into tagus20)
#   bash runpod/train.sh bee4 --preset big --tokens 10e9 --micro 32    # Bee4-200M, 198M all attention, Coral's recipe (reads tagus20 too)
#   bash runpod/train.sh coralmini --preset big --tokens 10e9 --micro 32   # Coral Mini, 198M experts (16, top 2), Bee4's recipe (tagus20)
#   bash runpod/train.sh pegasusmini --preset big --tokens 5e9 --micro 32   # Pegasus 1 Mini, 100M: Thinker (meaning) + Speaker (words) (tagus20)
#   bash runpod/train.sh pegasus2 --preset big --tokens 5e9 --micro 32 --data-sub sado   # Pegasus 2 Mini (pegasus/model2.py)
#   bash runpod/train.sh plain100 --preset big --tokens 5e9 --micro 32      # Plain-100M, the normal transformer Pegasus 1 Mini races
#   OAZY_DATA=/root/data bash runpod/train.sh plain100f --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name plain100f-race
#   OAZY_DATA=/root/data bash runpod/train.sh alphastar --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name alphastar-race   # Pegasus Alpha Star (alphastar/)
#   OAZY_DATA=/root/data bash runpod/train.sh pegasusbeta --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name beta-race   # Pegasus Beta (alphastar/beta.py)
#   OAZY_DATA=/root/data bash runpod/train.sh bee5 --preset big --tokens 0.5e9 --micro 32 --name bee5-race   # Bee5-200M (bee5/), reads Sado
#   OAZY_DATA=/root/data bash runpod/train.sh bee6 --preset big --tokens 0.5e9 --micro 32 --name bee6-race   # Bee6-200M (bee6/), Bee5 made fast
#     (Bee6 vs Bee5 at equal tokens and equal hours in one line: bash runpod/bee6_race.sh)
#     (Bee6 with 10% world practice, oazy/worldmix.py: add --world 0.10; the race in one line: bash runpod/bee6_world.sh)
#   OAZY_DATA=/root/data bash runpod/train.sh bee6 --preset big --tokens 10e9 --micro 32 --data-sub opal --mentor none --name bee6-opal-10b
#   OAZY_DATA=/workspace/data bash runpod/train.sh anima --preset big --tokens 10e9 --micro 32 --data-sub opal --mentor none --name anima-opal-10b   # Anima-300M: a mind that settles by surprise
#   OAZY_DATA=/workspace/data bash runpod/train.sh gnosis --preset big --tokens 10e9 --micro 32 --data-sub opal --mentor none --name gnosis-opal-10b   # Gnosis-300M: Bee6 with 22 layers
#     (Opal, our training set: our text + Field1, the world told in natural speech, 20%; build: python3 -m oazy.opal build)
#   OAZY_DATA=/root/data bash runpod/train.sh seed --preset big --tokens 0.5e9 --micro 32 --name seed-smoke   # Seed (seed/): growing core + library, reads Sado
#   OAZY_DATA=/root/data bash runpod/train.sh smol --preset big --tokens 0.5e9 --micro 32 --name smol-race      # SmolLM2-135M from scratch, their recipe (smol/)
#   OAZY_DATA=/root/data bash runpod/train.sh smolbee --preset big --tokens 0.5e9 --micro 32 --name smolbee-race   # the same body, our recipe
#     (--set size=360m for SmolLM2-360M)
#                                                                           # Plain-100F: Plain-100M + a training-only next-sentence head (foresight/)
#   bash runpod/train.sh pegasus --preset big --tokens 10e9 --micro 32      # Pegasus 1, 400M
#   bash runpod/train.sh plain100 --preset big --tokens 0.5e9 --micro 32 --data-sub sado --name ab-select --select 1   # it picks what it learns from (oazy/select.py)
#   bash runpod/train.sh blackt --preset race --minutes 30 --micro 32 --name blackt-125m   # a Code Black race (codeblack/README.md)
#   (--name X: the run, its tmux window and its log are called X, so several races of one model can live side by side)
#
# Each model gets its own tmux window; the app (dashboard, port 8000) gets one too, and OpenAI's
# GPT-2 is measured once as the reference line. Running it again for another model adds that model.
# Re-running after a stop or a pod restart resumes from the checkpoint.
#   OAZY_GPU=1 bash runpod/train.sh memtrix          # put the model on GPU 1
# Watch:  tmux attach -t oazy     (Ctrl-b then a window number to switch, Ctrl-b then d to leave)
set -euo pipefail
export OAZY_COLOR="${OAZY_COLOR:-1}"            # coloured logs, even through tee
cd "$(dirname "$0")/.."
REPO="$(pwd)"
MODEL="${1:-}"
[ "$MODEL" = engram ] && MODEL=memtrix          # Memtrix's old name
case "$MODEL" in memtrix|tide|fern|ember|fern2|fern3|bee|bee2|bee3|frog|coral|bee4|coralmini|pegasus|pegasusmini|pegasus2|plain100|plain100f|alphastar|pegasusbeta|bee5|bee6|gnosis|anima|seed|smol|smolbee|black|blackt|gpt) shift ;; *) echo "usage: bash runpod/train.sh memtrix|tide|fern|ember|fern2|fern3|bee|bee2|bee3|frog|coral|bee4|coralmini|pegasus|pegasusmini|pegasus2|plain100|plain100f|alphastar|pegasusbeta|bee5|bee6|gnosis|anima|seed|smol|smolbee|black|blackt|gpt [--gpt] [trainer options]"; exit 1 ;; esac
WITH_GPT=0
ARGS=()
WIN="$MODEL"
PREV=""
for a in "$@"; do
    if [ "$a" = "--gpt" ]; then WITH_GPT=1; else ARGS+=("$a"); fi
    if [ "$PREV" = "--name" ]; then WIN="$a"; fi
    PREV="$a"
done
EXTRA="${ARGS[*]:-}"
DATA="$(python3 -c "from oazy.data import default_dir; print(default_dir('data'))")"
RUNS="$(python3 -c "from oazy.data import default_dir; print(default_dir('runs'))")"
mkdir -p "$RUNS"
if [ ! -f "$DATA/train.bin" ] && [[ "$EXTRA" != *--data-sub* ]] && [ "$MODEL" != bee5 ] && [ "$MODEL" != bee6 ] && [ "$MODEL" != seed ] && [ "$MODEL" != smol ] && [ "$MODEL" != smolbee ]; then   # these read Sado (checked below)
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
if [ "$MODEL" = frog ] && [ ! -f "$DATA/smol/cosmo/train.bin" ]; then
    echo "Frog reads Tagus, our dataset; prepare it first: python3 -m oazy.tagus --run-tokens 4e9"
    exit 1
fi
if { [ "$MODEL" = black ] || [ "$MODEL" = blackt ]; } && [ ! -f "$DATA/smol/cosmo/train.bin" ]; then
    echo "Code Black and Black-T race on Tagus, our dataset; prepare it first: python3 -m oazy.tagus --run-tokens 2.5e9"
    exit 1
fi
if { [ "$MODEL" = coral ] || [ "$MODEL" = bee4 ] || [ "$MODEL" = coralmini ] || [ "$MODEL" = pegasus ] || [ "$MODEL" = pegasusmini ] || [ "$MODEL" = pegasus2 ] || [ "$MODEL" = plain100 ] || [ "$MODEL" = plain100f ] || [ "$MODEL" = alphastar ] || [ "$MODEL" = pegasusbeta ]; } && [ ! -f "$DATA/tagus20/cosmo/train.bin" ] && [[ "$EXTRA" != *--data-sub* ]]; then
    echo "Coral, Bee4, Coral Mini, Pegasus and Plain-100M read the 20,000M-token Tagus (tagus20); prepare it first: nice -n 10 python3 -m oazy.tagus --run-tokens 20e9 --into tagus20"
    exit 1
fi

if { [ "$MODEL" = bee5 ] || [ "$MODEL" = bee6 ] || [ "$MODEL" = seed ] || [ "$MODEL" = smol ] || [ "$MODEL" = smolbee ]; } && [ ! -f "$DATA/sado/train.bin" ] && [[ "$EXTRA" != *--data-sub* ]]; then
    echo "$MODEL reads Sado; build it first (nice -n 10 python3 -m oazy.sado --run-tokens 20e9) or point OAZY_DATA at the copy that has it"
    exit 1
fi

tmux has-session -t oazy 2>/dev/null || tmux new-session -d -s oazy -n app -c "$REPO"
win() { tmux list-windows -t oazy -F '#W' | grep -qx "$1"; }
idle() {    # is the window's shell waiting at a prompt (its program stopped, e.g. with Ctrl-C)?
    case "$(tmux display-message -p -t "oazy:$1" '#{pane_current_command}' 2>/dev/null)" in bash|zsh|sh|fish) return 0;; esac; return 1
}
start() {   # start <window> <gpu> <command>
    if win "$1"; then
        if idle "$1"; then echo "window '$1' exists but nothing runs in it: starting there"
        else echo "window '$1' already exists and is busy: not starting it twice"; return; fi
    else
        tmux new-window -t oazy: -n "$1" -c "$REPO"
    fi
    tmux send-keys -t "oazy:$1" "cd $REPO && CUDA_VISIBLE_DEVICES=$2 $3" C-m
}
if ! (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    win app || tmux new-window -t oazy: -n app -c "$REPO"
    tmux send-keys -t oazy:app "python3 -m oazy.dashboard --runs $RUNS --port 8000" C-m
else
    echo "something already serves port 8000 (the app); leaving it"
fi
start "$WIN" "$GPU_M" "python3 -m oazy.train --model $MODEL --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/$WIN.log"
if [ "$WITH_GPT" = 1 ] && [ "$MODEL" != gpt ]; then
    start gpt "$GPU_G" "python3 -m oazy.train --model gpt --data $DATA --runs $RUNS $EXTRA 2>&1 | tee -a $RUNS/gpt.log"
fi
if [ ! -f "$RUNS/openai-gpt2/status.json" ]; then
    start ref "$GPU_G" "python3 -m oazy.compare --data $DATA --runs $RUNS 2>&1 | tee -a $RUNS/openai-gpt2.log"
fi
tmux select-window -t "oazy:$WIN"
echo "started. app: port 8000.  watch with: tmux attach -t oazy"
