# Running the lab on RunPod

## The pod

| setting | value |
|---|---|
| GPU | 1 × B200 or B300 (2 GPUs let a model and the GPT-2 baseline train side by side at full speed) |
| template | RunPod PyTorch, CUDA 12.8 or newer |
| volume | 100 GB at `/workspace` |
| HTTP ports | 8000 (the app), 8001 (chat) |

## Once per pod

```bash
cd /workspace
git clone https://github.com/jpcpais01/Oazy-AI-Research.git oazy
cd oazy
bash runpod/setup.sh          # ~20-30 min: 4B tokens of FineWeb-Edu, HellaSwag, self-tests
```

Chat on port 8001: the RunPod template's nginx already owns 8001 and forwards it to 8000, so point it
at 8002 once (`sed -i '83s#localhost:8000#localhost:8002#' /etc/nginx/nginx.conf; nginx -s reload`) and
run every chat with `--port 8002`; it then opens on the pod's 8001 link.

## Every day

```bash
cd /workspace/oazy && git pull
bash runpod/train.sh <model> [options]    # start or resume a model: coral, frog, bee, bee2, bee3, memtrix, tide, ember, fern, fern2, fern3, gpt
python3 -m oazy.train --bench --model <model> --hours 2   # how many tokens fit in 2 hours
```

Add `--gpt` to train the GPT-2 baseline next to a model. Re-running `train.sh` after a stop or a pod
restart resumes from the checkpoint.

Open the pod's **8000** link: the app shows the models training now, with their own graphs, and a
library of every model trained so far. `tmux attach -t oazy` shows the logs.

Talk to a model (works during training, from its latest checkpoint):

```bash
python3 -m bee.chat --port 8002
python3 -m bee2.chat --port 8002
python3 -m memtrix.chat --port 8002
python3 -m memtrix.chat --run ember --port 8002
python3 -m tide.chat --port 8002
python3 -m fern.chat --port 8002                 # also --run fern2, --run fern3
```

## The main models

### Bee0.2B (204M, the main model)

```bash
cd /workspace/oazy && git pull
bash runpod/setup.sh bee                                # once: self-tests + DCLM (2B tokens) + conversations (~1 h)
python3 -m oazy.train --bench --model bee --preset big  # speed, mentor included, and the hours for 5B tokens
bash runpod/train.sh bee --preset big                   # 5B tokens (~10 h on a B200); downloads GPT-2 large once
python3 -m bee.chat --port 8002                         # chat (Chat or Write), from its latest checkpoint
```

On a 32 GB GPU (RTX 5090) add `--micro 8`. If DCLM or the conversations are missing it trains on
what is there and says so. Ablations are in `bee/README.md`.

### Bee_v2_0.2B (203M, SmolLM2-360M as mentor)

Needs Bee0.2B's text first (`bash runpod/setup.sh bee`). Safe to prepare while Bee0.2B trains (CPU only):

```bash
cd /workspace/oazy && git pull
nice -n 10 python3 -m oazy.retok                         # the same text, re-cut with SmolLM2's tokenizer -> <data>/smol (~15-40 min)
python3 -m bee2.tests --device cuda                       # self-tests with the real SmolLM2 tokenizer
python3 -m oazy.train --bench --model bee2 --preset big   # speed, mentor included (best when the GPU is free)
bash runpod/train.sh bee2 --preset big                    # the same text as Bee0.2B's 5B GPT-2 tokens; downloads SmolLM2-360M once
python3 -m bee2.chat --port 8002                          # chat
```

(`bash runpod/setup.sh bee2` does the preparation in one go, but it also re-runs the general setup.)

**Tagus, our dataset** (seven kinds of text, sized for the run; see `bee2/README.md`):

```bash
python3 -m oazy.tagus --plan --run-tokens 2.5e9          # what a 2.5B run reads, and what is there
nice -n 10 python3 -m oazy.tagus --run-tokens 2.5e9      # download + cut Cosmopedia, Wikipedia, FineMath, code
tmux kill-window -t oazy:bee2 2>/dev/null; bash runpod/train.sh bee2 --preset big --tokens 2.5e9 --name bee2-tagus
```

### Bee3 (203M, Bee_v2_0.2B's body on Tagus with the new recipe)

Needs Tagus (`python3 -m oazy.tagus --run-tokens 2.5e9`, see above) and the extra tests once. Recipe and
reasons in `bee3/README.md`.

```bash
cd /workspace/oazy && git pull
python3 -m oazy.evals --prepare                          # once: ARC-Easy, PIQA, LAMBADA (tiny downloads)
python3 -m bee3.tests --device cuda                      # self-tests: the kernels, the document rule, the loss in slices
python3 -m oazy.profile --model bee3 --preset big        # where a step's time goes, tokens/s, MFU (paste it in the thread)
python3 -m oazy.profile --model bee3 --preset big --mentor none    # the student alone
bash runpod/train.sh bee3 --preset big --tokens 2.5e9    # the run; resumes if stopped
python3 -m oazy.sft --run bee3                           # afterwards: the chat fine-tune -> runs/bee3-chat
python3 -m bee3.chat --run bee3-chat --port 8002         # chat (the pre-trained one: --run bee3)
```

To turn the learning rate down on resume: `bash runpod/train.sh bee3 --preset big --tokens 2.5e9 --lr 1.8e-3 --muon-lr 0.03`.
`--loss plain` is the old way of computing the loss, for comparison; `--mentor-floor 0 --mentor-off 0` is Bee2's mentor schedule.

### Frog (406M, Bee3 at twice the size, 4,000M tokens)

Start it only after Bee3 has finished: growing Tagus uses many CPU threads, and starved CPU threads slowed Bee3
before. Recipe and reasons in `frog/README.md`.

```bash
cd /workspace/oazy && git pull
TV=$(python3 -c "import torch; print(torch.__version__)")
python3 -m pip install -q torchao --constraint <(echo "torch==${TV}")   # once: FP8 training (keeps your torch)
python3 -m pip install -q -U flash-linear-attention --constraint <(echo "torch==${TV}")   # 0.6 or newer: the memory operator
nice -n 10 python3 -m oazy.tagus --run-tokens 4e9        # once: Tagus grown to what a 4,000M-token run reads
python3 -m frog.tests --device cuda                      # self-tests; prints which speed paths are in use (operator, flex, FP8)
python3 -m oazy.profile --model frog --preset big --micro 32   # tokens/s, memory, and a whole run's speed (paste it in the thread)
bash runpod/train.sh frog --preset big --tokens 4e9 --micro 32 # the run (~7.5-9 h on a B200, to be confirmed by the profile)
python3 -m oazy.sft --run frog                           # afterwards: the chat fine-tune -> runs/frog-chat
python3 -m frog.chat --run frog-chat --port 8002         # chat
```

If the loss turns noisy, resume with `--lr 1.2e-3 --muon-lr 0.02`; `--mentor-share 0` is Bee3's way of mixing the mentor in.

### Coral (1,037M mixture of experts, about 160M at work per word, 20,000M tokens)

A next-word model built for about 1M tokens/s on one B200 (~5.5-8 h for the run). Recipe and reasons in
`coral/README.md`. Its Tagus is a separate folder (`tagus20`, about 42 GB), so it can be prepared while Frog trains;
keep it at low priority so Frog's GPU-feeding thread keeps its CPU.

```bash
cd /workspace/oazy && git pull
df -h /workspace                                         # about 45 GB free needed for tagus20 (plus a few GB of downloads at a time)
nice -n 19 python3 -m oazy.tagus --run-tokens 20e9 --into tagus20 --workers 8   # once, a few hours of CPU (safe while Frog trains)
# after Frog has finished (the GPU is free):
python3 -m coral.tests --device cuda                      # self-tests: the experts, balancing, compiled step, flex, FP8
python3 -m oazy.profile --model coral --preset big --micro 32   # tokens/s and where the time goes (paste it in the thread)
bash runpod/train.sh coral --preset big --tokens 20e9 --micro 32 # the run; resumes if stopped
python3 -m oazy.sft --run coral                           # afterwards: the chat fine-tune -> runs/coral-chat
python3 -m coral.chat --run coral-chat --port 8002        # chat
```

Watch the log line's `experts uneven` (busiest expert / average, 1.0 = perfect; it should settle near 1.1-1.3)
and `dropped` (share of word-expert pairs that found their expert full; should settle near 1%). If `uneven` stays
above 2 after the first few hundred steps, resume with `--set balance_rate=0.003`. If the loss turns noisy, resume
with `--lr 1.2e-3 --muon-lr 0.02`. `--set pattern=MMMA` is the Memtrix hybrid (profile it the same way to compare),
`--mentor none` drops the mentor, `--data-sub smol` reads the usual Tagus (a quick test before tagus20 is ready).
`--fp8 0` trains in bf16; `--set grow=28:0` starts at full depth (only on a fresh run).

## Code Black (the flagship research line: two clocks vs a strong transformer, on the clock)

Races, not one long run: every run trains for the same 30 B200 minutes, Code Black against Black-T (a modern
transformer at 125M, 250M and 500M). The GPU must be free of anything else (Bee3, Frog, Coral), or the minutes mean
nothing; the ladder checks. Plan, pass and kill lines in `codeblack/README.md`.

```bash
cd /workspace/oazy && git pull
TV=$(python3 -c "import torch; print(torch.__version__)")
python3 -m pip install -q torchao --constraint <(echo "torch==${TV}")   # once (Frog already did it): FP8 training
python3 -m codeblack.tests --device cuda                  # self-tests: the fair next-token rule, compiled step = grids, FP8
tmux new-window -t oazy -n race "bash codeblack/ladder.sh"   # R0-R3: 2 profiles + 6 races of 30 min, ~3.5 h, resumable
python3 -m codeblack.race --png /workspace/oazy_runs/race.png  # the report (also printed after every race)
```

The first answer comes after the first two races (~65 min): the report's "R2 pass line" says whether Code Black
reached Black-T 125M's 30-minute loss in 20 minutes or less. One race by hand:
`bash runpod/train.sh black --preset race --minutes 30 --micro 32 --name black-k4 --set phrase_rows=0`.
Out of memory: `MICRO=16 bash codeblack/ladder.sh`. The races show on the dashboard like any run.

## Experiments

### Memtrix, Tide and Ember (~120M experiments, next-word)

```bash
cd /workspace/oazy && git pull
bash runpod/train.sh memtrix --tokens 2e9
bash runpod/train.sh tide --tokens 2e9
python3 -m ember.tests --device cuda       # Ember downloads its mentor, GPT-2 medium, on first start
bash runpod/train.sh ember --tokens 2e9
```

### Fern (198M, writes passages coarse to fine; not next-word)

```bash
cd /workspace/oazy && git pull
python3 -m fern.tests --device cuda                    # ~1 min: no peeking, writing == scoring
python3 -m oazy.train --bench --model fern --hours 2   # Fern's speed next to the GPT-2
bash runpod/train.sh fern --tokens 2e9                 # train; the app adds "Fern's rounds"
```

Fern has 198M parameters (18 reader blocks of Memtrix memory + 6 unfolder blocks), so it costs
roughly 2-3x Memtrix per token: check `--bench` before picking `--tokens`. If it runs out of GPU
memory, add `--micro 16`. Its loss is an exact per-word loss, directly comparable with GPT-2's.

### Fern2 (119M, Fern raised the Ember way; not next-word)

```bash
cd /workspace/oazy && git pull
python3 -m fern2.tests --device cuda                    # ~1 min
python3 -m oazy.train --bench --model fern2 --hours 2   # speed (fully grown, without the mentor's cost)
bash runpod/train.sh fern2 --tokens 2e9                 # downloads GPT-2 medium on first start
python3 -m fern.chat --run fern2 --port 8002            # watch it unfold, from its latest checkpoint
```

It trains 16 sequences at a time by default (same 262k tokens per step); if it runs out of memory,
add `--micro 8`. Ablations: `--mentor none`, `--set grow=9:0 unfold_grow=3:0`, `--opt adamw`,
`--set reader_weight=0`, each with its own `--name`.

### Fern3 (117M, remembers old passages by their gist; not next-word)

```bash
cd /workspace/oazy && git pull
python3 -m fern3.tests --device cuda                    # ~1 min, includes a small recall exercise
python3 -m oazy.train --bench --model fern3 --hours 2   # speed (fully grown)
bash runpod/train.sh fern3 --tokens 2e9                 # 4,096-word windows, 262k tokens per step
python3 -m fern.chat --run fern3 --port 8002
```

It trains 4 windows of 4,096 words at a time; if it runs out of memory, add `--micro 2`.
Ablations: `--set recall=1`, `--set hindsight_weight=0`, `--set recall_warm=3`, `--mentor none`.

## Where things live

The code lives on `main` (it was on `oazy-ai-research` until Nov 5 2026). A pod cloned before then moves over once with
`git fetch origin main && git checkout main` (the files are the same, so running training is not disturbed).

`/workspace/oazy_data` (text) and `/workspace/oazy_runs` (one folder per model). Pods that already
have `/workspace/engram_data` and `/workspace/engram_runs` keep using those, nothing has to move.
