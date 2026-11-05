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
git clone -b oazy-ai-research https://github.com/jpcpais01/simulations.git oazy
cd oazy
bash runpod/setup.sh          # ~20-30 min: 4B tokens of FineWeb-Edu, HellaSwag, self-tests
```

Chat on port 8001: the RunPod template's nginx already owns 8001 and forwards it to 8000, so point it
at 8002 once (`sed -i '83s#localhost:8000#localhost:8002#' /etc/nginx/nginx.conf; nginx -s reload`) and
run every chat with `--port 8002`; it then opens on the pod's 8001 link.

## Every day

```bash
cd /workspace/oazy && git pull
python3 -m oazy.train --bench --model tide --hours 2   # how many tokens fit in 2 hours
bash runpod/train.sh tide --tokens 2e9                 # train (add --gpt to train the GPT-2 baseline too)
```

Ember (the fast-learning model, `ember/README.md`) trains the same way and downloads its mentor,
GPT-2 medium, from Hugging Face on first start:

```bash
python3 -m ember.tests --device cuda
bash runpod/train.sh ember --tokens 2e9
```

Open the pod's **8000** link: the app shows the models training now, with their own graphs, and a
library of every model trained so far. `tmux attach -t oazy` shows the logs.

Talk to a model (works during training, from its latest checkpoint):

```bash
python3 -m memtrix.chat --port 8002
python3 -m tide.chat --port 8002
python3 -m fern.chat --port 8002
python3 -m memtrix.chat --run ember --port 8002
python3 -m fern.chat --run fern2 --port 8002
python3 -m bee.chat --port 8002
```

## Bee0.2B (204M, the main model)

```bash
cd /workspace/oazy && git pull
bash runpod/setup.sh bee                                # once: self-tests + DCLM (2B tokens) + conversations (~1 h)
python3 -m oazy.train --bench --model bee --preset big  # speed, mentor included, and the hours for 5B tokens
bash runpod/train.sh bee --preset big                   # 5B tokens (~10 h on a B200); downloads GPT-2 large once
python3 -m bee.chat --port 8002                         # chat (Chat or Write), from its latest checkpoint
```

On a 32 GB GPU (RTX 5090) add `--micro 8`. If DCLM or the conversations are missing it trains on
what is there and says so. Ablations are in `bee/README.md`.

## Bee0.2B v2 (203M, SmolLM2-360M as mentor)

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

## Fern2 (119M, Fern raised the Ember way)

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

## Fern3 (117M, remembers old passages by their gist)

```bash
cd /workspace/oazy && git pull
python3 -m fern3.tests --device cuda                    # ~1 min, includes a small recall exercise
python3 -m oazy.train --bench --model fern3 --hours 2   # speed (fully grown)
bash runpod/train.sh fern3 --tokens 2e9                 # 4,096-word windows, 262k tokens per step
python3 -m fern.chat --run fern3 --port 8002
```

It trains 4 windows of 4,096 words at a time; if it runs out of memory, add `--micro 2`.
Ablations: `--set recall=1`, `--set hindsight_weight=0`, `--set recall_warm=3`, `--mentor none`.

## Fern (200M, writes passages coarse to fine)

```bash
cd /workspace/oazy && git pull
python3 -m fern.tests --device cuda                    # ~1 min: no peeking, writing == scoring
python3 -m oazy.train --bench --model fern --hours 2   # Fern's speed next to the GPT-2
bash runpod/train.sh fern --tokens 2e9                 # train; the app adds "Fern's rounds"
```

Fern has 198M parameters (18 reader blocks of Memtrix memory + 6 unfolder blocks), so it costs
roughly 2-3x Memtrix per token: check `--bench` before picking `--tokens`. If it runs out of GPU
memory, add `--micro 16`. Its loss is an exact per-word loss, directly comparable with GPT-2's.

## Where things live

`/workspace/oazy_data` (text) and `/workspace/oazy_runs` (one folder per model). Pods that already
have `/workspace/engram_data` and `/workspace/engram_runs` keep using those, nothing has to move.
