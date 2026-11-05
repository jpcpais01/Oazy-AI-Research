# Tide on RunPod

Tide (121.6M, next-word model with a belt of 8 drafts on top of Engram's memories) trains on the
same FineWeb-Edu text, in the same order and with the same schedule as Engram and the GPT-2
baseline, and shows up on the same dashboard. See `tide/README.md` for what it is.

## 1. The pod

Same as Engram (`docs/ENGRAM_RUNPOD.md`): a B200 (or H200), RunPod PyTorch image with CUDA 12.8+,
100 GB volume at `/workspace`, HTTP ports **8000** (dashboard) and **8001** (chat).

## 2. Setup (once)

If the pod already ran Engram, the data is there: skip to the `git` lines and the self-test.

```bash
cd /workspace
git clone -b claude/project-thread-u0u9ea https://github.com/jpcpais01/simulations.git simulations-tide
cd simulations-tide
bash runpod/setup_engram.sh          # FineWeb-Edu + HellaSwag into /workspace/engram_data (skip if done)
python3 -m tide.tests --device cuda  # Tide's own self-tests
```

## 3. Measure speed and pick the budget (1 min)

```bash
python3 -m engram.train --bench --model tide --hours 2
```

It prints Tide's and the GPT's speed and the `--tokens` value for 2 hours. If Engram is training on
the same GPU right now, the numbers will look slower than they really are.

## 4. Train

```bash
bash runpod/train_tide.sh --tokens 2e9      # use the number from step 3 (same as Engram's run, to compare fairly)
```

Tide's run lands in `/workspace/engram_runs/tide`, next to `engram` and `gpt`, so the dashboard
on port 8000 shows all three. If no GPT baseline exists in that folder yet (and Engram's session
is not running), the script trains it too. With two GPUs and Engram on GPU 0: `TIDE_GPU=1 bash runpod/train_tide.sh ...`.
Re-running the same command resumes. `tmux attach -t tide` shows the logs.

The dashboard gets a new card for Tide: **how far ahead can it see?** (the loss of its draft of the
word 1, 2, ... 8 ahead). Flat lines there mean the drafts are not planning anything.

## 5. Talk to it

```bash
python3 -m tide.chat --ckpt /workspace/engram_runs/tide/ckpt.pt    # port 8001 (stop engram.chat first)
```

You see the belt live (what it expects 1 ... 8 words ahead), and for any word, how its thought
arrived: the draft of it made 8 words before, 7 words before, ..., 1 word before.

## What counts as winning

Same finish line as Engram: validation loss below OpenAI GPT-2's on the same text, and HellaSwag
above 29.6%. The fairest test of the belt itself is Tide vs Engram at the same token budget.
