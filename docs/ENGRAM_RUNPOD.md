# Engram on RunPod

Engram (117M) and a GPT-2 small baseline (124M) train side by side on FineWeb-Edu, with a live
dashboard, and OpenAI's released GPT-2 is measured on the same tests as the line to beat.

## 1. The pod

| setting | value |
|---|---|
| GPU | **1 × B200** (both models share it, ~2 h for 2B tokens each), or **2 × B200** (one model per GPU, about half the time) |
| template | RunPod PyTorch, CUDA 12.8 or newer (any recent `runpod/pytorch` image) |
| container disk | 30 GB |
| volume disk | **100 GB**, mounted at `/workspace` |
| HTTP ports | **8000** (dashboard), **8001** (chat) |

## 2. Setup (once, ~20-30 min, mostly downloading text)

```bash
cd /workspace
git clone -b claude/project-thread-5x6ixv https://github.com/jpcpais01/simulations.git
cd simulations
bash runpod/setup_engram.sh
```

This prepares 4B tokens of FineWeb-Edu in `/workspace/engram_data` plus HellaSwag, installs the
fast memory kernels (flash-linear-attention), and runs the self-tests. A line saying
`flash-linear-attention kernel OK` means the fast path is on. If it says "unavailable", Engram
still works with its pure-PyTorch version, only slower.

## 3. Measure speed and pick the budget (1 min)

```bash
python3 -m engram.train --bench --hours 2
```

It prints both models' speed and the `--tokens` value that fits 2 hours on this pod.

## 4. Train

```bash
bash runpod/train_engram.sh --tokens 2e9      # use the number from step 3
```

Open the pod's **port 8000** link for the dashboard. Everything resumes if you re-run the same
command after a stop. `tmux attach -t engram` shows the logs (Ctrl-b then d to leave).
The bigger run: `ENGRAM_PREP_TOKENS=8e9 bash runpod/setup_engram.sh`, then
`bash runpod/train_engram.sh --preset big`.

## 5. Talk to it

```bash
python3 -m engram.chat --ckpt /workspace/engram_runs/engram/ckpt.pt    # port 8001
```

Works during training too (it loads the latest checkpoint, saved every 10 minutes).

## What counts as winning

On the dashboard's "finish line": Engram's validation loss below OpenAI GPT-2's on the same text,
and HellaSwag above 29.6%. The trained-here GPT-2 is the fair control (same data, same budget).
