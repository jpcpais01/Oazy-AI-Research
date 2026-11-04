# Training Settle on RunPod (H200 or B300)

One GPU pod, one setup command, one start command. Everything resumes after
an interruption, and two web pages show what is happening: the training
dashboard (port 8000) and the chat app (port 8001).

## 1. Create the pod

1. **RunPod → Pods → Deploy.**
2. **GPU:** **1× H200** (141 GB) or **1× B300**. One GPU is enough; the code
   does not use more.
3. **Template:** the official **RunPod PyTorch** template, a recent tag with
   CUDA 12.8 or newer, for example
   `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`. (The B300
   needs a PyTorch built for Blackwell. The setup script checks this and
   upgrades PyTorch if needed.)
4. **Edit Template** before deploying:
   * Container Disk: **30 GB**
   * Volume Disk: **60 GB**, mounted at **`/workspace`** (the default)
   * **Expose HTTP Ports:** `8888,8000,8001` (8888 is JupyterLab, 8000 the
     dashboard, 8001 the chat app)
5. **On-Demand**, then **Deploy**. When it is running: **Connect → Start Web
   Terminal** (or SSH, or JupyterLab → Terminal).

## 2. Set up (about 10 minutes, once)

```bash
cd /workspace
git clone https://github.com/jpcpais01/simulations.git
cd simulations
git checkout claude/project-thread-yvtbmn      # until the PR is merged; afterwards just use main
bash runpod/setup_settle.sh
```

This installs the few Python packages, checks the GPU, downloads TinyStories
and tokenizes it with the GPT-2 tokenizer into `/workspace/settle_data`
(roughly half a billion tokens), and runs the self-tests on the GPU. It ends with
`all Settle self-tests passed` and `== setup complete`.

## 3. Check the speed (1 minute)

```bash
python -m settle.train --bench
```

It times a few training steps of both models and prints a line per model
like `... k tok/s ... -> 150M tokens in NN min`. The default run is 150M
tokens. If Settle's projected time is more than you want to spend (the plan
was about 15–30 minutes), pick a token count that fits, for example
`--tokens 1e8` in the next step. Both models always get the same number of
tokens.

## 4. Train both models with the live dashboard

```bash
bash runpod/train_settle.sh                    # or: bash runpod/train_settle.sh --tokens 1e8
```

To train **only Settle**, without the GPT baseline, use
`bash runpod/train_settle.sh --solo` instead (it also takes `--tokens`, e.g.
`bash runpod/train_settle.sh --solo --tokens 1e8`). The dashboard then shows
Settle alone.

The normal command opens a tmux session called `settle` with three windows: the dashboard,
Settle training, and the GPT baseline training (both models share the GPU;
the GPT is much faster and finishes first). To watch the terminal output:

```bash
tmux attach -t settle          # Ctrl-b then n / p switches window; Ctrl-b then d detaches
```

**Open the dashboard:** in RunPod, **Connect → HTTP Services → Port 8000**.
That opens `https://<pod-id>-8000.proxy.runpod.net`. It refreshes itself
every 5 seconds and shows:

* both runs' progress, speed and time left,
* loss of Settle vs the GPT (by words seen or by training minutes),
* how many thinking steps Settle takes per word, and how many words never settle,
* **the sheet, live:** every 4M tokens the trainer reads a test passage word
  by word and photographs the sheet (scrub through training with the slider,
  or press *watch it settle* to replay the sheet settling),
* which words are hard, as training goes on,
* a check that generation matches training,
* sample stories from both models, with Settle's words tinted by thinking time.

**If anything stops** (you press Ctrl-c, the pod restarts, the terminal
closes): run `bash runpod/train_settle.sh` again with the same arguments.
Both runs continue from their last checkpoint (saved every 5 minutes, on
Ctrl-c and at the end). Use `--fresh` only if you want to throw a run away
and start over.

The files are in `/workspace/settle_runs/settle/` and
`/workspace/settle_runs/gpt/`. The model is `ckpt.pt`, about 270 MB.

## 5. Talk to it and watch it think

When Settle has finished (or even during training, using its latest
checkpoint), open a new terminal window (`Ctrl-b` then `c` inside tmux) and run:

```bash
cd /workspace/simulations
python -m settle.chat --ckpt /workspace/settle_runs/settle/ckpt.pt
```

Open **Connect → HTTP Services → Port 8001**. Type the start of a story and
press **Write**. Every word gets a new column; you see its cells change
colour step by step until they freeze, the lens re-reading the last few words
(green bracket), arcs for the long-range links, the gist ribbon, the
step count per word (hard words take visibly longer), and Settle's top
guesses for the next word. Hover any column or word to inspect it. The sheet
is kept between messages, so you can keep adding text and the thinking
continues. **New sheet** clears it. The *animation* slider slows it down or
makes it instant.

## 6. Stop paying

Download `/workspace/settle_runs/settle/ckpt.pt` if you want to keep the
model (JupyterLab: right-click → Download, or `runpodctl send <file>`). Then
**Stop** the pod (the volume is kept, at a small storage cost) or
**Terminate** it (everything is deleted).

## Running on another machine

Everything also runs on a CPU, just slowly. The `tiny` preset is the one used
to test the pipeline:

```bash
python -m settle.data --out ./settle_data
python -m settle.train --model settle --preset tiny --data ./settle_data --runs ./settle_runs
python -m settle.dashboard --runs ./settle_runs
```
