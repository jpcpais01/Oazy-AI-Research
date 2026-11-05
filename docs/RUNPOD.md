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

Open the pod's **8000** link: the app shows the models training now, with their own graphs, and a
library of every model trained so far. `tmux attach -t oazy` shows the logs.

Talk to a model (works during training, from its latest checkpoint):

```bash
python3 -m engram.chat --port 8002
python3 -m tide.chat --port 8002
```

## Where things live

`/workspace/oazy_data` (text) and `/workspace/oazy_runs` (one folder per model). Pods that already
have `/workspace/engram_data` and `/workspace/engram_runs` keep using those, nothing has to move.
