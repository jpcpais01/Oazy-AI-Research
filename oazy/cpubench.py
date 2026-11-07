"""Head-to-head writing speed on the CPU: Coral against SmolLM2-360M, in plain PyTorch, batch 1.

    python3 -m oazy.cpubench                          # both models, 4 threads, float32, 128-word prompt, 128 new words
    python3 -m oazy.cpubench --threads 1 --dtype bf16
    python3 -m oazy.cpubench --models coral --coral-run runs/coral   # a trained checkpoint instead of random weights

Speed does not depend on what the weights have learned: a model with random weights and the same shapes does the
same arithmetic and reads the same bytes per word, so Coral is timed with random weights unless --coral-run is given.
SmolLM2-360M is built the same way, from its exact config (Hugging Face's Llama code, the code people normally run
it with), or loaded with its real weights with --smol-path.

Both models get the same machine, the same thread count, the same prompt (the same token ids: both use SmolLM2's
tokenizer), the same number of new words and the same loop: read the prompt in one pass (keeping the attention keys),
then write one word at a time, greedy, each new word read with the keys of all the words before it. Each model runs
in its own process, so the memory numbers are its own.

Printed per model: prompt reading speed (words/s), writing speed (words/s, the median of --repeats runs), the weights'
size, the memory in use once loaded, and the process's peak (the peak includes building the model in float32 first).

Measured Nov 7 2026 on a 4-core Xeon (Sapphire Rapids, 2.3 GHz), PyTorch 2.14 CPU, prompt 128, 128 new words:
                      writing tok/s (4 thr / 1 thr)    reading tok/s (4 thr)   memory in use
  Coral   float32         25.9 / 10.6                     357                  4.4 GB
  SmolLM2 float32         21.0 /  7.7                     499                  1.8 GB
  Coral   bf16            23.3 / 10.8                     472                  2.4 GB
  SmolLM2 bf16            20.8 /  8.9                    1291                  1.1 GB
For reference, Coral in the phone engine (phone/engine, 8-bit, plain C build on x86): 45 / 13 tok/s writing, 1.2 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time

# SmolLM2-360M's shapes, from its config.json on Hugging Face (HuggingFaceTB/SmolLM2-360M)
SMOL = dict(vocab_size=49152, hidden_size=960, intermediate_size=2560, num_hidden_layers=32, num_attention_heads=15,
            num_key_value_heads=5, max_position_embeddings=8192, rope_theta=100000.0, rms_norm_eps=1e-5,
            tie_word_embeddings=True, hidden_act="silu")


def prompt_ids(n):
    """The same n token ids for both models (they share SmolLM2's 49,152-piece tokenizer; which ids makes no
    difference to the speed)."""
    import random
    r = random.Random(0)
    return [r.randrange(100, 49000) for _ in range(n)]


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024      # Linux: kilobytes


# ---------------------------------------------------------------------------------------------- one model, one process
def load_coral(run, dtype):
    import torch
    from coral.model import Coral, CoralConfig
    if run:
        ck = torch.load(os.path.join(run, "ckpt.pt"), map_location="cpu", weights_only=False)
        m = Coral(CoralConfig.from_dict(ck["config"]))
        m.load_state_dict(ck["model"])
    else:
        torch.manual_seed(0)
        m = Coral(CoralConfig(grow="20:0"))           # the full grown body: 20 layers, 32 experts, top-2
    m = m.to(dtype).eval()
    total = sum(p.numel() for p in m.parameters())
    read = lambda idx, st: m.read(idx, st)          # (logits, states)
    return m, read, total


def load_smol(dtype, path):
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    if path:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    else:
        torch.manual_seed(0)
        m = LlamaForCausalLM(LlamaConfig(**SMOL)).to(dtype)
    m = m.eval()
    total = sum(p.numel() for p in m.parameters())

    def read(idx, st):
        o = m(input_ids=idx, past_key_values=st, use_cache=True)
        return o.logits, o.past_key_values
    return m, read, total


def bench_one(name, a):
    import torch
    torch.set_num_threads(a.threads)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    ids = prompt_ids(a.prompt)
    if name == "smollm2":
        m, read, total = load_smol(dtype, a.smol_path)
    else:
        m, read, total = load_coral(a.coral_run, dtype)
    wbytes = sum(p.numel() * p.element_size() for p in m.parameters())

    def once(n_new):
        with torch.inference_mode():
            t0 = time.perf_counter()
            logits, st = read(torch.tensor([ids]), None)
            t1 = time.perf_counter()
            nxt = int(logits[0, -1].argmax())
            for _ in range(n_new):
                logits, st = read(torch.tensor([[nxt]]), st)
                nxt = int(logits[0, -1].argmax())
            t2 = time.perf_counter()
        return len(ids) / (t1 - t0), n_new / (t2 - t1)

    once(16)                                       # warm-up: first-call allocations, kernels picked
    runs = [once(a.new) for _ in range(a.repeats)]
    return {"model": name, "threads": a.threads, "dtype": a.dtype, "prompt": len(ids), "new": a.new,
            "params_m": total / 1e6, "weights_mb": wbytes / 2**20,
            "read_tps": statistics.median(r[0] for r in runs),
            "write_tps": statistics.median(r[1] for r in runs),
            "write_runs": [round(r[1], 2) for r in runs],
            "peak_mb": peak_mb(), "rss_mb": rss_mb(), "torch": torch.__version__}


# ---------------------------------------------------------------------------------------------- the comparison
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="coral,smollm2", help="comma-separated: coral, smollm2")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--prompt", type=int, default=128, help="prompt length in tokens")
    p.add_argument("--new", type=int, default=128, help="new tokens to write")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--coral-run", default="", help="a run folder with ckpt.pt (default: random weights)")
    p.add_argument("--smol-path", default="", help="SmolLM2-360M's real weights (a Hugging Face id or folder; default: "
                   "random weights with its exact shapes)")
    p.add_argument("--json", default="", help="also append the results to this file, one JSON line each")
    p.add_argument("--one", default="", help=argparse.SUPPRESS)                # internal: run one model and print JSON
    a = p.parse_args()
    if a.one:
        print("RESULT " + json.dumps(bench_one(a.one, a)))
        return

    rows = []
    for name in a.models.split(","):
        args = [sys.executable, "-m", "oazy.cpubench", "--one", name] + [x for x in sys.argv[1:]]
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
        r = subprocess.run(args, capture_output=True, text=True, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"{name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
            continue
        row = json.loads(line[0][7:])
        rows.append(row)
        print(f"{name:11s} {row['params_m']:7.1f}M params  weights {row['weights_mb']:6.0f} MB  in use {row['rss_mb']:6.0f} MB (peak {row['peak_mb']:6.0f})  "
              f"reading {row['read_tps']:7.1f} tok/s  writing {row['write_tps']:6.2f} tok/s  {row['write_runs']}", flush=True)
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
