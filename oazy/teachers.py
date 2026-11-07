"""The teachers write too: the finished models we learn from, on the same prompts as our runs' samples.

    nice -n 10 python3 -m oazy.teachers                       # SmolLM2-360M (Coral's and the Bees' teacher), on the CPU
    nice -n 10 python3 -m oazy.teachers --models smollm2-360m,smollm2-1.7b

The dashboard shows them under each prompt, next to what our models write, so you can see how far a run still is from
the model teaching it. Same prompts and the same sampling as the trainer (temperature 0.8, top 40, seed 1234,
oazy/train.py write_samples). It runs on the CPU by default, with a few threads, so a training run on the GPU is not
disturbed; it writes <runs>/teacher_samples.jsonl (each run replaces that model's lines).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import default_dir
from . import term

TEACHERS = {   # name -> (Hugging Face repo, how much text it read, for the label)
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "2,000,000M"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "4,000,000M"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "11,000,000M"),
    "gpt2": ("openai-community/gpt2", "not published"),
}
NICE = {"smollm2-135m": "SmolLM2-135M", "smollm2-360m": "SmolLM2-360M", "smollm2-1.7b": "SmolLM2-1.7B", "gpt2": "OpenAI GPT-2"}


def write(model_name: str, prompts, n_tok: int, device: str, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, read = TEACHERS[model_name]
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(repo)
    net = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    net.to(device).eval()
    out = []
    for p in prompts:
        torch.manual_seed(1234)
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        t0 = time.time()
        with torch.no_grad():
            g = net.generate(ids, max_new_tokens=n_tok, do_sample=True, temperature=0.8, top_k=40,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        out.append({"model": model_name, "name": NICE[model_name], "read": read, "prompt": p, "text": text,
                    "made": time.time()})
        term.check(f"{NICE[model_name]}: {p}", True, f"{time.time() - t0:.0f} s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="smollm2-360m", help=f"comma-separated: {', '.join(TEACHERS)}")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--device", default="cpu", help="cpu (default: leaves the GPU to training) or cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=120, help="words written per prompt (the trainer's sample_len)")
    a = ap.parse_args()
    from .train import SAMPLE_PROMPTS
    runs = Path(a.runs or default_dir("runs"))
    path = runs / "teacher_samples.jsonl"
    names = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in names:
        if m not in TEACHERS:
            raise SystemExit(f"unknown teacher {m}: pick from {', '.join(TEACHERS)}")
    term.header("Teachers write", f"{', '.join(NICE[m] for m in names)} · on {a.device}")
    keep = []
    if path.exists():                                   # the other teachers' lines stay
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("model") not in names:
                keep.append(r)
    for m in names:
        keep += write(m, SAMPLE_PROMPTS, a.tokens, a.device, a.threads)
    runs.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
    tmp.replace(path)
    term.out(term.c(f"\n  saved to {path}; the dashboard shows them under each prompt (refresh the page)", "dim"))


if __name__ == "__main__":
    main()
