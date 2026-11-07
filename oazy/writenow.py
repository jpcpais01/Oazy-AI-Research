"""Write a run's samples now, from its latest checkpoint, without touching the training.

    nice -n 10 python3 -m oazy.writenow --run coral           # the prompts its last check did not write (on the CPU)
    nice -n 10 python3 -m oazy.writenow --run coral --all     # every prompt again

The trainer writes the prompts of oazy/train.py SAMPLE_PROMPTS at each check, but a running trainer keeps the list it
started with. When a prompt is added mid-run, this writes it straight away, from the checkpoint the trainer saves every
few minutes, so the dashboard's "What they write" box for it fills without a restart. Same sampling as the trainer
(temperature 0.8, top 40, repetition guard 1.1, seed 1234, 80 words by default), labelled with the checkpoint's token count, and added to the
run's samples.jsonl next to the trainer's own lines. On the CPU with a few threads by default, so the GPU is not
disturbed; the checkpoint is read straight from the file (memory-mapped), so its optimizer state never fills the RAM.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .data import default_dir, run_dir
from . import tok
from .train import SAMPLE_PROMPTS, Run, load_model
from .sample import REP


def last_check(path):
    """The prompts the run wrote at its newest check."""
    recs = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    own = [r for r in recs if not r.get("late")]
    top = max((r.get("step", 0) for r in own), default=-1)
    return ({r["prompt"] for r in own if r.get("step", 0) == top}
            | {r["prompt"] for r in recs if r.get("late") and r.get("step", 0) >= top})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder's name, e.g. coral")
    ap.add_argument("--runs", default=None, help="where the runs live (default: the trainer's)")
    ap.add_argument("--all", action="store_true", help="write every prompt, not only the missing ones")
    ap.add_argument("--tokens", type=int, default=80, help="how many words (tokens) each sample runs to")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (keep it low while a run trains)")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    rd = run_dir(a.runs or default_dir("runs"), a.run)
    ck = rd / "ckpt.pt"
    if not ck.exists():
        raise SystemExit(f"no checkpoint at {ck}")
    have = last_check(rd / "samples.jsonl")
    todo = SAMPLE_PROMPTS if a.all else [p for p in SAMPLE_PROMPTS if p not in have]
    if not todo:
        print(f"{a.run} already wrote all {len(SAMPLE_PROMPTS)} prompts at its last check; --all writes them again")
        return
    t0 = time.time()
    model, st = load_model(ck, torch.device(a.device), mmap=True)
    enc = tok.get(tok.name_of(model))
    print(f"{a.run}: checkpoint after {st.get('tokens', 0) / 1e6:,.0f}M tokens, loaded in {time.time() - t0:.0f}s")
    run = Run(rd)
    with torch.no_grad():
        for p in todo:
            torch.manual_seed(1234)
            ids, _ = model.generate(enc.encode_ordinary(p), max_new_tokens=a.tokens, temperature=0.8, top_k=40,
                                    eos=enc.eot_token, rep=REP)
            text = enc.decode(ids)
            run.log("samples.jsonl", {"step": st.get("step", 0), "tokens": st.get("tokens", 0), "prompt": p,
                                      "text": text, "late": True})
            print(f"\n{p}{text}")
    print(f"\nwrote {len(todo)} sample(s) to {rd / 'samples.jsonl'}; the dashboard shows them on its next refresh")


if __name__ == "__main__":
    main()
