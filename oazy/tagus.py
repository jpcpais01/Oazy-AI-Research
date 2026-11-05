"""Tagus: our dataset. Seven kinds of text for the SmolLM2-tokenizer models, sized for one run.
(Named after the river: many streams, one flow.)

    python -m oazy.tagus --run-tokens 2.5e9        # prepare what a 2.5B-token run reads, + 5%
    python -m oazy.tagus --run-tokens 5e9          # for a 5B run: sources that are short are prepared again, bigger
    python -m oazy.tagus --plan --run-tokens 2.5e9 # only print what it would read and what is there

All of it is written in SmolLM2's pieces under <data>/smol/ (what Bee_v2_0.2B reads):

  source  folder           what                                                      from
  fwe     smol/            FineWeb-Edu: educational web pages                        already there (oazy.retok)
  dclm    smol/dclm        DCLM: varied, natural web writing                         already there (oazy.retok)
  chat    smol/chat        smol-smoltalk conversations, "User: / Assistant:"         already there (oazy.retok)
  cosmo   smol/cosmo       Cosmopedia v2: synthetic textbooks, stories, how-tos      HuggingFaceTB/smollm-corpus
  wiki    smol/wiki        English Wikipedia (title, then the article)               wikimedia/wikipedia
  math    smol/math        FineMath 4+: the best maths explanations from the web     HuggingFaceTB/finemath
  code    smol/code        Python files, cleaned                                     codeparrot/codeparrot-clean

How much of each a run reads comes from the mix (oazy/mix.py TAGUS_MIX): the first 80% of the run
uses the main shares, the last 20% (the cool-down) the cool-down shares. This script works out the
tokens each source needs for --run-tokens, adds --margin (5%), and downloads just that, so nothing
is repeated and almost nothing is left unused. Every document ends with <|endoftext|>, which is how
document boundaries are kept. Each source also gets 2M held-out tokens (val.bin), never trained on.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

from .data import default_dir, write_tokens
from .mix import TAGUS_MIX, SOURCES, parse_mix

NEW = {   # name: (repo, config, text fields, about)
    "cosmo": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", ("text",), "Cosmopedia v2"),
    "wiki": ("wikimedia/wikipedia", "20231101.en", ("title", "text"), "English Wikipedia"),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus", ("text",), "FineMath 4+"),
    "code": ("codeparrot/codeparrot-clean", None, ("content",), "Python (codeparrot-clean)"),
}
OLD = {"fwe": "python3 -m oazy.retok --sources fwe", "dclm": "python3 -m oazy.retok --sources dclm",
       "chat": "python3 -m oazy.retok --sources chat"}


def needs(run_tokens: float, mix: str = TAGUS_MIX, margin: float = 0.05):
    """Tokens each source is read for in a run of run_tokens (the mix's phases by their length), + margin."""
    plan = parse_mix(mix)
    out = {}
    for i, (at, w) in enumerate(plan):
        end = plan[i + 1][0] if i + 1 < len(plan) else 1.0
        tot = sum(w.values())
        for k, v in w.items():
            out[k] = out.get(k, 0.0) + run_tokens * (end - at) * v / tot
    return {k: int(v * (1 + margin)) for k, v in out.items()}


def folder(smol: Path, name: str) -> Path:
    return smol / SOURCES[name] if SOURCES[name] else smol


def have(smol: Path, name: str) -> int:
    p = folder(smol, name) / "train.bin"
    return p.stat().st_size // 2 if p.exists() else 0


def text_of(fields):
    if len(fields) == 1:
        return lambda r: r.get(fields[0]) or ""
    return lambda r: "\n\n".join(str(r.get(f) or "").strip() for f in fields if r.get(f))


def stream(name: str, batch_docs: int = 1024, repo=None):
    """Batches of document texts from a Hugging Face dataset, read as a stream (only what is needed)."""
    from datasets import load_dataset
    rp, cfg, fields, _ = NEW[name]
    rp = repo or rp
    ds = load_dataset(rp, cfg, split="train", streaming=True) if cfg else load_dataset(rp, split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10_000)     # spread over the whole dataset, not its first files
    get = text_of(fields)
    it = (get(r) for r in ds)
    while True:
        batch = [t for t in itertools.islice(it, batch_docs) if t and t.strip()]
        if not batch:
            return
        yield batch


def prepare(smol: Path, name: str, tokens: int, val_tokens: int, workers: int = 0, repo=None):
    out = folder(smol, name)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from . import tok
    tok.get("smollm2")                                   # once, before the worker processes start
    print(f"== {name}: {NEW[name][3]} -> {out} ({tokens / 1e6:,.0f}M tokens + {val_tokens / 1e6:.0f}M held out)", flush=True)
    nv, nt = write_tokens(stream(name, repo=repo), out, val_tokens, tokens, workers, tokenizer="smollm2")
    meta = {"dataset": repo or NEW[name][0], "config": NEW[name][1], "tokenizer": "smollm2",
            "val_tokens": nv, "train_tokens": nt, "exhausted": nt < tokens}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"   done: {meta}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--run-tokens", type=float, default=2.5e9, help="the length of the run Tagus is prepared for")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--sources", default=",".join(NEW), help="which new sources to prepare")
    ap.add_argument("--code-repo", default=None, help="another code dataset with a 'content' column")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--plan", action="store_true", help="only show what a run reads and what is there")
    a = ap.parse_args()
    smol = Path(a.data or default_dir("data")) / "smol"
    want = needs(a.run_tokens, margin=a.margin)
    print(f"Tagus: a {a.run_tokens / 1e9:g}B-token run reads (with {a.margin:.0%} margin):")
    todo = []
    for k, v in sorted(want.items(), key=lambda kv: -kv[1]):
        h = have(smol, k)
        state = "ok" if h >= v else ("to prepare" if k in NEW else f"too little: {OLD[k]} --max-tokens ...")
        print(f"  {k:6s} {v / 1e6:8,.0f}M   there: {h / 1e6:8,.0f}M   {state}")
        if h < v and k in NEW and k in a.sources.split(","):
            todo.append(k)
    if a.plan:
        return
    for k in todo:
        prepare(smol, k, want[k], int(a.val_tokens), a.workers, a.code_repo if k == "code" else None)
    short = [k for k in want if have(smol, k) < want[k] * 0.98]
    print("Tagus is ready." if not short else f"still short: {short} (the run repeats a little of those; fine)")


if __name__ == "__main__":
    main()
