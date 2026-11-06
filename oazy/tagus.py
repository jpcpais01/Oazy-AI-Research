"""Tagus: our dataset. Seven kinds of text for the SmolLM2-tokenizer models, sized for one run.
(Named after the river: many streams, one flow.)

    python -m oazy.tagus --run-tokens 2.5e9        # prepare what a 2.5B-token run reads, + 5%
    python -m oazy.tagus --run-tokens 5e9          # for a 5B run: sources that are short are prepared again, bigger
    python -m oazy.tagus --plan --run-tokens 2.5e9 # only print what it would read and what is there
    python -m oazy.tagus --run-tokens 20e9 --into tagus20   # Coral: a separate, 20,000M-token Tagus (see below)

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

--into NAME builds a whole Tagus in <data>/NAME instead of <data>/smol (Coral's 20,000M tokens: about 40 GB on
disk, a few hours of CPU; run it with nice so a training run keeps its CPU). The usual Tagus in smol/ is not
touched, so Bee3's and Frog's runs can still resume. There, FineWeb-Edu, DCLM and the conversations are streamed
straight from Hugging Face in SmolLM2's pieces (FineWeb-Edu from its 100BT sample, which has room for it); their
held-out text is copied from smol/ so every model is still scored on the same held-out text, and any document of
it met in the stream is left out of training.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import shutil
import json
import os
from pathlib import Path

import numpy as np

from .data import default_dir, write_tokens
from .mix import TAGUS_MIX, SOURCES, parse_mix
from . import term

NEW = {   # name: (repo, config, text fields, about)
    "cosmo": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", ("text",), "Cosmopedia v2"),
    "wiki": ("wikimedia/wikipedia", "20231101.en", ("title", "text"), "English Wikipedia"),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus", ("text",), "FineMath 4+"),
    "code": ("codeparrot/codeparrot-clean", None, ("content",), "Python (codeparrot-clean)"),
}
STREAMED = {   # Coral's --into: the sources smol/ got by re-cutting, streamed instead (name: about)
    "fwe": "FineWeb-Edu (sample-100BT)", "dclm": "DCLM baseline", "chat": "smol-smoltalk conversations"}
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


def streamed(name: str):
    """Batches of document texts for a STREAMED source, read as parquet files one ahead of use (and deleted after)."""
    from . import mix
    if name == "dclm":
        return mix.dclm_batches()
    if name == "chat":
        return mix.chat_batches("HuggingFaceTB/smol-smoltalk")
    repo = "HuggingFaceFW/fineweb-edu"
    files = mix._list_parquet(repo, ["sample/100BT/"])
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.permutation(len(files))]
    term.out(term.c(f"  {len(files)} parquet files in {repo} sample/100BT", "dim"), flush=True)
    return mix._parquet_batches(repo, files, "text", drop=True)


def _key(text: str) -> bytes:
    return hashlib.blake2b(text.strip().encode("utf-8", "ignore"), digest_size=12).digest()


def held_out_keys(val_bin: Path):
    """The documents of a held-out file (SmolLM2 pieces), as keys to leave out of training."""
    from . import tok
    enc = tok.get("smollm2")
    arr = np.fromfile(val_bin, dtype=np.uint16)
    ends = np.flatnonzero(arr == enc.eot_token)
    keys, a = set(), 0
    for e in ends:
        keys.add(_key(enc.decode(arr[a:e].tolist())))
        a = e + 1
    return keys


def without(batches, keys):
    """The stream with the held-out documents taken out."""
    skipped = 0
    for b in batches:
        keep = [t for t in b if _key(t) not in keys]
        skipped += len(b) - len(keep)
        if keep:
            yield keep
    if skipped:
        term.out(term.c(f"  left out {skipped} held-out documents met in the stream", "dim"), flush=True)


def prepare_streamed(base: Path, smol: Path, name: str, tokens: int, val_tokens: int, workers: int = 0):
    """A STREAMED source into the --into folder; held-out text copied from smol/ (same as every other model's)."""
    out = folder(smol, name)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from . import tok
    tok.get("smollm2")
    old = folder(base / "smol", name)
    have_val = (old / "val.bin").exists()
    keys = held_out_keys(old / "val.bin") if have_val else set()
    term.header(f"{name}", f"{STREAMED[name]} · {term.tokens(tokens)} tokens · held-out text "
                f"{'copied from ' + str(old) if have_val else 'new'}", color="blue")
    nv, nt = write_tokens(without(streamed(name), keys), out, 0 if have_val else val_tokens, tokens, workers, tokenizer="smollm2")
    meta = {"dataset": STREAMED[name], "tokenizer": "smollm2", "train_tokens": nt, "exhausted": nt < tokens}
    if have_val:
        shutil.copyfile(old / "val.bin", out / "val.bin")
        try:     # the held-out text is smol/'s, so are its token ratios (the training ratio: same source, a fair estimate)
            om = json.loads((old / "meta.json").read_text())
            meta.update({k: om[k] for k in ("val", "train") if k in om})
        except Exception:
            pass
        meta["val_tokens"] = int((out / "val.bin").stat().st_size // 2)
    else:
        meta["val_tokens"] = nv
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    _done(name, meta, out)


def _done(name, meta, out):
    term.out(f"  {term.c('✓', 'green', 'bold')} {name}: {term.tokens(meta['train_tokens'])} training tokens + "
             f"{term.tokens(meta['val_tokens'])} held out" + (term.c("  (the source ran out)", "yellow") if meta.get("exhausted") else "")
             + term.c(f"  -> {out}", "dim"), flush=True)


def prepare(smol: Path, name: str, tokens: int, val_tokens: int, workers: int = 0, repo=None):
    out = folder(smol, name)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from . import tok
    tok.get("smollm2")                                   # once, before the worker processes start
    term.header(f"{name}", f"{NEW[name][3]} · {term.tokens(tokens)} tokens + {term.tokens(val_tokens)} held out", color="blue")
    nv, nt = write_tokens(stream(name, repo=repo), out, val_tokens, tokens, workers, tokenizer="smollm2")
    meta = {"dataset": repo or NEW[name][0], "config": NEW[name][1], "tokenizer": "smollm2",
            "val_tokens": nv, "train_tokens": nt, "exhausted": nt < tokens}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    _done(name, meta, out)


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
    ap.add_argument("--into", default=None, help="build a separate Tagus in <data>/<this> (Coral: tagus20); every source, "
                    "FineWeb-Edu, DCLM and the conversations streamed in SmolLM2's pieces")
    a = ap.parse_args()
    base = Path(a.data or default_dir("data"))
    smol = base / (a.into or "smol")
    want = needs(a.run_tokens, margin=a.margin)
    term.header("Tagus", f"a {term.tokens(a.run_tokens)}-token run, + {a.margin:.0%} margin · {smol}")
    todo, body = [], []
    for k, v in sorted(want.items(), key=lambda kv: -kv[1]):
        h = have(smol, k)
        can = k in NEW or (a.into and k in STREAMED)
        state = term.c("✓ ready", "green") if h >= v else (term.c("to prepare", "yellow") if can else
                                                             term.c(f"too little: {OLD[k]} --max-tokens ...", "red"))
        body.append([k, term.tokens(v), term.tokens(h), term.bar(h / max(v, 1), 12, "green" if h >= v else "blue"), state])
        if h < v and can and (k in a.sources.split(",") or k in STREAMED):
            todo.append(k)
    term.table(["source", "needs", "there", "", ""], body, right=(1, 2))
    if a.plan:
        return
    for k in todo:
        if k in NEW:
            prepare(smol, k, want[k], int(a.val_tokens), a.workers, a.code_repo if k == "code" else None)
        else:
            prepare_streamed(base, smol, k, want[k], int(a.val_tokens), a.workers)
    short = [k for k in want if have(smol, k) < want[k] * 0.98]
    term.out(term.c("━━ ✓ Tagus is ready", "green", "bold") if not short else
             term.c(f"━━ still short: {', '.join(short)} (the run repeats a little of those; fine)", "yellow"))


if __name__ == "__main__":
    main()
