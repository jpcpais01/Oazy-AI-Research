"""The models pick Sado's data: two of our trained models read the start of every document, and that decides what
is left out and what is reread in the cool-down. (The "model-chosen data" idea from the 55% HellaSwag plan, done once, before training.)

    python3 -m oazy.sado_score --plan                     # how many tokens it will read, and a time guess
    python3 -m oazy.sado_score                            # score (GPU; resumes if stopped). SmolLM2-135M and -360M
    python3 -m oazy.sado_score --apply                    # then: leave out the worst, copy the best into best/ (CPU)

How it decides, for each document, from the first --prefix (256) tokens:

  small  SmolLM2-135M's average surprise (loss per token)    big  SmolLM2-360M's: the same recipe and text, 2.7x the size
         (--small/--big take our own runs too, e.g. --small bee4 --big coral: a run's ckpt.pt, or a path to one)

  noise  the big model is still very surprised: garbled text, tables of numbers, spam, other languages
         -> the top 5% by the big model's loss are left out (--drop-noise)
  easy   even the small model is barely surprised: boilerplate, lists, near-copies
         -> the bottom 5% by the small model's loss are left out (--drop-easy)
  best   the big model finds it much easier than the small one: there is something to learn here, and a model
         can learn it ("learnability", as in RHO-LOSS, Mindermann et al. 2022)
         -> the top 10% by (small - big) are copied into <sado>/best/, which the cool-down rereads (--best)

Each kind of text is ranked on its own (maths is not compared with Wikipedia). Books are not scored (the start of a
book is its title page) and neither are the conversations (kept whole for the cool-down). --apply keeps the
unscored file as train.unscored.bin (delete it to free the disk once you are happy), so it can be undone:
python3 -m oazy.sado_score --undo.

Why SmolLM2 by default: two sizes of one recipe on the same 2-4 trillion tokens, so the gap between them is about
size alone (a cleaner "can be learned" signal than two of our models with different bodies), and they need nothing
uploaded. They read FineWeb-Edu and DCLM too; a document they saw scores a little low, which makes it look "easy",
the safe side.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .data import default_dir
from .mix import SOURCES
from . import term

SCORED = ("fwe", "dclm", "cosmo", "wiki", "math", "code")
MIN_TOKENS = 16                     # shorter documents are not scored (kept)
EOT = 0                             # SmolLM2's <|endoftext|>


def src_dir(sado: Path, name: str) -> Path:
    return sado / SOURCES[name] if SOURCES[name] else sado


def doc_bounds(path: Path, chunk: int = 1 << 28):
    """Start and end (exclusive, the <|endoftext|> included) of every document in a train.bin."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    ends = [np.flatnonzero(arr[i:i + chunk] == EOT) + i for i in range(0, arr.size, chunk)]
    e = np.concatenate(ends) + 1 if ends else np.zeros(0, np.int64)
    if e.size == 0 or e[-1] != arr.size:            # a last document cut by the token budget
        e = np.append(e, arr.size)
    s = np.concatenate([[0], e[:-1]])
    return s.astype(np.int64), e.astype(np.int64)


def score_source(model, arr, starts, ends, out: Path, prefix: int, batch: int, device, tag: str):
    """Average loss per token of each document's first `prefix` tokens -> out (float16, NaN = not scored). Resumes."""
    import torch
    import torch.nn.functional as F
    n = starts.size
    done_p = out.with_suffix(".done")
    scores = np.load(out) if out.exists() else np.full(n, np.nan, np.float16)
    i0 = int(done_p.read_text()) if done_p.exists() and out.exists() else 0
    lens = np.minimum(ends - starts - 1, prefix)          # without the <|endoftext|>
    t0, last_save, toks = time.time(), time.time(), 0
    i = i0
    while i < n:
        j = min(n, i + batch)
        idx = np.arange(i, j)
        idx = idx[lens[idx] >= MIN_TOKENS]
        if idx.size:
            L = int(lens[idx].max())
            x = np.zeros((idx.size, L), np.int64)
            for r, d in enumerate(idx):
                x[r, :lens[d]] = arr[starts[d]:starts[d] + lens[d]]
            xt = torch.from_numpy(x).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=xt.is_cuda):
                logits = model._body(xt)
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), xt[:, 1:].reshape(-1),
                                   reduction="none").view(idx.size, L - 1)
            pos = torch.arange(L - 1, device=xt.device)[None]
            m = (pos < torch.from_numpy(lens[idx] - 1).to(xt.device)[:, None]).float()
            scores[idx] = ((loss * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16)
            toks += int(lens[idx].sum())
        i = j
        if time.time() - last_save > 120 or i >= n:
            np.save(out, scores)
            done_p.write_text(str(i))
            last_save = time.time()
            dt = time.time() - t0
            term.out(f"  {tag} {term.bar(i / n, 24, 'blue')} {i / n * 100:5.1f}%  {i:,}/{n:,} documents  "
                     + term.c(f"· {term.rate(toks / max(dt, 1e-6))} · {term.duration((n - i) * dt / max(i - i0, 1))} left", "dim"),
                     flush=True)
    return scores


def plan(sado: Path, prefix: int):
    body, total = [], 0
    for k in SCORED:
        p = src_dir(sado, k) / "train.bin"
        if not p.exists():
            body.append([k, "–", "–", term.c("not built", "yellow")])
            continue
        s, e = doc_bounds(p)
        t = int(np.minimum(e - s - 1, prefix)[(e - s - 1) >= MIN_TOKENS].sum())
        total += t
        body.append([k, f"{s.size:,}", term.tokens(t), ""])
    term.table(["source", "documents", f"tokens to read (first {prefix})", ""], body, right=(1, 2))
    return total


def apply(sado: Path, small: str, big: str, drop_noise: float, drop_easy: float, best: float):
    best_dir = sado / SOURCES["best"]
    best_dir.mkdir(parents=True, exist_ok=True)
    fb = open(best_dir / "train.bin.tmp", "wb")
    report, nbest = {}, 0
    for k in SCORED:
        d = src_dir(sado, k)
        full = d / "train.unscored.bin" if (d / "train.unscored.bin").exists() else d / "train.bin"
        sp, bp = d / "scores" / f"{small}.npy", d / "scores" / f"{big}.npy"
        if not full.exists() or not sp.exists():
            term.out(term.c(f"  {k}: no scores yet, left as it is", "yellow"))
            continue
        arr = np.memmap(full, dtype=np.uint16, mode="r")
        s, e = doc_bounds(full)
        ls = np.load(sp).astype(np.float32)
        lb = np.load(bp).astype(np.float32) if bp.exists() else None
        ok = ~np.isnan(ls) & (np.ones_like(ls, bool) if lb is None else ~np.isnan(lb))
        keep = np.ones(s.size, bool)
        noisy = lb if lb is not None else ls
        if ok.any():
            keep &= ~(ok & (noisy > np.quantile(noisy[ok], 1 - drop_noise)))
            keep &= ~(ok & (ls < np.quantile(ls[ok], drop_easy)))
        pick = np.zeros(s.size, bool)
        if lb is not None and best > 0 and (ok & keep).any():
            gap = ls - lb
            g = gap[ok & keep]
            pick = ok & keep & (gap >= np.quantile(g, 1 - best))
        tmp = d / "train.bin.tmp"
        with open(tmp, "wb") as f:
            for a in range(0, s.size, 1 << 16):          # in runs, so a big source never sits in memory
                sl = slice(a, a + (1 << 16))
                ss, ee, kk, pp = s[sl], e[sl], keep[sl], pick[sl]
                lo, hi = int(ss[0]), int(ee[-1])
                chunk = np.asarray(arr[lo:hi])
                rep = np.repeat(kk, ee - ss)
                f.write(chunk[rep].tobytes())
                if pp.any():
                    b = chunk[np.repeat(pp, ee - ss)]
                    fb.write(b.tobytes())
                    nbest += b.size
        if full.name == "train.bin":
            os.replace(d / "train.bin", d / "train.unscored.bin")
        os.replace(tmp, d / "train.bin")
        kept_tok = (d / "train.bin").stat().st_size // 2
        report[k] = {"documents": int(s.size), "noise_out": int((ok & (noisy > np.quantile(noisy[ok], 1 - drop_noise))).sum()) if ok.any() else 0,
                     "easy_out": int((ok & (ls < np.quantile(ls[ok], drop_easy))).sum()) if ok.any() else 0,
                     "kept": int(keep.sum()), "kept_tokens": int(kept_tok), "best": int(pick.sum())}
        term.out(f"  {term.c('✓', 'green')} {k}: kept {keep.sum():,} of {s.size:,} documents ({term.tokens(kept_tok)} tokens), "
                 f"{pick.sum():,} to reread")
    fb.close()
    os.replace(best_dir / "train.bin.tmp", best_dir / "train.bin")
    meta = {"dataset": "Sado best (oazy/sado_score.py)", "tokenizer": "smollm2", "train_tokens": nbest, "val_tokens": 0,
            "small": small, "big": big, "drop_noise": drop_noise, "drop_easy": drop_easy, "best": best, "sources": report}
    (best_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    term.out(term.c(f"━━ ✓ best/: {term.tokens(nbest)} tokens for the cool-down to reread", "green", "bold"))


def undo(sado: Path):
    import shutil
    for k in SCORED:
        d = src_dir(sado, k)
        if (d / "train.unscored.bin").exists():
            os.replace(d / "train.unscored.bin", d / "train.bin")
            term.out(f"  {k}: back to the unscored text")
    shutil.rmtree(sado / SOURCES["best"], ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--into", default="sado")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--small", default="smollm2-135m", help="the small scorer: smollm2-135m/360m/1.7b, a run name (its ckpt.pt) or a path")
    ap.add_argument("--big", default="smollm2-360m", help="the big scorer; 'none' = one model only (then no best/)")
    ap.add_argument("--prefix", type=int, default=256, help="tokens read from the start of each document")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--sources", default=",".join(SCORED))
    ap.add_argument("--device", default=None)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--drop-noise", type=float, default=0.05)
    ap.add_argument("--drop-easy", type=float, default=0.05)
    ap.add_argument("--best", type=float, default=0.10)
    a = ap.parse_args()
    sado = Path(a.data or default_dir("data")) / a.into
    runs = Path(a.runs or default_dir("runs"))
    name = lambda m: Path(m).parent.name if m.endswith(".pt") else m      # noqa: E731
    if a.undo:
        return undo(sado)
    if a.apply:
        return apply(sado, name(a.small), name(a.big), a.drop_noise, a.drop_easy, a.best)
    term.header("Sado: the models pick the data", f"{a.small} and {a.big} read the first {a.prefix} tokens of each document")
    total = plan(sado, a.prefix)
    speed = {"smollm2-135m": 400e3, "smollm2-360m": 180e3, "smollm2-1.7b": 40e3}     # rough reading speeds on an RTX 5090
    models = [a.small] if a.big == "none" else [a.small, a.big]
    secs = sum(total / speed.get(m, 300e3) for m in models)
    term.out(term.c(f"  {term.tokens(total)} tokens for each of {len(models)} models: about {term.duration(secs)} on an RTX 5090 "
                    f"(a guess; about a fifth of that on a B200, but only while it is not training)", "dim"))
    if a.plan:
        return
    import torch
    from .train import load_model
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for m in models:
        if m.startswith("smollm2"):
            from .compare import smol
            model, ck = smol(m, device), m
        else:
            ck = Path(m) if m.endswith(".pt") else runs / m / "ckpt.pt"
            model, _ = load_model(ck, device, mmap=True)
        term.out(term.c(f"  scoring with {ck}", "blue"), flush=True)
        for k in a.sources.split(","):
            d = src_dir(sado, k)
            p = d / "train.unscored.bin" if (d / "train.unscored.bin").exists() else d / "train.bin"
            if not p.exists():
                continue
            (d / "scores").mkdir(exist_ok=True)
            out = d / "scores" / f"{name(m)}.npy"
            if (d / "scores" / f"{name(m)}.done").exists() and out.exists():
                s, _ = doc_bounds(p)
                if int((d / "scores" / f"{name(m)}.done").read_text()) >= s.size:
                    continue
            s, e = doc_bounds(p)
            arr = np.memmap(p, dtype=np.uint16, mode="r")
            score_source(model, arr, s, e, out, a.prefix, a.batch, device, f"{name(m)}·{k}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    term.out(term.c("━━ ✓ scored; now: python3 -m oazy.sado_score --apply", "green", "bold"))


if __name__ == "__main__":
    main()
