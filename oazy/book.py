"""The memory table ("book"): what the training text itself says comes next, counted once and kept outside the weights.

    python3 -m oazy.book build --gb 2            # count Opal's text, keep the 2 GB of phrases seen most (CPUs, no GPU)
    python3 -m oazy.book show                    # what is in it
    python3 -m oazy.book check                   # the file is whole and the model can read it
    python3 -m oazy.book verdict                 # the 4,000M test against Bee-135, by the rule fixed before (runpod/book.sh)

What it is. For every phrase of 2, 3 and 4 pieces that the text holds often enough, the table keeps the 4 pieces that
most often came next and how often (one byte each), and how many times the phrase was seen. Nothing is learned: it is
counting. A model that has the table (gnosis2/model.py, `book=<file>`) looks up, at every word, the phrases ending
there and gets "the text usually goes on like this" added to its input, through one small learned layer that starts at
zero. It is still a NEXT-WORD model, scored the same way; the table is a file next to the weights, not weights.

Why. A small network spends much of itself remembering wording. Counts remember wording for free and exactly; the
network then only has to learn what the counts get wrong (COUNT, arXiv 2610.20456, did this with an 800 GB table and
models of 60M to 1B: 46-89% less compute for the same loss). Our phrase hint (Phi-135) was a 75 MB table the network
had to learn by itself and gave nothing; this one is exact and about 25 times bigger. Whether a table this small helps
at 135M is not known: runpod/book.sh measures it.

How it is built. Two passes over a sample of the text (default 2,000M tokens, drawn from the parts in the shares the
run reads them), one phrase length at a time:
  1. every chunk of text -> its (phrase, next piece) pairs as 64-bit keys, counted and sorted, one file per chunk
  2. the keys in 64 ranges, each merged over the chunks, and per phrase the 4 commonest next pieces
then one count threshold is chosen so that the table fits the size asked for.

One limit to know: the counts include the text the model trains on, so on a phrase seen only a few times the table
partly "knows the answer" in training and not on new text. Phrases seen under 5 times are never kept for that reason.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import term

K = 4                                   # next pieces kept per phrase
ORDERS = (2, 3, 4)                      # phrase lengths, in pieces
BUCKETS = 64                            # key ranges of pass 2 (the top 6 bits of a key)
CHUNK = 24_000_000                      # tokens per chunk of pass 1
MIN_COUNT = 5                           # a phrase seen fewer times is never kept
ENTRY_BYTES = 8 + 2 * K + K + 1         # key, next pieces, their shares, the count's size
M1, M2 = 0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F
MASK48 = (1 << 48) - 1
NAME = "book.pt"


# ---- the phrase key: the same bits in numpy (building) and in torch (the model reading) ----

def _hash_np(cols):
    """cols[j] = the piece j back (uint64 arrays of one shape) -> a 48-bit key of the phrase (uint64)."""
    with np.errstate(over="ignore"):
        h = np.zeros(cols[0].shape, np.uint64)
        for c in cols:
            h = (h ^ (c + np.uint64(1))) * np.uint64(M1)
        h = (h ^ (h >> np.uint64(32))) * np.uint64(M2)
    return h >> np.uint64(16)


def _signed(c: int) -> int:
    return c - (1 << 64) if c >= (1 << 63) else c


def hash_torch(back, order: int):
    """back (..., n) int64, [..., j] = the piece j back -> the same 48-bit key (int64). Wraps in 64 bits like numpy."""
    h = (back[..., 0] + 1) * _signed(M1)                     # 0 ^ x = x
    for j in range(1, order):
        h = (h ^ (back[..., j] + 1)) * _signed(M1)
    h = (h ^ ((h >> 32) & 0xFFFFFFFF)) * _signed(M2)         # a signed shift fills with the sign: mask it away
    return (h >> 16) & MASK48


# ---- pass 1: a chunk of text -> its counted keys ----

def _chunk(job):
    path, start, n, order, out = job
    a = np.asarray(np.memmap(path, dtype=np.uint16, mode="r")[start:start + n]).astype(np.uint64)
    L = a.size
    if L <= order + 1:
        return 0
    ctx = _hash_np([a[order - 1 - j:L - 1 - j] for j in range(order)])
    key = (ctx << np.uint64(16)) | a[order:]
    key.sort()
    first = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    cnt = np.diff(np.concatenate((first, [key.size]))).astype(np.uint32)
    np.save(out + ".k.npy", key[first])
    np.save(out + ".c.npy", cnt)
    return int(first.size)


# ---- pass 2: one range of keys, merged over every chunk -> per phrase its commonest next pieces ----

def _bucket(job):
    b, files, out, min_count = job
    lo = np.uint64(b << 58)
    ks, cs = [], []
    for f in files:
        k = np.load(f + ".k.npy", mmap_mode="r")
        i = int(np.searchsorted(k, lo, "left"))
        j = k.size if b == BUCKETS - 1 else int(np.searchsorted(k, np.uint64((b + 1) << 58), "left"))
        if j > i:
            ks.append(np.asarray(k[i:j]))
            cs.append(np.asarray(np.load(f + ".c.npy", mmap_mode="r")[i:j]))
    empty = (np.zeros(0, np.uint64), np.zeros(0, np.uint32), np.zeros((0, K), np.uint16), np.zeros((0, K), np.uint32))
    if not ks:
        np.savez(out, ctx=empty[0], total=empty[1], nxt=empty[2], cnt=empty[3])
        return 0
    key, cnt = np.concatenate(ks), np.concatenate(cs).astype(np.uint64)
    del ks, cs
    o = np.argsort(key, kind="stable")
    key, cnt = key[o], cnt[o]
    first = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    key, cnt = key[first], np.add.reduceat(cnt, first)                    # each (phrase, next) once, with its count
    ctx = key >> np.uint64(16)
    g0 = np.flatnonzero(np.concatenate(([True], ctx[1:] != ctx[:-1])))    # where each phrase starts
    total = np.add.reduceat(cnt, g0)
    size = np.diff(np.concatenate((g0, [ctx.size])))
    keep = np.repeat(total >= min_count, size)
    gid = np.repeat(np.arange(g0.size), size)[keep]
    key, cnt = key[keep], cnt[keep]
    good = total >= min_count
    newid = np.cumsum(good) - 1                                           # phrase number among the kept ones
    gid = newid[gid]
    G = int(good.sum())
    if G == 0:
        np.savez(out, ctx=empty[0], total=empty[1], nxt=empty[2], cnt=empty[3])
        return 0
    o = np.lexsort((-cnt.astype(np.int64), gid))                          # within a phrase, the commonest next first
    gid, key, cnt = gid[o], key[o], cnt[o]
    start = np.flatnonzero(np.concatenate(([True], gid[1:] != gid[:-1])))
    rank = np.arange(gid.size) - np.repeat(start, np.diff(np.concatenate((start, [gid.size]))))
    top = rank < K
    nxt = np.zeros((G, K), np.uint16)
    cn = np.zeros((G, K), np.uint32)
    nxt[gid[top], rank[top]] = (key[top] & np.uint64(0xFFFF)).astype(np.uint16)
    cn[gid[top], rank[top]] = np.minimum(cnt[top], 2 ** 32 - 1).astype(np.uint32)
    np.savez(out, ctx=ctx[g0][good], total=np.minimum(total[good], 2 ** 32 - 1).astype(np.uint32), nxt=nxt, cnt=cn)
    return G


# ---- the text to count ----

def sources(base: Path, mix: str):
    """[(train.bin, share)] of the first phase of a mix, the parts that are there."""
    from .mix import parse_mix, resolve_mix, source_path
    from .opal3 import opal_dir
    opal = opal_dir(base)
    shares = parse_mix(resolve_mix(mix))[0][1]
    out = [(source_path(opal, n), w) for n, w in shares.items() if w > 0 and source_path(opal, n).exists()]
    if not out:
        raise SystemExit(f"no text found in {opal} for the mix {mix!r}")
    s = sum(w for _, w in out)
    return [(p, w / s) for p, w in out]


def plan_chunks(srcs, tokens: float):
    """The stretches of each part to count: its share of the tokens asked for, from the start of its file, in chunks."""
    jobs = []
    for path, w in srcs:
        have = os.path.getsize(path) // 2
        want = int(min(have, tokens * w))
        for s in range(0, want, CHUNK):
            jobs.append((str(path), s, min(CHUNK + max(ORDERS), want - s)))
    return jobs


def build(base: Path, out: Path, gb: float, tokens: float, mix: str, workers: int, orders=ORDERS, tmp: Path | None = None):
    from .sado import cpu_limit
    workers = workers or max(1, min(24, cpu_limit() - 2))      # each holds about 1 GB in pass 1
    srcs = sources(base, mix)
    jobs = plan_chunks(srcs, tokens)
    counted = sum(n for _, _, n in jobs)
    budget = int(gb * 2 ** 30 / ENTRY_BYTES)
    tmp = Path(tmp or out.parent / "book_tmp")
    term.header("Memory table", f"{gb:g} GB (room for {budget / 1e6:.0f}M phrases) from {term.tokens(counted)} tokens of {mix} · {workers} workers")
    term.out(term.c("  parts: " + ", ".join(f"{p.parent.name or 'fwe'} {w:.0%}" for p, w in srcs), "dim"), flush=True)
    per = {}
    t0 = time.time()
    for order in orders:
        d = tmp / f"o{order}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        files = [str(d / f"c{i:05d}") for i in range(len(jobs))]
        with Pool(workers) as pool:
            pairs = sum(pool.imap_unordered(_chunk, [(p, s, n, order, f) for (p, s, n), f in zip(jobs, files)]))
        with Pool(max(1, min(workers, 12))) as pool:         # pass 2 holds about 3 GB each at 2,000M tokens
            n = sum(pool.imap_unordered(_bucket, [(b, files, str(d / f"b{b:02d}.npz"), MIN_COUNT) for b in range(BUCKETS)]))
        for f in files:
            for ext in (".k.npy", ".c.npy"):
                Path(f + ext).unlink(missing_ok=True)
        parts = [np.load(d / f"b{b:02d}.npz") for b in range(BUCKETS)]
        per[order] = {k: np.concatenate([p[k] for p in parts]) for k in ("ctx", "total", "nxt", "cnt")}
        shutil.rmtree(d, ignore_errors=True)
        term.out(f"  phrases of {order}: {pairs / 1e6:,.0f}M (phrase, next) pairs in the chunks, {n / 1e6:,.1f}M phrases seen "
                 f"{MIN_COUNT}+ times  ({term.duration(time.time() - t0)})", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    # one threshold on "times seen" for every length, so that the table fits
    allc = np.concatenate([per[o]["total"] for o in orders])
    if allc.size > budget:
        cut = int(np.partition(allc, allc.size - budget)[allc.size - budget])
        if int((allc >= cut).sum()) > budget:
            cut += 1
    else:
        cut = MIN_COUNT
    del allc
    import torch
    book = {"orders": list(orders), "k": K, "tokens": int(counted), "min_count": int(cut), "mix": mix, "made": time.strftime("%Y-%m-%d %H:%M")}
    rows = []
    for o in orders:
        p = per[o]
        m = p["total"] >= cut
        tot = p["total"][m].astype(np.float64)
        prob = np.rint(255.0 * p["cnt"][m] / tot[:, None]).clip(0, 255).astype(np.uint8)
        lg = np.rint(8.0 * np.log2(tot)).clip(0, 255).astype(np.uint8)
        book[f"keys_{o}"] = torch.from_numpy(p["ctx"][m].astype(np.int64))
        book[f"nxt_{o}"] = torch.from_numpy(p["nxt"][m].view(np.int16).copy())
        book[f"prob_{o}"] = torch.from_numpy(prob)
        book[f"lg_{o}"] = torch.from_numpy(lg)
        sure = float((prob[:, 0] >= 128).mean()) if prob.size else 0.0
        rows.append([f"{o} pieces", f"{int(m.sum()) / 1e6:,.1f}M", f"{int(m.sum()) * ENTRY_BYTES / 2 ** 30:.2f} GB", f"{sure:.0%}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(book, out)
    term.table(["phrase length", "phrases kept", "size", "next piece over half sure"], rows, right=(1, 2, 3))
    term.out(term.c(f"  ✓ {out}  ({os.path.getsize(out) / 2 ** 30:.2f} GB; kept phrases seen {cut}+ times; {term.duration(time.time() - t0)})", "green", "bold"), flush=True)
    return out


def load(path):
    """The table as a dict of tensors (keys sorted, as built)."""
    import torch
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"the memory table {p} is not there: python3 -m oazy.book build")
    return torch.load(p, map_location="cpu", weights_only=False)


def lookup_np(book, tokens, order: int):
    """Plain numpy lookup, the reference the model's own lookup is checked against: tokens (T,) -> (hit (T,), nxt (T,K), prob (T,K))."""
    a = np.asarray(tokens).astype(np.uint64)
    T = a.size
    keys = book[f"keys_{order}"].numpy()
    hit = np.zeros(T, bool)
    nxt = np.zeros((T, K), np.int64)
    prob = np.zeros((T, K), np.float32)
    if T >= order and keys.size:
        q = _hash_np([a[order - 1 - j:T - j] for j in range(order)]).astype(np.int64)
        i = np.minimum(np.searchsorted(keys, q), keys.size - 1)
        h = keys[i] == q
        hit[order - 1:] = h
        nxt[order - 1:][h] = book[f"nxt_{order}"].numpy()[i[h]].view(np.uint16).astype(np.int64)
        prob[order - 1:][h] = book[f"prob_{order}"].numpy()[i[h]] / 255.0
    return hit, nxt, prob


def show(path, data: Path | None, n: int = 6):
    b = load(path)
    term.header("Memory table", f"{path} · counted from {term.tokens(b['tokens'])} tokens of {b.get('mix', '?')} · phrases seen {b['min_count']}+ times")
    rows = [[f"{o} pieces", f"{b[f'keys_{o}'].numel() / 1e6:,.1f}M", f"{b[f'keys_{o}'].numel() * ENTRY_BYTES / 2 ** 30:.2f} GB"] for o in b["orders"]]
    term.table(["phrase length", "phrases", "size"], rows, right=(1, 2))
    if data is None:
        return
    try:
        from . import tok as toks
        tok = toks.get("smollm2")
        from .opal3 import opal_dir
        val = np.memmap(opal_dir(data) / "val.bin", dtype=np.uint16, mode="r")[:200_000]
    except Exception as e:
        term.out(term.c(f"  (no examples: {type(e).__name__}: {str(e)[:80]})", "dim"))
        return
    a = np.asarray(val)
    rows = []
    for o in b["orders"]:
        hit, nxt, prob = lookup_np(b, a, o)
        right = hit[:-1] & (nxt[:-1, 0] == a[1:])
        rows.append([f"{o} pieces", f"{hit.mean():.0%}", f"{right.sum() / max(1, hit[:-1].sum()):.0%}", f"{right.mean():.0%}"])
    term.out("  on held-out text (never counted):")
    term.table(["phrase length", "words with an entry", "its first guess right", "of all words"], rows, right=(1, 2, 3))
    o = max(b["orders"])
    hit, nxt, prob = lookup_np(b, a, o)
    for t in np.flatnonzero(hit)[:: max(1, int(hit.sum()) // n)][:n]:
        ctx = tok.decode([int(x) for x in a[t - o + 1:t + 1]])
        nx = ", ".join(f"{tok.decode([int(nxt[t, k])])!r} {prob[t, k]:.0%}" for k in range(K) if prob[t, k] > 0)
        term.out(f"    {ctx!r}  ->  {nx}")


# Bee-135 without a table, 4,000M tokens of Opal (Field1 off), the 135M race's arm B (18 Nov 2026, a B300)
REF = {"loss": 2.869, "HellaSwag": 33.4, "PIQA": 63.6, "CommonsenseQA": 28.4, "ARC-Easy": 50.0, "ARC-Challenge": 27.0, "tokens": 4e9}
NEED_GAIN, MAX_SLOWER = 0.02, 0.10
RUN = "bee-book-race-135"


def verdict(runs: Path, new: str = RUN, old: str = "bee-race-135"):
    """The table's test by the rule fixed before the run: keep it only if the held-out loss is 0.02 or more better than
    Bee-135's at equal tokens and the pass is under 10% slower (the speed pair is read from speed135-book.json)."""
    from .data import run_dir
    from .opal3 import finals
    a = finals(run_dir(runs, old)) or dict(REF)
    b = finals(run_dir(runs, new))
    term.header("Bee-135 with the memory table against Bee-135", "the same text and tokens, next-word, no mentor")
    if not b:
        term.out(f"  not run yet: {new}")
        return None
    names = [n for n in ("HellaSwag", "PIQA", "CommonsenseQA", "ARC-Easy", "ARC-Challenge") if n in a and n in b]
    body = [[n, f"{a[n]:.1f}", f"{b[n]:.1f}", f"{b[n] - a[n]:+.1f}"] for n in names]
    body.append(["held-out loss", f"{a['loss']:.3f}", f"{b['loss']:.3f}", f"{b['loss'] - a['loss']:+.3f}"])
    body.append(["tokens", term.tokens(a["tokens"]), term.tokens(b["tokens"]), ""])
    slower = None
    try:
        rows = {r["key"]: r["result"] for r in json.loads((Path(runs) / "speed135-book.json").read_text()) if r.get("result")}
        slower = 1 - rows["book"]["tok_s"] / rows["bee"]["tok_s"]
        body.append(["pass speed (same GPU)", term.rate(rows["bee"]["tok_s"]), term.rate(rows["book"]["tok_s"]), f"{-slower:+.0%}"])
    except (OSError, ValueError, KeyError):
        pass
    term.table(["", "Bee-135", "with the table", "change"], body, right=(1, 2, 3))
    if abs(a["tokens"] - b["tokens"]) > 0.02 * max(a["tokens"], 1):
        term.out(term.c("  no verdict yet: the two runs are not at the same number of tokens", "yellow"))
        return None
    gain = a["loss"] - b["loss"]
    ok = gain >= NEED_GAIN and (slower is None or slower < MAX_SLOWER)
    term.header("verdict by the rule fixed before the run", color="green")
    term.out(f"  {'KEEP' if ok else 'DROP'} the table: held-out loss {-gain:+.3f} (rule: {-NEED_GAIN} or better)"
             + (f", pass speed {-slower:+.0%} (rule: under {MAX_SLOWER:.0%} slower)" if slower is not None else ", speed pair not measured")
             + ". One run each: a gap under about 0.01 is within what a different seed gives.")
    return ok


def default_path(base: Path) -> Path:
    from .opal3 import opal_dir
    return opal_dir(base) / NAME


def main():
    from .data import default_dir
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "show", "check", "verdict", "path"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=None, help="the table's file (default <data>/opal/book.pt)")
    ap.add_argument("--gb", type=float, default=2.0)
    ap.add_argument("--tokens", type=float, default=2e9, help="how much text to count (default 2,000M)")
    ap.add_argument("--mix", default="opal:flat:0", help="whose shares of the parts to count (opal:flat:0 or opal3)")
    ap.add_argument("--orders", default=",".join(map(str, ORDERS)))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--again", action="store_true")
    a = ap.parse_args()
    term.quiet()
    base = Path(a.data or default_dir("data"))
    out = Path(a.out) if a.out else default_path(base)
    if a.cmd == "path":
        return print(out)
    if a.cmd == "verdict":
        return verdict(Path(a.runs or default_dir("runs")))
    if a.cmd == "build":
        if out.exists() and not a.again:
            term.out(term.c(f"  the table is there already: {out} (--again builds it again)", "green"))
            return show(out, None)
        return build(base, out, a.gb, a.tokens, a.mix, a.workers, tuple(int(s) for s in a.orders.split(",")))
    if a.cmd == "show":
        return show(out, base)
    b = load(out)
    ok = all(bool((b[f"keys_{o}"][1:] > b[f"keys_{o}"][:-1]).all()) for o in b["orders"])
    term.check("the table's keys are in order, each once", ok)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
