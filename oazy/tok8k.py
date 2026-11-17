"""Our own 8,192-piece tokenizer, learned from Opal, and Opal re-cut with it (for MoE-150M, bee6/moe150.py).

    python3 -m oazy.tok8k train            # learn the 8k pieces from a ~2 GB sample of Opal's text (~15-30 min, ~20 GB RAM)
    python3 -m oazy.tok8k cut              # re-cut all of Opal into <data>/opal8k/ (CPU only; ~1-2 h on 32 vCPUs)
    python3 -m oazy.tok8k check            # bytes per piece on the held-out text: SmolLM2's 49k vs ours

Why: at 512 wide, scoring all 49,152 SmolLM2 pieces for every word is ~40% of a training step and 25M of the weights
(+ 12.6M in the value tables). With 8,192 pieces the score head is a sixth of that and the freed weights go to the
experts. The cost: each piece holds less text (about 3.6-3.9 bytes instead of ~4.4), so the same text is ~15-20% more
pieces. Loss per piece is not comparable across tokenizers; the trainer's bits per byte (val_bpb) is.

The pieces: byte-level BPE (any text can be written, nothing is unknown), <|endoftext|> = piece 0 like SmolLM2's, the
same GPT-2-style word split, and digits one by one (a small vocabulary cannot afford number pieces, and single digits
help arithmetic). Learned from Opal's main mix (web, DCLM, Cosmopedia, books, wiki, maths, code, chat, religion, Field1)
in its proportions, so the pieces fit what the model reads.

Re-cutting is exact: SmolLM2's pieces are bytes, so every document is turned back into its text and cut again; the
order of documents, the held-out text and every folder's layout stay as in <data>/opal. Each folder's meta.json keeps
both counts and the "ratio" (pieces per GPT-2 token) the trainer uses to put losses on one scale.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .data import default_dir
from . import tok

VOCAB = 8192
EOT = 0                                       # <|endoftext|> in both SmolLM2's pieces and ours
SAMPLE_GB = 2.0
SPLIT = 2_000_000                             # pieces per job when re-cutting


def opal_dir(data: Path) -> Path:
    for name in ("opal", "sado"):
        if (data / name / "train.bin").exists():
            return data / name
    raise SystemExit(f"no Opal in {data}/opal: build it first (python3 -m oazy.opal build)")


def folders(src: Path):
    """[(name, sub-folder)] of every part of Opal that has text: '' is FineWeb-Edu at the top."""
    out = [("fwe", "")] if (src / "train.bin").exists() else []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        if (d / "train.bin").exists():
            out.append((d.name, d.name))
    return out


_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        enc = tok.get("smollm2")
        _TABLE = [b"" if t == enc.eot_token else enc.decode_single_token_bytes(t) for t in range(enc.n_vocab)]
    return _TABLE


def docs(arr: np.ndarray):
    """The texts of the documents in a run of SmolLM2 pieces (ends at each <|endoftext|>)."""
    tb = _table()
    ends = np.flatnonzero(arr == EOT)
    starts = np.concatenate([[0], ends + 1])
    stops = np.concatenate([ends, [len(arr)]])
    for s, e in zip(starts, stops):
        yield b"".join(tb[t] for t in arr[s:e].tolist()).decode("utf-8", errors="replace"), e < len(arr)


# ------------------------------------------------------------------ learning the pieces
def weights(src: Path) -> dict:
    """Each part's share of the sample: Opal's main mix for MoE-150M (oazy/mix.py), Field1 at 2%."""
    from .mix import resolve_mix
    main = resolve_mix("opal100:flat:0.02").split(";")[0].split("@")[0]
    w = {k: float(v) for k, v in (s.split(":") for s in main.split(","))}
    from .mix import SOURCES
    by_folder = {SOURCES.get(k, k): v for k, v in w.items()}
    have = {sub: by_folder.get(sub, 0.0) for _, sub in folders(src)}
    have = {k: v for k, v in have.items() if v > 0}
    tot = sum(have.values())
    return {k: v / tot for k, v in have.items()}


def sample(src: Path, gb: float, seed: int = 0):
    """Texts from every part in proportion, from 400 random places in each (so no part is read from its start only)."""
    rng = np.random.default_rng(seed)
    for sub, share in weights(src).items():
        arr = np.memmap((src / sub if sub else src) / "train.bin", dtype=np.uint16, mode="r")
        want = int(gb * 1e9 * share / 4.2)                    # pieces (~4.2 bytes each)
        per = max(20_000, want // 400)
        got = 0
        print(f"  {sub or 'fwe':10s} {share:6.1%}  {want / 1e6:7.0f}M pieces of {len(arr) / 1e9:.2f}B", flush=True)
        while got < want:
            at = int(rng.integers(0, max(1, len(arr) - per)))
            chunk = np.asarray(arr[at:at + per])
            first = np.flatnonzero(chunk == EOT)
            if len(first) < 2:
                got += per
                continue
            for text, _ in docs(chunk[first[0] + 1:first[-1]]):
                if text:
                    yield text
            got += per


def train(data: Path, gb: float, vocab: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    src = opal_dir(data)
    out = data / "opal8k" / "tokenizer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"learning {vocab:,} pieces from ~{gb:g} GB of {src}", flush=True)
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, min_frequency=2, special_tokens=["<|endoftext|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    t0 = time.time()
    tk.train_from_iterator(sample(src, gb), trainer=tr)
    tmp = out.with_suffix(".json.tmp")
    tk.save(str(tmp))
    os.replace(tmp, out)
    assert tk.token_to_id("<|endoftext|>") == 0
    print(f"saved {out} ({tk.get_vocab_size():,} pieces) in {(time.time() - t0) / 60:.0f} min", flush=True)
    os.environ["OAZY_8K_TOKENIZER"] = str(out)
    check(data)


def check(data: Path, n: int = 3_000_000):
    """Bytes per piece on each part's held-out text, SmolLM2's and ours."""
    src = opal_dir(data)
    ours = tok.get("oazy8k")
    print(f"{'part':10s} {'SmolLM2':>8s} {'ours':>8s}  bytes per piece (held-out)")
    tb, to = 0, 0
    for name, sub in folders(src):
        p = (src / sub if sub else src) / "val.bin"
        if not p.exists():
            continue
        arr = np.asarray(np.memmap(p, dtype=np.uint16, mode="r")[:n])
        texts = [t for t, _ in docs(arr)]
        nb = sum(len(t.encode()) for t in texts)
        no = sum(len(x) for x in ours.encode_ordinary_batch(texts))
        ns = int((arr != EOT).sum())
        tb, to = tb + nb, to + no
        print(f"{name:10s} {nb / max(ns, 1):8.2f} {nb / max(no, 1):8.2f}")
    print(f"{'all':10s} {'':8s} {tb / max(to, 1):8.2f}")


# ------------------------------------------------------------------ re-cutting Opal
def _cuts(arr, step):
    n, cuts, at = len(arr), [0], 0
    while at < n:
        nxt = min(n, at + step)
        if nxt < n:
            j = np.flatnonzero(arr[nxt:min(n, nxt + 4_000_000)] == EOT)
            nxt = nxt + int(j[0]) + 1 if len(j) else min(n, nxt + 4_000_000)
        cuts.append(nxt)
        at = nxt
    return list(zip(cuts[:-1], cuts[1:]))


def _work(job):
    path, a, b = job
    arr = np.asarray(np.memmap(path, dtype=np.uint16, mode="r")[a:b])
    texts, ends = [], []
    for t, end in docs(arr):
        texts.append(t)
        ends.append(end)
    ours = tok.get("oazy8k")
    out, nbytes = [], 0
    for t, ids, end in zip(texts, ours.encode_ordinary_batch(texts), ends):
        nbytes += len(t.encode())
        out.extend(ids)
        if end:
            out.append(EOT)
    return np.asarray(out, dtype=np.uint16), b - a, nbytes


def convert(src: Path, dst: Path, pool):
    arr = np.memmap(src, dtype=np.uint16, mode="r")
    jobs = [(str(src), a, b) for a, b in _cuts(arr, SPLIT)]
    tmp = dst.with_suffix(".bin.tmp")
    n_old = n_new = n_bytes = 0
    t0 = time.time()
    with open(tmp, "wb") as f:
        for out, a, nb in pool.imap(_work, jobs):
            f.write(out.tobytes())
            n_old, n_new, n_bytes = n_old + a, n_new + out.size, n_bytes + nb
            if n_old // 500_000_000 != (n_old - a) // 500_000_000:
                rate = n_old / max(time.time() - t0, 1e-9)
                print(f"    {n_old / 1e9:.2f}B of {len(arr) / 1e9:.2f}B pieces, {rate / 1e6:.1f}M/s, "
                      f"~{(len(arr) - n_old) / rate / 60:.0f} min left", flush=True)
    os.replace(tmp, dst)
    return {"smol_tokens": n_old, "tokens": n_new, "bytes": n_bytes}


def cut(data: Path, workers: int):
    src = opal_dir(data)
    dst_root = data / "opal8k"
    if not (dst_root / "tokenizer.json").exists():
        raise SystemExit(f"no tokenizer at {dst_root / 'tokenizer.json'}: run  python3 -m oazy.tok8k train  first")
    os.environ["OAZY_8K_TOKENIZER"] = str(dst_root / "tokenizer.json")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok.get("oazy8k")
    _table()
    try:
        workers = workers or max(1, len(os.sched_getaffinity(0)) - 1)
    except AttributeError:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"re-cutting {src} -> {dst_root} with {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for name, sub in folders(src):
            a, b = (src / sub if sub else src), (dst_root / sub if sub else dst_root)
            b.mkdir(parents=True, exist_ok=True)
            old = json.loads((a / "meta.json").read_text()) if (a / "meta.json").exists() else {}
            done = json.loads((b / "meta.json").read_text()) if (b / "meta.json").exists() else {}
            if done.get("tokenizer") == "oazy8k" and (b / "train.bin").exists():
                print(f"{name}: already re-cut", flush=True)
                continue
            for f in a.iterdir():                     # the small files (its notes, level tables) come along as they are
                if f.is_file() and f.suffix not in (".bin", ".tmp") and f.name != "meta.json" and f.stat().st_size < 50e6:
                    shutil.copy2(f, b / f.name)
            meta = {k: v for k, v in old.items() if k not in ("train", "val", "train_tokens", "val_tokens")}
            meta.update(tokenizer="oazy8k", source=str(a))
            for split in ("val", "train"):
                if not (a / f"{split}.bin").exists():
                    continue
                print(f"{name} {split}", flush=True)
                m = convert(a / f"{split}.bin", b / f"{split}.bin", pool)
                was = old.get(split, {}) if isinstance(old.get(split), dict) else {}
                smol_ratio = float(was.get("ratio") or 0.96)          # SmolLM2 pieces per GPT-2 token (Opal's meta, else ~0.96)
                m["gpt2_tokens"] = int(m["smol_tokens"] / smol_ratio)
                m["ratio"] = m["tokens"] / max(m["gpt2_tokens"], 1)
                meta[split] = m
                print(f"  {m['smol_tokens'] / 1e6:.0f}M SmolLM2 pieces -> {m['tokens'] / 1e6:.0f}M ours "
                      f"({m['tokens'] / max(m['smol_tokens'], 1):.3f}x, {m['bytes'] / max(m['tokens'], 1):.2f} bytes each)", flush=True)
            meta["train_tokens"] = meta.get("train", {}).get("tokens", 0)
            (b / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {(time.time() - t0) / 60:.0f} min: {dst_root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "cut", "check"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-gb", type=float, default=SAMPLE_GB)
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data or default_dir("data"))
    if a.what != "train" and "OAZY_8K_TOKENIZER" not in os.environ:
        os.environ["OAZY_8K_TOKENIZER"] = str(data / "opal8k" / "tokenizer.json")
    {"train": lambda: train(data, a.sample_gb, a.vocab), "cut": lambda: cut(data, a.workers), "check": lambda: check(data)}[a.what]()


if __name__ == "__main__":
    main()
