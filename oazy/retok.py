"""Re-cut the lab's prepared text with another tokenizer (for Bee0.2B v2: SmolLM2's).

    python -m oazy.retok                       # every source there is -> <data>/smol/   (~15-40 min)
    python -m oazy.retok --sources fwe,chat    # only some
    python -m oazy.retok --max-tokens 3.2e9    # at most this many (GPT-2) tokens of each source's training text

Nothing is downloaded again. The GPT-2 token files made by oazy/data.py and oazy/mix.py are turned
back into text document by document (GPT-2's pieces are bytes, so this is exact), and each document
is cut again with the new tokenizer:

  <data>/train.bin, val.bin                ->  <data>/smol/train.bin, val.bin          (FineWeb-Edu)
  <data>/dclm/train.bin, val.bin           ->  <data>/smol/dclm/train.bin, val.bin     (DCLM)
  <data>/chat/train.bin, val.bin           ->  <data>/smol/chat/train.bin, val.bin     (conversations)

So Bee0.2B v2 reads exactly the same documents as Bee0.2B, in the same order, and its held-out
text is exactly the same held-out text: that is what makes the two comparable. Each folder gets a
meta.json with the token counts of both tokenizers and the number of bytes, e.g. "ratio": 0.96
means SmolLM2 needs 4% fewer tokens than GPT-2 for this text. It is safe to run while a model trains
(it only uses the CPU; lower its priority with `nice -n 19` if you like).
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .data import default_dir, EOT
from .mix import SOURCES, source_path
from . import tok

FOLDER = {"smollm2": "smol"}


def target_dir(data: Path, name: str) -> Path:
    return Path(data) / FOLDER[name]


def _cuts(arr: np.ndarray, step: int):
    """Split points about `step` tokens apart, each just after an end-of-text token (never inside a document)."""
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
    path, a, b, name = job
    src = np.memmap(path, dtype=np.uint16, mode="r")[a:b]
    gpt = tok.get("gpt2")
    new = tok.get(name)
    ends = np.flatnonzero(src == EOT)
    starts = np.concatenate([[0], ends + 1])
    stops = np.concatenate([ends, [len(src)]])
    texts, nbytes = [], 0
    for s, e in zip(starts, stops):
        if e > s:
            raw = gpt.decode_bytes(src[s:e].tolist())
            nbytes += len(raw)
            texts.append(raw.decode("utf-8", errors="replace"))
        elif e < len(src):
            texts.append("")                             # an empty document: keep its end mark
    has_end = len(src) > 0 and src[-1] == EOT
    out = []
    for i, ids in enumerate(new.encode_ordinary_batch(texts)):
        out.extend(ids)
        if i < len(texts) - 1 or has_end:
            out.append(new.eot_token)
    return np.asarray(out, dtype=np.uint16), b - a, nbytes


def convert(src: Path, dst: Path, name: str, max_tokens: int = 0, workers: int = 0, step: int = 2_000_000):
    arr = np.memmap(src, dtype=np.uint16, mode="r")
    n = len(arr)
    if max_tokens and max_tokens < n:
        j = np.flatnonzero(arr[max_tokens:max_tokens + 4_000_000] == EOT)
        n = max_tokens + (int(j[0]) + 1 if len(j) else 0)
    jobs = [(str(src), a, b, name) for a, b in _cuts(arr[:n], step)]
    workers = workers or max(1, min(64, (os.cpu_count() or 2) - 1))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"       # one process per core already
    tok.get(name)                                         # fetch the tokenizer once, before the workers start
    tmp = dst.with_suffix(".bin.tmp")
    n_old = n_new = n_bytes = 0
    with open(tmp, "wb") as f, Pool(workers) as pool:
        for out, a, b in pool.imap(_work, jobs):
            f.write(out.tobytes())
            n_old, n_new, n_bytes = n_old + a, n_new + out.size, n_bytes + b
            if n_old // 200_000_000 != (n_old - a) // 200_000_000:
                print(f"  {dst.parent.name}/{dst.name}: {n_old / 1e9:.2f}B of {n / 1e9:.2f}B GPT-2 tokens done", flush=True)
    os.replace(tmp, dst)
    return {"gpt2_tokens": n_old, "tokens": n_new, "bytes": n_bytes, "ratio": n_new / max(n_old, 1)}


def run(data: Path, name: str, sources, max_tokens: int = 0, workers: int = 0):
    out_root = target_dir(data, name)
    done = {}
    for s in sources:
        if not source_path(data, s, "train").exists():
            print(f"{s}: no GPT-2 text at {source_path(data, s, 'train')}, skipped (prepare it first)")
            continue
        out = out_root / SOURCES[s] if SOURCES[s] else out_root
        out.mkdir(parents=True, exist_ok=True)
        meta_path = out / "meta.json"
        old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        size = len(np.memmap(source_path(data, s, "train"), dtype=np.uint16, mode="r"))
        want = min(size, max_tokens) if max_tokens else size
        if old.get("train", {}).get("gpt2_tokens", 0) >= want and (out / "val.bin").exists():
            print(f"{s}: already re-cut in {out}: {old['train']}")
            done[s] = old
            continue
        meta = {"tokenizer": name, "from": "gpt2", "source": str(source_path(data, s, "train").parent)}
        for split in ("val", "train"):
            p = source_path(data, s, split)
            if not p.exists():
                continue
            print(f"{s} {split}: {p} -> {out / (split + '.bin')}", flush=True)
            meta[split] = convert(p, out / f"{split}.bin", name, max_tokens if split == "train" else 0, workers)
            print(f"  {meta[split]}", flush=True)
        meta["train_tokens"] = meta.get("train", {}).get("tokens", 0)
        meta_path.write_text(json.dumps(meta, indent=1))
        done[s] = meta
    hs = Path(data) / "hellaswag_val.jsonl"
    if hs.exists() and not (out_root / hs.name).exists():
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / hs.name).write_bytes(hs.read_bytes())   # HellaSwag is plain text: the same file serves both
    return done


def ratio(folder: Path, split: str = "val", default: float = 1.0) -> float:
    """New tokens per GPT-2 token for the text in this folder (from its meta.json)."""
    try:
        m = json.loads((Path(folder) / "meta.json").read_text())
        return float(m[split]["ratio"])
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default="smollm2", choices=list(FOLDER))
    ap.add_argument("--data", default=None)
    ap.add_argument("--sources", default="fwe,dclm,chat")
    ap.add_argument("--max-tokens", type=float, default=0, help="at most this many GPT-2 tokens of each source's training text")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data or default_dir("data"))
    got = run(data, a.to, [s.strip() for s in a.sources.split(",") if s.strip()], int(a.max_tokens), a.workers)
    for s, m in got.items():
        t = m.get("train", {})
        print(f"{s:5s} {t.get('gpt2_tokens', 0) / 1e9:6.2f}B GPT-2 tokens -> {t.get('tokens', 0) / 1e9:6.2f}B {a.to} tokens "
              f"(ratio {t.get('ratio', 0):.3f})")


if __name__ == "__main__":
    main()
