"""TinyStories with the GPT-2 tokenizer, stored as flat uint16 token files (nanoGPT style).

    python -m settle.data --out /workspace/settle_data

Downloads roneneldan/TinyStories from Hugging Face, tokenizes every story with
GPT-2's BPE (tiktoken), puts an <|endoftext|> token after each story, and
writes train.bin / val.bin. Training then samples random windows from them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

EOT = 50256


def get_tokenizer():
    import tiktoken
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception:
        # offline fallback: a local copy of GPT-2's ranks (same vocabulary), e.g. whisper's assets/gpt2.tiktoken
        path = os.environ.get("SETTLE_GPT2_TIKTOKEN")
        if not path:
            raise
        import base64
        ranks = {}
        for line in open(path, "rb"):
            if line.strip():
                tok, r = line.split()
                ranks[base64.b64decode(tok)] = int(r)
        return tiktoken.Encoding(
            name="gpt2",
            pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=ranks,
            special_tokens={"<|endoftext|>": EOT},
        )


def _encode_batch(texts):
    enc = get_tokenizer()
    out = enc.encode_ordinary_batch([t.strip() for t in texts])
    return [ids + [EOT] for ids in out]


def write_split(texts, path: Path, workers: int = 0, chunk: int = 4096):
    """Tokenize an iterable of strings into a flat uint16 file. Returns the number of tokens."""
    from concurrent.futures import ProcessPoolExecutor
    texts = list(texts)
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    tmp = path.with_suffix(".tmp")
    n = 0
    workers = workers or max(1, min(32, (os.cpu_count() or 2) - 1))
    with open(tmp, "wb") as f:
        if workers > 1 and len(chunks) > 1:
            with ProcessPoolExecutor(workers) as ex:
                for i, res in enumerate(ex.map(_encode_batch, chunks)):
                    arr = np.fromiter((t for ids in res for t in ids), dtype=np.uint16)
                    f.write(arr.tobytes())
                    n += arr.size
                    if i % 50 == 0:
                        print(f"  {path.name}: {i + 1}/{len(chunks)} chunks, {n / 1e6:.1f}M tokens", flush=True)
        else:
            for res in map(_encode_batch, chunks):
                arr = np.fromiter((t for ids in res for t in ids), dtype=np.uint16)
                f.write(arr.tobytes())
                n += arr.size
    os.replace(tmp, path)
    return n


def prepare(out: Path, dataset=None, workers: int = 0):
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists():
        print(f"data already prepared in {out}: {json.loads(meta_path.read_text())}")
        return
    if dataset is None:
        from datasets import load_dataset
        dataset = load_dataset("roneneldan/TinyStories")
    get_tokenizer()          # fetch the GPT-2 vocabulary once, before the worker processes start
    meta = {"tokenizer": "gpt2", "eot": EOT}
    for split, name in (("train", "train"), ("validation", "val")):
        texts = dataset[split]["text"]
        print(f"tokenizing {split}: {len(texts)} stories", flush=True)
        meta[name + "_tokens"] = write_split(texts, out / f"{name}.bin", workers)
        meta[name + "_stories"] = len(texts)
    meta_path.write_text(json.dumps(meta, indent=1))
    print("done:", meta)


class Batches:
    """Random windows of length T+1 from a uint16 token file."""

    def __init__(self, path, block: int, batch: int, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block, self.batch = block, batch
        self.rng = np.random.default_rng(seed)

    def get(self, device):
        import torch
        ix = self.rng.integers(0, len(self.data) - self.block - 1, self.batch)
        x = np.stack([self.data[i:i + self.block + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        return x[:, :-1], x[:, 1:]

    def state(self):
        return self.rng.bit_generator.state

    def set_state(self, s):
        self.rng.bit_generator.state = s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/settle_data")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    prepare(Path(a.out), workers=a.workers)


if __name__ == "__main__":
    main()
