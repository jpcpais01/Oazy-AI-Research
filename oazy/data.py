"""FineWeb-Edu, tokenized with GPT-2's tokenizer into flat uint16 files (nanoGPT style).

    python -m oazy.data --out /workspace/oazy_data --tokens 4e9

FineWeb-Edu is web pages filtered for educational value (Hugging Face, 2024); small models trained
on it beat GPT-2 with far less text. We download the "sample-10BT" parquet files one by one,
tokenize them on every CPU core, and write:

  val.bin     the first --val-tokens tokens (whole documents, never trained on)
  train.bin   the next --tokens tokens
  hellaswag_val.jsonl   the HellaSwag benchmark (10,042 sentence endings to choose from)
  meta.json

Every document ends with <|endoftext|>. Training reads windows of 1,025 tokens in a shuffled
order without repeats (an "epoch" covers every window once), the same order for both models.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import threading
import queue
from pathlib import Path

import numpy as np

EOT = 50256


def default_dir(what: str) -> str:
    """Where the lab keeps its text ("data") and its training runs ("runs") on the pod.

    $OAZY_DATA / $OAZY_RUNS win; otherwise /workspace/oazy_data and /workspace/oazy_runs, except that
    the folders from the Engram days (/workspace/oazy_data, /workspace/oazy_runs) keep being used
    while they are the only ones there, so nothing has to be moved."""
    env = os.environ.get("OAZY_" + what.upper())
    if env:
        return env
    new, old = f"/workspace/oazy_{what}", f"/workspace/engram_{what}"
    return old if (os.path.exists(old) and not os.path.exists(new)) else new
HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"


@functools.lru_cache(maxsize=None)
def get_tokenizer():
    import tiktoken
    path = os.environ.get("ENGRAM_GPT2_TIKTOKEN") or os.environ.get("SETTLE_GPT2_TIKTOKEN")
    try:
        if path:
            raise RuntimeError("use the local copy")
        return tiktoken.get_encoding("gpt2")
    except Exception:
        # offline: a local copy of GPT-2's ranks (same vocabulary), e.g. whisper's assets/gpt2.tiktoken
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
    out = enc.encode_ordinary_batch(texts)
    arr = np.fromiter((t for ids in out for t in ids + [EOT]), dtype=np.uint16)
    return arr


def write_tokens(text_batches, out: Path, val_tokens: int, train_tokens: int, workers: int = 0):
    """Tokenize batches of documents in parallel; fill val.bin first (whole documents), then train.bin."""
    from multiprocessing import Pool
    workers = workers or max(1, min(64, (os.cpu_count() or 2) - 1))
    fv, ft = open(out / "val.bin.tmp", "wb"), open(out / "train.bin.tmp", "wb")
    nv = nt = 0
    import itertools
    it = iter(text_batches)
    with Pool(workers) as pool:
        while nt < train_tokens:
            group = list(itertools.islice(it, 2 * workers))      # a bounded amount of text in flight
            if not group:
                print("  the dataset ran out before the requested number of tokens", flush=True)
                break
            for arr in pool.imap(_encode_batch, group):
                if nv < val_tokens:
                    fv.write(arr.tobytes())
                    nv += arr.size
                    continue
                arr = arr[: train_tokens - nt]
                ft.write(arr.tobytes())
                nt += arr.size
                if nt // 100_000_000 != (nt - arr.size) // 100_000_000:
                    print(f"  train.bin: {nt / 1e9:.2f}B / {train_tokens / 1e9:.2f}B tokens", flush=True)
    fv.close(), ft.close()
    os.replace(out / "val.bin.tmp", out / "val.bin")
    os.replace(out / "train.bin.tmp", out / "train.bin")
    return nv, nt


def fineweb_edu_batches(sample: str, batch_docs: int = 2048):
    """Yield lists of document texts, downloading the parquet files one ahead of use."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    repo = "HuggingFaceFW/fineweb-edu"
    folder = sample.removeprefix("sample-")          # "sample-10BT" lives in sample/10BT/
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(f"sample/{folder}/") and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no parquet files found for sample {sample}")
    print(f"{len(files)} parquet files in {repo} sample/{sample}", flush=True)
    q: queue.Queue = queue.Queue(maxsize=1)

    def fetch():
        for f in files:
            q.put(hf_hub_download(repo, f, repo_type="dataset"))
        q.put(None)
    threading.Thread(target=fetch, daemon=True).start()
    while (path := q.get()) is not None:
        print(f"  reading {Path(path).name}", flush=True)
        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=batch_docs, columns=["text"]):
            yield rb.column(0).to_pylist()


def fetch_hellaswag(out: Path):
    dst = out / "hellaswag_val.jsonl"
    if dst.exists():
        return
    import urllib.request
    try:
        urllib.request.urlretrieve(HELLASWAG_URL, dst.with_suffix(".tmp"))
        os.replace(dst.with_suffix(".tmp"), dst)
        print(f"HellaSwag -> {dst}")
        return
    except Exception as e:
        print(f"GitHub download of HellaSwag failed ({e}); trying the Hugging Face copy")
    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation")
        with open(dst.with_suffix(".tmp"), "w") as f:
            for ex in ds:
                f.write(json.dumps({"ctx": ex["ctx"], "endings": ex["endings"], "label": int(ex["label"])}) + "\n")
        os.replace(dst.with_suffix(".tmp"), dst)
        print(f"HellaSwag ({len(ds)} questions) -> {dst}")
    except Exception as e:
        print(f"could not download HellaSwag ({e}); training will skip it")


def prepare(out: Path, tokens: int, val_tokens: int, sample: str, workers: int = 0):
    out.mkdir(parents=True, exist_ok=True)
    fetch_hellaswag(out)
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["train_tokens"] >= tokens:
            print(f"data already prepared in {out}: {meta}")
            return
        print(f"existing data has {meta['train_tokens'] / 1e9:.2f}B tokens; preparing {tokens / 1e9:.2f}B")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        import hf_transfer  # noqa: F401  (much faster downloads when installed)
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    get_tokenizer()           # fetch GPT-2's vocabulary once, before the worker processes start
    nv, nt = write_tokens(fineweb_edu_batches(sample), out, val_tokens, tokens, workers)
    meta = {"dataset": f"HuggingFaceFW/fineweb-edu {sample}", "tokenizer": "gpt2", "eot": EOT,
            "val_tokens": nv, "train_tokens": nt}
    meta_path.write_text(json.dumps(meta, indent=1))
    print("done:", meta)


class Windows:
    """Training windows of T+1 tokens, in a shuffled order without repeats (reproducible and resumable)."""

    def __init__(self, path, block: int, batch: int, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block, self.batch, self.seed = block, batch, seed
        self.n = (len(self.data) - 1) // block
        self.epoch, self.pos = 0, 0
        self._order()

    def _order(self):
        self.order = np.random.default_rng(self.seed + 1000 * self.epoch).permutation(self.n)

    def get(self, device):
        import torch
        if self.pos + self.batch > self.n:
            self.epoch += 1
            self.pos = 0
            self._order()
        ix = self.order[self.pos:self.pos + self.batch] * self.block
        self.pos += self.batch
        x = np.stack([self.data[i:i + self.block + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x)
        x = x.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else x.to(device)
        return x[:, :-1], x[:, 1:]

    def state(self):
        return {"epoch": self.epoch, "pos": self.pos}

    def set_state(self, s):
        self.epoch, self.pos = s["epoch"], s["pos"]
        self._order()


def val_set(path, block: int, n: int):
    """The fixed validation windows: n evenly spaced windows of T+1 tokens (same for every model and step)."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = np.linspace(0, len(data) - block - 2, n).astype(np.int64)
    return np.stack([data[i:i + block + 1] for i in starts]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokens", type=float, default=4e9, help="training tokens to prepare")
    ap.add_argument("--val-tokens", type=float, default=1e7)
    ap.add_argument("--sample", default="sample-10BT", help="FineWeb-Edu sample: sample-10BT, sample-100BT, ...")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--hellaswag-only", action="store_true", help="only fetch the HellaSwag file")
    a = ap.parse_args()
    a.out = a.out or default_dir("data")
    if a.hellaswag_only:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        return fetch_hellaswag(Path(a.out))
    prepare(Path(a.out), int(a.tokens), int(a.val_tokens), a.sample, a.workers)


if __name__ == "__main__":
    main()
