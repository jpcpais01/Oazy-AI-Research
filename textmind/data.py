"""Stories cut into sentences, as bytes.

A story becomes S sentences (the "moments"), each at most L bytes followed by an end mark.
Both minds read exactly the same bytes: the sentence mind sentence by sentence, the next-byte
baseline as one stream (sentence bytes + end mark, one after another).

Source: TinyStories (V2, GPT-4 stories), downloaded once from the Hugging Face hub.
"""
from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

import torch

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt"
EOS, PAD = 256, 257
VOCAB = 258
SPLIT = re.compile(r"(?<=[.!?])\s+")


def download(root, split, max_mb):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"tinystories_{split}_{max_mb}mb.txt"
    if not p.exists():
        req = urllib.request.Request(URL.format(split), headers={"Range": f"bytes=0-{max_mb * 1_000_000}",
                                                                 "User-Agent": "textmind"})
        with urllib.request.urlopen(req) as r, open(p, "wb") as f:
            f.write(r.read())
    return p.read_text(errors="ignore")


def stories_to_tensor(text, S, L):
    """(N, S, L+1) uint8-ish long tensor of sentence bytes + EOS + PAD; only stories with >= S sentences."""
    out = []
    for story in text.split("<|endoftext|>"):
        sents = [s.strip() for s in SPLIT.split(" ".join(story.split())) if s.strip()]
        if len(sents) < S:
            continue
        sents = sents[:S]
        if any(len(s.encode()) > L for s in sents):
            continue
        row = torch.full((S, L + 1), PAD, dtype=torch.int16)
        for i, s in enumerate(sents):
            b = list(s.encode())
            row[i, :len(b)] = torch.tensor(b, dtype=torch.int16)
            row[i, len(b)] = EOS
        out.append(row)
    return torch.stack(out)


def load(root, S=12, L=96, train_mb=400):
    cache = Path(root) / f"ts_S{S}_L{L}_{train_mb}mb.pt"
    if cache.exists():
        return torch.load(cache)
    tr = stories_to_tensor(download(root, "train", train_mb), S, L)
    va = stories_to_tensor(download(root, "valid", 25), S, L)
    d = dict(train=tr, valid=va)
    torch.save(d, cache)
    return d


def as_stream(x):
    """(B, S, L+1) -> (B, S*(L+1)): each story's bytes in order with the PADs moved to the end."""
    v = x.flatten(1)
    order = torch.sort((v == PAD).to(torch.int8), dim=1, stable=True).indices
    return torch.gather(v, 1, order)
