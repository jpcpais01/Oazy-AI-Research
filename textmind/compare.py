"""Compare the two minds and let the sentence mind dream.

    python3 -m textmind.compare /workspace/oazy_runs/textmind
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .data import EOS, load
from .model import SentenceMind


def text(row):
    return bytes([b for b in row.tolist() if b < 256]).decode(errors="replace")


def main():
    root = Path(sys.argv[1])
    for arm in ("gpt", "sent"):
        p = root / arm / "result.json"
        if p.exists():
            print(arm, json.loads(p.read_text()))
    ck = torch.load(root / "sent" / "model.pt", map_location="cpu")
    a = ck["args"]
    m = SentenceMind(a["L"]); m.load_state_dict(ck["model"]); m.eval()
    va = load(a["data"], a["S"], a["L"], a["train_mb"])["valid"]
    torch.manual_seed(0)
    x = va[:4].long()
    said = m.dream(x[:, :3], 4)
    for i in range(4):
        print("\n--- read:", " ".join(text(s) for s in x[i, :3]))
        print("    dreamed on:", " ".join(text(s[i]) for s in said))
        print("    real next:", " ".join(text(s) for s in x[i, 3:7]))


if __name__ == "__main__":
    main()
