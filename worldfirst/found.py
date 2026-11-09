"""How cleanly a no-marks Laws model's finder separates the things, per family, from a run's last saved state.

    python3 -m worldfirst.found --run world-teach            # its latest train_state.pt (or ckpt.pt)

A thing counts as found when all of its words went mostly to one slot and no other thing's words went there.
Also shows how many slots a thing's words are spread over, and how many things share a slot, on average."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from . import world as W
from .data import Vocab, finder_batch
from .model import from_config


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="world-teach")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--n", type=int, default=48, help="scenes per family and level")
    a = ap.parse_args(argv)
    if a.runs is None:
        from oazy.data import default_dir
        a.runs = default_dir("runs")
    d = Path(a.runs) / a.run / "laws"
    st = d / "train_state.pt" if (d / "train_state.pt").exists() else d / "ckpt.pt"
    sd = torch.load(st, map_location="cpu", weights_only=False)
    cfg = sd.get("config") or json.loads((d / "config.json").read_text())
    if not cfg.get("finder"):
        raise SystemExit("this run's Laws model reads the marks: it has no finder")
    W.STYLE["natural"] = True
    vocab = Vocab()
    m = from_config(cfg)
    m.load_state_dict(sd["model"])
    m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev)
    print(f"{a.run}: Laws model at step {sd.get('step', 'end')}, {cfg['finder']} slots")
    print(f"{'family':10} {'level':>5} {'found':>7} {'slots/thing':>12} {'things/slot':>12}")
    for fam in W.FAMILIES:
        for lv in sorted({W.TRAIN_LEVELS[fam][0], W.TRAIN_LEVELS[fam][-1]}):
            eps = [W.make(fam, lv, "train", 900_000 + s) for s in range(a.n)]
            got = tot = 0
            spread, share = [], []
            for i in range(0, a.n, 16):
                b = finder_batch(eps[i:i + 16], vocab, dev)
                with torch.no_grad():
                    g, t = m.found(b)
                got, tot = got + g, tot + t
                att = m._att.float().argmax(1).cpu()
                ment = b["ment"].cpu()
                for j in range(att.shape[0]):
                    sl = {}
                    for t_, s_ in zip(ment[j].tolist(), att[j].tolist()):
                        if t_ >= 0:
                            sl.setdefault(t_, set()).add(s_)
                    if not sl:
                        continue
                    spread += [len(v) for v in sl.values()]
                    c = Counter(s_ for v in sl.values() for s_ in v)
                    share += list(c.values())
            print(f"{fam:10} {lv:>5} {100 * got / max(1, tot):6.1f}% {sum(spread) / max(1, len(spread)):12.2f} "
                  f"{sum(share) / max(1, len(share)):12.2f}")
    print("\nfound: all of a thing's words in one slot of its own. slots/thing: 1 is ideal. things/slot: 1 is ideal.")


if __name__ == "__main__":
    main()
