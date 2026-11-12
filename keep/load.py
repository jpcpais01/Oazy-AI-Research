"""How much can Keep hold at once? The saved models are told a book of plain facts, 8 to 2,048 of them, every fact
about a different subject, and are then asked about some of them. No training.

    python3 -m keep.load --name keep-fast                       # a few minutes on the GPU
    python3 -m keep.load --name keep-fast --facts 512,2048 --gap 4096 --arms keep,mixed

The exam of keep.run holds one scene of 3 to 10 facts at a time, which is what the models trained on. Here nothing is
between the facts: a book of 512 is about 6,000 tokens of them, paragraph after paragraph (keep/ledger.py: book), then
`--gap` tokens of ordinary text, then up to 24 of the facts are asked, drawn evenly over the book. A store has 16,384
rows and every token writes 8 of them, so the question is whether one fact's rows survive the next thousand facts.

Keep answers out of its stores alone (the questions are a window of their own). The twin sees the last 1,024 tokens:
a book that fits is its ceiling under the same questions, and one that does not fit is mostly out of its view.
Keep-wide and Keep-wide mixed read as a stream does. Writes <runs>/<name>/load.md and load.json.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch

from . import data as D
from .model import Keep, KeepConfig
from .run import ARMS, NAMES, WIDE, exam, runs_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="keep")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sub", default="opal")
    ap.add_argument("--facts", default="8,32,128,512,2048", help="facts in a book, each size examined")
    ap.add_argument("--n", type=int, default=64, help="books per size")
    ap.add_argument("--ask", type=int, default=24, help="facts asked about per book, at most")
    ap.add_argument("--gap", type=int, default=0, help="tokens of ordinary text between the book and the questions")
    ap.add_argument("--arms", default=None, help="which saved models, e.g. keep,full (default: all that are there)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if args.data is None:
        from oazy.data import default_dir
        args.data = default_dir("data")
    if args.sub and (Path(args.data) / args.sub).exists():
        args.data = str(Path(args.data) / args.sub)
    from oazy import tok
    enc = tok.get("smollm2")
    out = Path(args.runs or runs_dir()) / args.name
    text = D.Text(Path(args.data), "val")
    sizes = [int(f) for f in args.facts.split(",")]
    arms = [a for a in (args.arms.split(",") if args.arms else ARMS) if (out / a / "ckpt.pt").exists()]
    if not arms:
        raise SystemExit(f"no saved model under {out}")
    say = lambda s: print(s, flush=True)
    make = partial(D.load_set, ask=args.ask, gap=args.gap)
    tokens = {m: int((s[:, 1] - s[:, 0]).mean()) for m in sizes for s in [D.load_set(enc, text, m, 4)[3]]}
    res = {}
    for arm in arms:
        sd = torch.load(out / arm / "ckpt.pt", map_location="cpu", weights_only=False)
        model = Keep(KeepConfig.from_dict(sd["config"]))
        model.load_state_dict(sd["model"])
        model.to(device)
        say(f"[load] {NAMES[arm]}:")
        res[arm] = exam(model, enc, text, sizes, args.n, device, say=say, alone=arm not in WIDE, make=make, word="facts")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def cell(arm, m, kind=None):
        r = res[arm][str(m)]
        if kind is None:
            return f"{r['groups']['facts']['right']}%"
        a = r["families"].get(kind)
        return f"{100 * a[0] / a[1]:.1f}%" if a else "-"
    head = "| | " + " | ".join(f"{m:,} facts" for m in sizes) + " |"
    L = ["# Keep: how much can it hold at once?", "",
         f"The saved models of `{args.name}`. A book of plain facts, every fact about a different subject, is told once; "
         + (f"then {args.gap:,} tokens of ordinary text; " if args.gap else "")
         + f"then up to {args.ask} of its facts are asked, drawn evenly over the book. {args.n} books per size. Share right:", "",
         head, "|---|" + "---|" * len(sizes),
         "| Tokens in the book | " + " | ".join(f"{tokens[m]:,}" for m in sizes) + " |"]
    L += [f"| {NAMES[a]} | " + " | ".join(cell(a, m) for m in sizes) + " |" for a in arms]
    L += ["| Guessing | " + " | ".join(f"{res[arms[0]][str(m)]['groups']['facts']['chance']}%" for m in sizes) + " |", "",
          "Keep answers out of its stores alone. The twin sees the last "
          f"{json.loads((out / 'results.json').read_text())['window_full'] if (out / 'results.json').exists() else 1024:,} tokens "
          "(the questions included): a book that fits is its ceiling, a larger one is mostly out of its view. "
          "The models trained on one scene of 3 to 10 facts at a time.", ""]
    for a in arms:
        L += [f"**{NAMES[a]}, by when the fact was told**", "", head, "|---|" + "---|" * len(sizes)]
        L += [f"| {name} | " + " | ".join(cell(a, m, kind) for m in sizes) + " |"
              for kind, name in zip(D.WHEN, ("In the first third of the book", "In the middle third", "In the last third"))]
        L.append("")
    report = "\n".join(L)
    (out / "load.md").write_text(report)
    (out / "load.json").write_text(json.dumps({"facts": sizes, "gap": args.gap, "n": args.n, "tokens": tokens, "exam": res}, indent=1))
    print("\n" + report + f"\nwritten: {out / 'load.md'}", flush=True)


if __name__ == "__main__":
    main()
