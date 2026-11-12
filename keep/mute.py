"""Which store carries the facts? The saved Keep takes its exam again with the reads of one store thrown away, then
of the other, then of all. No training.

    python3 -m keep.mute --name keep-fast                 # about three minutes on the GPU
    python3 -m keep.mute --name keep-fast --gaps 0,2048,32768 --n 256

A store whose muting leaves the score where it was is not what answers; one whose muting drops the score to guessing
is. The model was trained with all its stores, so a drop says "this model uses it", not "no model could do without":
the proof that one store is enough is a run with one (`python3 -m keep.run --only keep --stores 6 --name keep-one`).
Writes <runs>/<name>/mute.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import data as D
from .model import Keep, KeepConfig
from .run import exam, runs_dir


KINDS = (("facts", "Plain facts"), ("changed", "Facts changed after telling"), ("field1", "Field1 stories"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="keep")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sub", default="opal")
    ap.add_argument("--gaps", default="0,2048,32768")
    ap.add_argument("--n", type=int, default=256, help="scenes per gap")
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
    sd = torch.load(out / "keep" / "ckpt.pt", map_location="cpu", weights_only=False)
    model = Keep(KeepConfig.from_dict(sd["config"]))
    model.load_state_dict(sd["model"])
    model.to(device)
    text = D.Text(Path(args.data), "val")
    gaps = [int(g) for g in args.gaps.split(",")]
    stores = sorted(int(i) for i in model.store)
    cases = [("nothing muted", set())] + [(f"store after block {i} muted", {i}) for i in stores]
    if len(stores) > 1:
        cases.append(("all stores muted", set(stores)))
    cells = {}
    for name, muted in cases:
        model.muted = set(muted)
        print(f"[mute] {name}:", flush=True)
        ex = exam(model, enc, text, gaps, args.n, device, say=lambda s: print(s, flush=True))
        for kind, _ in KINDS:
            cells[(name, kind)] = [f"{g[kind]['right']}% ({g[kind]['chance']}%)" if kind in g else "-" for g in (ex[str(gap)]["groups"] for gap in gaps)]
    model.muted = set()
    L = ["# Keep: which store carries the facts?", "",
         f"The saved model of `{args.name}`, examined again with the reads of some stores thrown away. "
         f"{args.n} scenes per gap, answers out of the stores alone. Share right (guessing in brackets).", ""]
    for kind, title in KINDS:
        L += [f"**{title}**", "", "| | " + " | ".join(f"gap {g:,}" for g in gaps) + " |", "|---|" + "---|" * len(gaps)]
        L += [f"| {name} | " + " | ".join(cells[(name, kind)]) + " |" for name, _ in cases]
        L.append("")
    report = "\n".join(L)
    (out / "mute.md").write_text(report)
    print("\n" + report + f"\nwritten: {out / 'mute.md'}", flush=True)


if __name__ == "__main__":
    main()
