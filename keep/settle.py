"""Settle: does a fed model's store stop changing when its reasoning is done?

    python3 -m keep.settle --name c13                  # a trained think run (--align runs only), its fed model
    python3 -m keep.settle --name c13 --bar 0.03       # another bar

A model that loops works a scene out one step a telling, and writes what it worked out into the stores. Once every
value is in place a further telling writes the same values again, so the stores stand still. This measures, on the
exam's circuits (built to depth 1, 2, 3, 4, 6, 8, never trained on), how much the stores change from one telling to
the next (the norm of the change over the norm of the stores, each scene on its own), and tries one rule:

    stop reading when the stores changed by less than --bar (0.05)

For each depth: the tellings the rule uses, the share of the deepest question right when it stops, and the share
right after the most tellings. If the rule loses nothing and uses more tellings on deeper scenes, the model itself
says how long it has to read, and nobody has to tell it the depth. Writes <runs>/<name>/settle.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import think as T
from .model import Keep, KeepConfig
from .run import amp, runs_dir


@torch.no_grad()
def settle(model: Keep, enc, n: int, tellings, device, family="circuits", batch=32, seed=0):
    """-> {depth: [(change after telling 2.., right after telling 1..) per scene]}: for every exam scene of `family`
    built to a depth, the relative change of the stores at each telling and whether its deepest question is right."""
    assert model.cfg.feed and model.store, "a fed model (stores written from the top)"
    model.eval()
    eot = enc.eot_token
    told = [T.told(enc, family, i, (T.EXAM_SEED, seed)) for i in range(n)]
    C = max(max(len(st), sum(len(lead) + len(ans) for lead, ans, *_ in lines)) for st, _, lines, _ in told)
    C = max(model.cfg.chunk, -(-C // 16) * 16)
    was, model.cfg.chunk = model.cfg.chunk, C
    out = {}
    try:
        for lo in range(0, n, batch):
            ids = list(range(lo, min(n, lo + batch)))
            prev, change, right, deep = None, [], [], None
            for R in tellings:
                scs = [T.scene_tokens(enc, family, i, (T.EXAM_SEED, seed), R, 1 << 30, align=C) for i in ids]
                assert all(s is not None and s[3] == R for s in scs)
                L = max(len(s[0]) for s in scs)
                x = np.full((len(ids), L), eot, dtype=np.int64)
                for r, s in enumerate(scs):
                    x[r, L - len(s[0]):] = s[0]
                x = torch.from_numpy(x).to(device)
                with amp(device):
                    h = model.hidden(x, model.init_state(len(ids), device))[0]                  # the scene and its questions, in one go
                    st = model.hidden(x[:, :L - C], model.init_state(len(ids), device))[1]      # the stores after R tellings
                S = torch.cat([st[i].float().flatten(1) for i in sorted(st)], 1)
                if deep is None:                        # the deepest level asked of each scene
                    deep = [max((lv for *_, lv, _ in s[2] if lv and lv.startswith("depth ")), key=lambda lv: int(lv.split()[-1])) for s in scs]
                ok = []
                for r, s in enumerate(scs):
                    off = L - len(s[0])
                    got = [int(model.scores_at(h, torch.tensor([r], device=device), torch.tensor([off + p], device=device))[0][torch.tensor(cands, device=device)].argmax()) == rt
                           for p, cands, rt, lv, a in s[2] if lv == deep[r]]
                    ok.append(sum(got) / max(1, len(got)))
                right.append(ok)
                if prev is not None:
                    change.append(((S - prev).norm(dim=1) / S.norm(dim=1).clamp(min=1e-9)).tolist())
                prev = S
            for r in range(len(ids)):
                out.setdefault(int(deep[r].split()[-1]), []).append(([c[r] for c in change], [o[r] for o in right]))
    finally:
        model.cfg.chunk = was
        model.train()
    return out


def rule(rows, tellings, bar: float):
    """Stop at the first telling whose change is under the bar (else the last): -> mean tellings used, share right there."""
    used, right = [], []
    for change, ok in rows:
        k = next((j + 1 for j, c in enumerate(change) if c < bar), len(tellings) - 1)
        used.append(tellings[k])
        right.append(ok[k])
    return sum(used) / len(used), 100 * sum(right) / len(right)


def report(name: str, res: dict, tellings, bar: float) -> str:
    show = [r for r in (2, 4, 6, 8, 12) if r in tellings]
    L = [f"# Settle: {name}", "",
         f"How much the stores change from one telling to the next, and the rule \"stop reading when they changed by less than {100 * bar:.0f}%\". "
         f"Exam circuits, each scene on its own; tellings 1 to {tellings[-1]}.", "",
         "| Depth | Scenes | " + " | ".join(f"Change at telling {r}" for r in show) + " | Tellings the rule uses | Right when it stops | Right after "
         + f"{tellings[-1]} |", "|---|---|" + "---|" * (len(show) + 3)]
    for d in sorted(res):
        rows = res[d]
        used, right = rule(rows, tellings, bar)
        mean = lambda r: sum(c[tellings.index(r) - 1] for c, _ in rows) / len(rows)
        last = 100 * sum(ok[-1] for _, ok in rows) / len(rows)
        L.append(f"| {d} | {len(rows)} | " + " | ".join(f"{mean(r):.3f}" for r in show) + f" | {used:.1f} | {right:.0f} | {last:.0f} |")
    L += ["", "How to read it: a model that loops changes its stores less and less as the scene is worked out, later on deeper scenes. If the rule "
          "scores what the last column scores, and uses more tellings on deeper scenes, the model itself says when it has read enough."]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="a think run trained with --align")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--n", type=int, default=240, help="exam scenes (a sixth to each depth)")
    ap.add_argument("--most", type=int, default=12, help="tellings 1 to this many")
    ap.add_argument("--bar", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    from oazy import tok
    enc = tok.get("smollm2")
    out = Path(args.runs or runs_dir()) / args.name
    sd = torch.load(out / "feed" / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = KeepConfig.from_dict(sd["config"])
    if cfg.max_len != T.ALIGN_ROOM * cfg.chunk:
        raise SystemExit(f"{args.name} was not trained with --align (a telling to a chunk): the stores after a telling are not marked off")
    model = Keep(cfg)
    model.load_state_dict(sd["model"])
    model = model.to(device)
    tellings = list(range(1, args.most + 1))
    res = settle(model, enc, args.n, tellings, device)
    text = report(args.name, res, tellings, args.bar)
    (out / "settle.md").write_text(text + "\n")
    (out / "settle.json").write_text(json.dumps({"bar": args.bar, "tellings": tellings, "scenes": {str(d): rows for d, rows in res.items()}}))
    print(text, flush=True)
    print(f"written: {out / 'settle.md'}", flush=True)


if __name__ == "__main__":
    main()
