"""The race report: Code Black against Black-T, on the clock.

    python3 -m codeblack.race                          # every Code Black and Black-T run in the runs folder
    python3 -m codeblack.race blackt-125m black-k4     # these runs only
    python3 -m codeblack.race --ref blackt-125m        # the yardstick (default: the smallest Black-T run)
    python3 -m codeblack.race --png race.png           # also draw held-out loss against minutes (needs matplotlib)

Every race run trains for the same minutes (--preset race --minutes 30), so a faster model simply reads more text.
For each run it prints the held-out loss after 10, 20 and 30 minutes of training, the final loss and HellaSwag, and
how many minutes it took to reach the yardstick's final loss: the speed-up is the yardstick's minutes divided by that.

The plan's pass line for rung R2 (codeblack/README.md): Code Black reaches Black-T 125M's 30-minute loss in 20 minutes
or less (a speed-up of at least 1.5x), with HellaSwag no worse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oazy.data import default_dir

KINDS = ("black", "blackt")


def load(folder: Path):
    cfg = json.loads((folder / "config.json").read_text())
    recs = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line.strip()]
    evals = sorted(((r["elapsed"] / 60, r["val_loss"], r["tokens"]) for r in recs if r["kind"] == "eval"), key=lambda e: e[0])
    hs = [r for r in recs if r["kind"] == "hellaswag"]
    full = [r for r in hs if r.get("full")]
    train = [r for r in recs if r["kind"] == "train"]
    return {"name": folder.name, "model": cfg.get("model"), "params": cfg.get("params"), "layout": cfg.get("layout"),
            "minutes": cfg.get("minutes"), "gpu": cfg.get("gpu"), "evals": evals,
            "hellaswag": (full or hs or [{}])[-1].get("acc_norm"), "hellaswag_full": bool(full),
            "tokens": train[-1]["tokens"] if train else 0, "elapsed": train[-1]["elapsed"] / 60 if train else 0}


def loss_at(evals, minute):
    """Held-out loss after `minute` minutes of training (straight line between the two evaluations around it)."""
    if not evals or minute > evals[-1][0] + 1e-6:
        return None
    prev = evals[0]
    for e in evals:
        if e[0] >= minute:
            if e[0] == prev[0]:
                return e[1]
            f = (minute - prev[0]) / (e[0] - prev[0])
            return prev[1] + f * (e[1] - prev[1])
        prev = e
    return evals[-1][1]


def minutes_to(evals, target):
    """Training minutes until the held-out loss first reaches `target` (interpolated), or None."""
    prev = None
    for e in evals:
        if e[1] <= target:
            if prev is None or prev[1] == e[1]:
                return e[0]
            f = (prev[1] - target) / (prev[1] - e[1])
            return prev[0] + f * (e[0] - prev[0])
        prev = e
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run folder names (default: every Code Black and Black-T run)")
    ap.add_argument("--dir", default=None, help="the runs folder (default: the trainer's)")
    ap.add_argument("--ref", default=None, help="the yardstick run (default: the smallest Black-T)")
    ap.add_argument("--png", default=None, help="draw held-out loss against minutes into this file")
    a = ap.parse_args(argv)
    root = Path(a.dir or default_dir("runs"))
    folders = [root / n for n in a.runs] if a.runs else sorted(p.parent for p in root.glob("*/config.json"))
    runs = []
    for f in folders:
        try:
            r = load(f)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            if a.runs:
                raise SystemExit(f"cannot read {f}: {e}")
            continue
        if r["model"] in KINDS and r["evals"]:
            runs.append(r)
    if not runs:
        raise SystemExit(f"no Code Black or Black-T runs with evaluations in {root}")
    refs = [r for r in runs if r["name"] == a.ref] if a.ref else sorted((r for r in runs if r["model"] == "blackt"),
                                                                          key=lambda r: r["params"] or 0)[:1]
    ref = refs[0] if refs else None
    target = ref["evals"][-1][1] if ref else None
    ref_min = ref["evals"][-1][0] if ref else None
    print(f"{'run':18s} {'model':10s} {'params':>7s} {'min':>5s} {'tokens':>8s} {'tok/s':>7s} {'@10':>6s} {'@20':>6s} "
          f"{'@30':>6s} {'final':>6s} {'HellaS.':>7s} {'to ref':>7s} {'speed-up':>8s}")
    out = []
    for r in sorted(runs, key=lambda r: (r["model"] != "blackt", r["params"] or 0)):
        ev = r["evals"]
        at = [loss_at(ev, m) for m in (10, 20, 30)]
        mt = minutes_to(ev, target) if target is not None else None
        sp = ref_min / mt if mt else None
        tps = r["tokens"] / max(r["elapsed"] * 60, 1e-9)
        f = lambda v, d=3: f"{v:.{d}f}" if v is not None else "-"   # noqa: E731
        hs = f"{r['hellaswag'] * 100:.1f}%" + ("" if r["hellaswag_full"] else "*") if r["hellaswag"] is not None else "-"
        print(f"{r['name'][:18]:18s} {('Code Black' if r['model'] == 'black' else 'Black-T'):10s} {(r['params'] or 0) / 1e6:6.0f}M "
              f"{ev[-1][0]:5.1f} {r['tokens'] / 1e6:7.0f}M {tps / 1e3:6.0f}k {f(at[0]):>6s} {f(at[1]):>6s} {f(at[2]):>6s} "
              f"{f(ev[-1][1]):>6s} {hs:>7s} {f(mt, 1):>7s} {(f'{sp:.2f}x' if sp else '-'):>8s}")
        out.append({**{k: r[k] for k in ("name", "model", "params", "layout", "tokens", "hellaswag")}, "loss_at": at,
                    "final_loss": ev[-1][1], "minutes": ev[-1][0], "minutes_to_ref": mt, "speedup": sp})
    if ref:
        print(f"\nyardstick: {ref['name']}, final held-out loss {target:.3f} after {ref_min:.1f} min. "
              "'to ref' = minutes to reach that loss; speed-up = its minutes / those minutes. * = quick HellaSwag sample.")
        for o in out:
            if o["model"] == "black" and o["minutes_to_ref"] is not None:
                ok = o["speedup"] >= 1.5 and (o["hellaswag"] or 0) >= (ref["hellaswag"] or 0) - 1e-9
                print(f"R2 pass line for {o['name']}: speed-up {o['speedup']:.2f}x (needs 1.5x), HellaSwag "
                      f"{(o['hellaswag'] or 0) * 100:.1f}% vs {(ref['hellaswag'] or 0) * 100:.1f}%: {'PASS' if ok else 'not yet'}")
            elif o["model"] == "black":
                print(f"R2 pass line for {o['name']}: never reached the yardstick's loss: not passed")
    (root / "race.json").write_text(json.dumps({"ref": ref["name"] if ref else None, "runs": out}, indent=1))
    print(f"\nwritten to {root / 'race.json'}")
    if a.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed: python3 -m pip install matplotlib")
            return
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for r in runs:
            xs, ys = [e[0] for e in r["evals"] if e[0] > 0.5], [e[1] for e in r["evals"] if e[0] > 0.5]
            ax.plot(xs, ys, lw=2 if r["model"] == "black" else 1.2, ls="-" if r["model"] == "black" else "--", label=r["name"])
        if target is not None:
            ax.axhline(target, color="grey", lw=0.8)
        ax.set_xlabel("B200 minutes of training")
        ax.set_ylabel("held-out loss (per GPT-2 token)")
        ax.set_title("Code Black vs Black-T, on the clock")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(a.png)
        print(f"drawn to {a.png}")


if __name__ == "__main__":
    main()
