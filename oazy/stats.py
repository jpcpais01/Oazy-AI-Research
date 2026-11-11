"""A run's history, one row per stretch of tokens: spot a strange drop or jump at a glance.

    python3 -m oazy.stats                         # the newest run, a row every 200M tokens
    python3 -m oazy.stats --run bee6moe-emerald-10b --every 100
    python3 -m oazy.stats --every 50 --from 2000 --max 3200   # zoom into 2,000M-3,200M

Reads <runs>/<run>/metrics.jsonl only (safe while the run trains). Each row covers one stretch of --every
million tokens:

    loss      training loss, averaged over the stretch's log lines (the text, not the world's own loss)
    val       held-out loss, the last check in the stretch
    HS        HellaSwag per letter (quick checks; the full one at the end)
    PIQA, ARC (ARC-Easy) per letter like HS (what SmolLM2 publishes), LogT; last in the stretch
    secret    world exams never practised; prac = practised (only at the end of a run)
    uneven    busiest expert / average (1.00 = perfectly even), averaged; drop = share of words no expert took
    lr        learning rate at the end of the stretch; ctx = context length; world = world text share
    tok/s     training speed, averaged

A column with nothing in it is left out. "·" = no check in that stretch. --csv prints the same rows as CSV.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import term


def runs_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("OAZY_RUNS"):
        return Path(os.environ["OAZY_RUNS"])
    if Path("/workspace/runs").exists():
        return Path("/workspace/runs")
    from .data import default_dir
    return Path(default_dir("runs"))


def find_run(root: Path, name: str | None) -> Path | None:
    """The run's folder: a name under root, a path, or (no name) the run whose metrics changed last."""
    if name:
        for d in (Path(name), root / name):
            if (d / "metrics.jsonl").exists():
                return d
        hits = sorted(root.glob(f"*{name}*/metrics.jsonl"), key=lambda p: p.stat().st_mtime)
        return hits[-1].parent if hits else None
    hits = sorted(root.glob("*/metrics.jsonl"), key=lambda p: p.stat().st_mtime)
    return hits[-1].parent if hits else None


def read(path: Path):
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:          # a half-written last line while the run trains
            continue
        if isinstance(r, dict) and isinstance(r.get("tokens"), (int, float)):
            yield r


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None


# (column, how to fill it): "mean" averages every value in the stretch, "last" keeps the last one
COLUMNS = ["loss", "val", "HS", "PIQA", "ARC", "LogT", "secret", "prac", "uneven", "drop", "lr", "ctx", "world", "tok/s"]
MEAN = {"loss", "uneven", "drop", "tok/s"}


def values(r: dict) -> dict:
    """The columns one metrics line fills."""
    k, out = r.get("kind"), {}
    if k == "train":
        out["loss"] = _num(r.get("loss"))
        out["lr"] = _num(r.get("lr"))
        out["tok/s"] = _num(r.get("tok_s"))
        out["uneven"] = _num(r.get("experts_uneven"))
        out["drop"] = _num(r.get("experts_dropped"))
        out["ctx"] = _num(r.get("context"))
        w = _num(r.get("world_share"))
        if w is None and isinstance(r.get("mix"), dict):     # Emerald: the world is one of the mix's sources
            w = _num(r["mix"].get("world", r["mix"].get("field1")))
        out["world"] = w
    elif k == "eval":
        out["val"] = _num(r.get("val_loss"))
    elif k == "hellaswag":
        out["HS"] = _num(r.get("acc_char"))
    elif k == "evals":
        # per letter, like HS and like SmolLM2's published numbers (lm-eval acc_norm); per token if a line lacks it
        out["PIQA"] = _num(r.get("piqa_char", r.get("piqa")))
        out["ARC"] = _num(r.get("arc_easy_char", r.get("arc_easy")))
    elif k == "logres":
        out["LogT"] = _num(r.get("acc_char"))
    elif k == "world":
        out["secret"], out["prac"] = _num(r.get("secret")), _num(r.get("seen"))
    elif k == "context":
        out["ctx"] = _num(r.get("context"))
    return {c: v for c, v in out.items() if v is not None}


def buckets(recs, every_m: float, lo_m: float, hi_m: float | None):
    """{stretch start in M tokens: {column: value}}."""
    acc: dict[int, dict] = {}
    for r in recs:
        t = r["tokens"] / 1e6
        if t < lo_m or (hi_m is not None and t > hi_m):
            continue
        b = acc.setdefault(int(t // every_m), {})
        for col, v in values(r).items():
            if col in MEAN:
                s = b.setdefault(col, [0.0, 0])
                s[0] += v
                s[1] += 1
            else:
                b[col] = v
    out = {}
    for i in sorted(acc):
        out[i * every_m] = {c: (v[0] / v[1] if isinstance(v, list) else v) for c, v in acc[i].items()}
    return out


def fmt(col: str, v) -> str:
    if v is None:
        return term.c("·", "grey")
    if col in ("loss", "val"):
        return f"{v:.3f}"
    if col in ("HS", "PIQA", "ARC", "LogT", "secret", "prac", "world"):
        return f"{v * 100:.1f}"
    if col == "drop":
        return f"{v * 100:.2f}%"
    if col == "uneven":
        return f"{v:.2f}"
    if col == "lr":
        return f"{v:.2e}"
    if col == "ctx":
        return f"{int(v)}"
    if col == "tok/s":
        return f"{v / 1e3:.0f}k"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description="A run's history, one row per stretch of tokens.")
    ap.add_argument("--run", help="run name (or part of it, or a folder); default: the newest run")
    ap.add_argument("--runs", help="runs folder (default $OAZY_RUNS, else /workspace/runs)")
    ap.add_argument("--every", type=float, default=200, help="million tokens per row (default 200)")
    ap.add_argument("--from", dest="lo", type=float, default=0, help="start at this many million tokens")
    ap.add_argument("--max", type=float, default=None, help="stop at this many million tokens")
    ap.add_argument("--csv", action="store_true", help="print CSV instead of a table")
    a = ap.parse_args(argv)

    root = runs_root(a.runs)
    d = find_run(root, a.run)
    if d is None:
        term.say(f"[error] no run {'matching ' + repr(a.run) + ' ' if a.run else ''}with metrics.jsonl in {root}")
        raise SystemExit(1)
    recs = list(read(d / "metrics.jsonl"))
    rows = buckets(recs, max(a.every, 1e-6), a.lo, a.max)
    cols = [c for c in COLUMNS if any(c in v for v in rows.values())]

    if a.csv:
        term.out(",".join(["tokens_M"] + cols))
        for t, v in rows.items():
            term.out(",".join([f"{t:g}"] + ["" if v.get(c) is None else f"{v[c]:.6g}" for c in cols]))
        return

    last = max((r["tokens"] for r in recs), default=0)
    term.header(d.name, f"{term.tokens(last)} tokens · a row every {a.every:g}M")
    body = [[f"{t:,.0f}M"] + [fmt(c, v.get(c)) for c in cols] for t, v in rows.items()]
    term.table(["tokens"] + cols, body, right=tuple(range(len(cols) + 1)))
    if not body:
        term.out(term.c("  nothing logged in that range yet", "dim"))


if __name__ == "__main__":
    main()
