"""The world-mix race's table (runpod/world_mix_race.sh): plain dense Bee6 on Emerald with different shares of world
text over the run, with and without the SmolLM2-360M teacher. Only reads the run folders.

    python3 -m bee6.mix_race                      # the five 2B arms
    python3 -m bee6.mix_race --arms A B C ...     # any runs

All NEXT-WORD models, same body, same tokens (a run's tokens include its world text, so a bigger world share reads less
Sado). Held-out losses are per GPT-2 token, lower is better: overall (Sado's held-out text), dclm, wiki, and the world's
own held-out practice text. Tests: higher is better. tok/s = the run's average (the teacher costs reading time).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oazy import term
from oazy.data import default_dir, run_dir
from oazy.report import scores, pct

ARMS = ["bee6-mix-std-teach-2b", "bee6-mix-std-2b", "bee6-mix-flat10-2b", "bee6-mix-flat20-2b", "bee6-mix-30to15-2b"]
LABEL = {"bee6-mix-std-teach-2b": "A 30%->5% + teacher", "bee6-mix-std-2b": "B 30%->5%", "bee6-mix-flat10-2b": "C 10% flat",
         "bee6-mix-flat20-2b": "D 20% flat", "bee6-mix-30to15-2b": "E 30%->15%"}


def _js(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _plain(x):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", str(x))


def _loss(v):
    return "–" if v is None else f"{v:.3f}"


def row(runs: Path, rn: str):
    d = run_dir(runs, rn)
    if not (d / "metrics.jsonl").exists():
        return None
    s = scores(d)
    e, ex, st = s["eval"], _js(d / "world_exam.json"), s["st"]
    tok, sec = st.get("tokens") or 0, st.get("elapsed") or 0
    teach = s["cf"].get("mentor") not in (None, "none")
    plan = s["cf"].get("emerald_plan") or "30% until 20%, then 5% (10.0% on average)"
    return {"name": rn, "teach": teach, "plan": plan, "val": s["val"], "dclm": e.get("val_loss_dclm"),
            "wiki": e.get("val_loss_wiki"), "world": e.get("val_loss_world"), "hs": s["hs"], "hs_full": s["hs_full"],
            "arc": s["arc"], "piqa": s["piqa"], "logt": s["logres"], "secret": ex.get("secret"), "seen": ex.get("seen"),
            "real": ex.get("real"), "pro": ex.get("pro"), "tps": tok / sec if sec else None, "tok": tok}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, help="the first is the one the others are measured against")
    ap.add_argument("--out", default="world-mix-race-2b.md", help="the markdown copy's name, in the runs folder")
    a = ap.parse_args(argv)
    runs = Path(a.runs or default_dir("runs"))
    head = ["", "teacher", "world over the run", "held-out", "dclm", "wiki", "world text", "HellaSwag", "ARC-E", "PIQA",
            "LogT", "world secret", "practised", "Real World", "Pro", "tok/s"]
    got, body = {}, []
    for rn in a.arms:
        r = row(runs, rn)
        name = LABEL.get(rn, rn)
        if r is None:
            body.append([name, "not run yet"] + [""] * (len(head) - 2))
            continue
        got[rn] = r
        body.append([term.c(name, "bold"), "yes" if r["teach"] else "no", r["plan"].split(" (")[0],
                     _loss(r["val"]), _loss(r["dclm"]), _loss(r["wiki"]), _loss(r["world"]),
                     pct(r["hs"]) + ("" if r["hs_full"] or r["hs"] is None else "*"), pct(r["arc"]), pct(r["piqa"]),
                     pct(r["logt"]), pct(r["secret"]), pct(r["seen"]), pct(r["real"]), pct(r["pro"]),
                     "–" if not r["tps"] else f"{r['tps'] / 1e3:,.0f}k"])
    term.header("World-mix race", "plain dense Bee6 · Emerald · next-word models")
    term.table(head, body, right=tuple(range(3, len(head))))
    term.info("losses: lower is better (per GPT-2 token). * HellaSwag on part of the questions. world secret: exams no run "
              "practised; Real World: practised scenes in held-back wording; Pro: new names and things too. "
              "A held-out gap under 0.02 or a HellaSwag gap under ~1 point is a tie.")
    base = got.get(a.arms[0])
    verdict = []
    if base:
        for rn in a.arms[1:]:
            x = got.get(rn)
            if not x:
                continue
            bits = []
            for k, t, scale, low in (("val", "held-out", 1, True), ("world", "world text", 1, True),
                                     ("hs", "HellaSwag", 100, False), ("logt", "LogT", 100, False),
                                     ("secret", "world secret", 100, False), ("real", "Real World", 100, False)):
                if base[k] is not None and x[k] is not None:
                    d = (x[k] - base[k]) * scale
                    bits.append(f"{t} {d:+.3f}" if scale == 1 else f"{t} {d:+.1f} pts")
            verdict.append((LABEL.get(rn, rn), ", ".join(bits) or "no numbers yet"))
        if verdict:
            term.header(f"against {LABEL.get(a.arms[0], a.arms[0])}", color="green")
            term.rows(verdict)
    md = ["# World-mix race\n", "Plain dense Bee6 on Emerald, same tokens each, next-word models. "
          "Losses lower is better; tests higher is better.\n",
          "| " + " | ".join(h or "arm" for h in head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(_plain(c) for c in r) + " |" for r in body]
    if verdict:
        md += ["", f"## Against {LABEL.get(a.arms[0], a.arms[0])}", ""] + [f"- **{n}**: {t}" for n, t in verdict]
    out = runs / a.out
    out.write_text("\n".join(md) + "\n")
    term.info(f"table also saved: {out}")
    return got


if __name__ == "__main__":
    main()
