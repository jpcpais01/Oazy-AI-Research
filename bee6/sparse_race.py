"""The sparse-MLP race's table: Bee6 against Bee6 with the sparse MLP (bee6/sparse.py), at equal tokens and equal
hours. Only reads the run folders (runpod/bee6_sparse.sh trains them).

    python3 -m bee6.sparse_race                                    # bee6-race, bee6sparse-race, bee6sparse-hours
    python3 -m bee6.sparse_race --base bee6-race --arms bee6sparse-race bee6sparse-hours bee6sparse-nomod-race

Every run is a NEXT-WORD model with Bee6's body and recipe. Held-out loss (Sado, per GPT-2 token): lower is better.
HellaSwag (per letter) and LogT: higher is better. A 500M-token race is a quick filter: a held-out gap under 0.02, or a
HellaSwag gap under ~1 point, is a tie to confirm at ~2B tokens. "Equal hours" only means something on the same GPU:
the table says so when the GPUs differ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from oazy import term
from oazy.data import default_dir, run_dir
from oazy.report import scores, pct
from .world_race import TIE, HS_TIE, SAME, _plain

SKIP = ("fp8", "sparse_mlp", "sparse_block", "sparse_rank", "sparse_from", "sparse_aux", "sparse_check", "sparse_capacity",
        "sparse_balance", "mod_layers")


def differences(a: dict, b: dict):
    """Settings two arms should share but do not (the sparse switch and Mixture of Depths aside)."""
    out = [k for k in SAME if k != "total_tokens" and a.get(k) != b.get(k)]
    ca, cb = a.get("config") or {}, b.get("config") or {}
    out += [f"config.{k}" for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k) and k not in SKIP]
    return out


def last_train(m, key):
    v = [x[key] for x in m if x.get("kind") == "train" and x.get(key) is not None]
    return v[-1] if v else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--base", default="bee6-race", help="plain Bee6 at 500M tokens")
    ap.add_argument("--arms", nargs="+", default=["bee6sparse-race", "bee6sparse-hours"])
    ap.add_argument("--out", default="bee6-sparse-race.md", help="the markdown copy's name, in the runs folder")
    a = ap.parse_args(argv)
    runs = Path(a.runs or default_dir("runs"))
    got, body = {}, []
    for rn in [a.base] + a.arms:
        d = run_dir(runs, rn)
        if not (d / "metrics.jsonl").exists():
            body.append([rn, "not run yet"] + [""] * 8)
            continue
        s = scores(d)
        c = s["cf"].get("config") or {}
        tok, sec = s["st"].get("tokens") or 0, s["st"].get("elapsed") or 0
        s.update(tok=tok, sec=sec)
        got[rn] = s
        sp = c.get("sparse_mlp") or 0
        what = (f"sparse {sp:.0%}" if sp else "dense") + ("" if c.get("mod_layers", "x") else ", no MoD")
        rec = last_train(s["m"], "sparse_recall")
        body.append([term.c(rn, "bold"), what, term.tokens(tok), term.duration(sec), term.rate(tok / max(sec, 1e-9)),
                     "–" if s["val"] is None else f"{s['val']:.3f}",
                     pct(s["hs"]) + ("" if s["hs_full"] or s["hs"] is None else "*"), pct(s["logres"]), pct(rec)])
    head = ["", "MLP", "tokens", "training time", "speed", "held-out Sado", "HellaSwag", "LogT", "picker recall"]
    term.header("Bee6 with the sparse MLP", "0.5B-token race on Sado · next-word models")
    term.table(head, body, right=(2, 3, 4, 5, 6, 7, 8))
    term.info("held-out Sado: lower is better. * HellaSwag on part of the questions. picker recall: the share of each "
              "word's truly most useful blocks the picker ran, at the end (chance = the sparse share)")

    base = got.get(a.base)
    lines, md = [], []
    if base:
        for rn in a.arms:
            x = got.get(rn)
            if not x:
                continue
            diff = differences(base["cf"], x["cf"])
            if base["cf"].get("gpu") != x["cf"].get("gpu"):
                diff.append(f"GPU ({base['cf'].get('gpu')} vs {x['cf'].get('gpu')}: equal hours means nothing)")
            parts = []
            if base["val"] is not None and x["val"] is not None:
                d = x["val"] - base["val"]
                parts.append(("better" if d < -TIE else "worse" if d > TIE else "a tie") + f" on held-out Sado ({d:+.3f})")
            if base["hs"] is not None and x["hs"] is not None:
                d = x["hs"] - base["hs"]
                parts.append(("better" if d > HS_TIE else "worse" if d < -HS_TIE else "a tie") + f" on HellaSwag ({d * 100:+.1f} pts)")
            if base["logres"] is not None and x["logres"] is not None:
                parts.append(f"LogT {(x['logres'] - base['logres']) * 100:+.1f} pts")
            if base["sec"] and x["sec"]:
                parts.append(f"{(x['tok'] / x['sec']) / (base['tok'] / base['sec']):.2f}x the speed")
            if diff:
                lines.append((f"{rn} settings", term.c("differ from " + a.base + ": " + ", ".join(diff[:6]), "red")))
            lines.append((rn, ", ".join(parts) or "no numbers yet"))
            md.append(f"- **{rn}** vs {a.base}: " + (", ".join(parts) or "no numbers yet")
                      + (f" (settings differ: {', '.join(diff)})" if diff else ""))
    if lines:
        term.header("verdict", color="green")
        term.rows(lines)
    text = ["# Bee6 with the sparse MLP\n", "0.5B-token race on Sado, next-word models, Bee6's body and recipe.\n",
            "| " + " | ".join(h or "run" for h in head) + " |", "|" + "---|" * len(head)]
    text += ["| " + " | ".join(_plain(c) for c in row) + " |" for row in body]
    if md:
        text += ["", "## Verdict", ""] + md
    out = runs / a.out
    try:
        out.write_text("\n".join(text) + "\n")
        term.info(f"table also saved: {out}")
    except OSError:
        pass
    return got


if __name__ == "__main__":
    main()
