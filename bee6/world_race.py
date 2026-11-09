"""The world race's table: plain Bee6 against Bee6 with world practice (oazy/worldmix.py), side by side. Only reads the
run folders (runpod/bee6_world.sh trains them and runs the world exams first).

    python3 -m bee6.world_race                                         # bee6-race vs bee6-world10
    python3 -m bee6.world_race --arms bee6-race bee6-world10 bee6-world5

Every run is a NEXT-WORD model with the same body, recipe and token count; a world run's tokens include its episodes,
so it read less Sado text at the same compute. Held-out loss is on Sado's held-out text only (no world text in it):
lower is better. HellaSwag (per letter), LogT and the world exams: higher is better. A 500M-token race is a quick
filter: a held-out gap under 0.02, or a HellaSwag gap under ~1 point, is a tie to confirm at ~2B tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oazy import term
from oazy.data import default_dir, run_dir
from oazy.report import scores, pct

TIE, HS_TIE = 0.02, 0.01
SAME = ("preset", "total_tokens", "batch_tokens", "batch_ramp", "lr", "muon_lr", "decay", "decay_shape", "mentor",
        "mentor_weight", "mentor_until", "mentor_floor", "mentor_off", "mix", "tokenizer")
SPLIT = {"longer": "longer", "deeper": "deeper", "bigger": "bigger falls", "new-pairs": "links chained in new ways",
         "new-mixes": "rules mixed in new ways", "more-moves": "more moves in a row"}


def _js(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _plain(x):
    """A cell without its terminal colours (for the markdown copy)."""
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", str(x))


def differences(a: dict, b: dict):
    """Settings two arms should share but do not (the world switch itself aside)."""
    out = [k for k in SAME if a.get(k) != b.get(k) and not (k == "mix" and "world" in str(b.get(k)))]   # Emerald's mix has the world in it
    ca, cb = a.get("config") or {}, b.get("config") or {}
    out += [f"config.{k}" for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k) and k != "fp8"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--arms", nargs="+", default=["bee6-race", "bee6-world10"], help="the plain run first")
    ap.add_argument("--out", default="bee6-world-race.md", help="the markdown copy's name, in the runs folder")
    a = ap.parse_args(argv)
    runs = Path(a.runs or default_dir("runs"))
    got, body = {}, []
    for rn in a.arms:
        d = run_dir(runs, rn)
        if not (d / "metrics.jsonl").exists():
            body.append([rn, "not run yet"] + [""] * 7)
            continue
        s = scores(d)
        w = s["cf"].get("world") or 0.0
        ex = _js(d / "world_exam.json")
        s.update(w=w, ex=ex)
        m_tr = [x for x in s["m"] if x.get("kind") == "train" and "world_tokens" in x]
        wt = m_tr[-1]["world_tokens"] if m_tr else 0
        tok = s["st"].get("tokens") or 0
        s.update(tok=tok, wt=wt, sec=s["st"].get("elapsed") or 0)
        got[rn] = s
        body.append([term.c(rn, "bold"), f"{w:.0%}" if w else "none", f"{term.tokens(tok - wt)} + {term.tokens(wt)}",
                     term.duration(s["sec"]), "–" if s["val"] is None else f"{s['val']:.3f}",
                     pct(s["hs"]) + ("" if s["hs_full"] or s["hs"] is None else "*"), pct(s["logres"]),
                     pct(ex.get("secret")), pct(ex.get("seen"))])
    term.header("Bee6 with world practice", "0.5B tokens · Sado · next-word models")
    head = ["", "world", "text + world tokens", "training time", "held-out Sado", "HellaSwag", "LogT", "world secret",
            "world practised"]
    term.table(head, body, right=(2, 3, 4, 5, 6, 7, 8))
    term.info("held-out Sado: lower is better (per GPT-2 token, Sado text only). * HellaSwag on part of the questions. "
              "world secret: exams no run ever practised (longer, deeper, new mixes), asked as plain text.")

    base = got.get(a.arms[0])
    lines, md_verdict = [], []
    if base:
        for rn in a.arms[1:]:
            x = got.get(rn)
            if not x:
                continue
            diff = differences(base["cf"], x["cf"])
            if diff:
                lines.append((f"{rn} settings", term.c("differ from " + a.arms[0] + ": " + ", ".join(diff[:6]), "red")))
            parts = []
            if base["val"] is not None and x["val"] is not None:
                d = x["val"] - base["val"]
                parts.append(("better" if d < -TIE else "worse" if d > TIE else "a tie") + f" on held-out Sado ({d:+.3f})")
            if base["hs"] is not None and x["hs"] is not None:
                d = x["hs"] - base["hs"]
                parts.append(("better" if d > HS_TIE else "worse" if d < -HS_TIE else "a tie") + f" on HellaSwag ({d * 100:+.1f} pts)")
            if base["logres"] is not None and x["logres"] is not None:
                parts.append(f"LogT {(x['logres'] - base['logres']) * 100:+.1f} pts")
            if base["ex"].get("secret") is not None and x["ex"].get("secret") is not None:
                parts.append(f"world secret {(x['ex']['secret'] - base['ex']['secret']) * 100:+.1f} pts")
            lines.append((rn, ", ".join(parts) or "no numbers yet"))
            md_verdict.append(f"- **{rn}** vs {a.arms[0]}: " + (", ".join(parts) or "no numbers yet")
                              + (f" (settings differ: {', '.join(diff)})" if diff else ""))
    if lines:
        term.header("verdict", color="green")
        term.rows(lines)

    # the world exams family by family (secret only), every arm side by side
    have = [rn for rn in a.arms if rn in got and got[rn]["ex"].get("rows")]
    fam_rows = []
    if have:
        keys = []
        for r in got[have[0]]["ex"]["rows"]:
            k = (r["family"], r["split"])
            if r["split"] != "seen" and k not in keys:
                keys.append(k)
        for fam, sp in keys:
            row = [fam, SPLIT.get(sp, sp)]
            for rn in have:
                rs = [r["right"] for r in got[rn]["ex"]["rows"] if r["family"] == fam and r["split"] == sp]
                row.append(pct(sum(rs) / len(rs)) if rs else "–")
            fam_rows.append(row)
        term.header("world secret exams, family by family")
        term.table(["family", "exam"] + have, fam_rows, right=tuple(range(2, 2 + len(have))))

    md = ["# Bee6 with world practice\n", "0.5B tokens of Sado each, next-word models, same body and recipe. "
          "A world run's tokens include its episodes.\n", "| " + " | ".join(h or "run" for h in head) + " |",
          "|" + "---|" * len(head)]
    for row in body:
        md.append("| " + " | ".join(_plain(c) for c in row) + " |")
    if md_verdict:
        md += ["", "## Verdict", ""] + md_verdict
    if fam_rows:
        md += ["", "## World secret exams, family by family", "", "| family | exam | " + " | ".join(have) + " |",
               "|" + "---|" * (2 + len(have))] + ["| " + " | ".join(r) + " |" for r in fam_rows]
    out = runs / a.out
    out.write_text("\n".join(md) + "\n")
    term.info(f"table also saved: {out}")
    return got


if __name__ == "__main__":
    main()
