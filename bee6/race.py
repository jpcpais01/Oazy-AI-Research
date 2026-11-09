"""The Bee6 race's table: Bee5 and Bee6 at equal tokens, and Bee6 given Bee5's training time. Only reads the run folders.

    python3 -m bee6.race                          # bee5-race, bee6-race, bee6-hours (what runpod/bee6_race.sh trains)
    python3 -m bee6.race --bee6 bee6bloom         # the bloom race: bee6bloom-race, bee6bloom-hours

Lower held-out loss is better (per GPT-2 token, on the same 1,024-word windows for every model); higher HellaSwag
(per letter) and LogRes are better. A 500M-token race is a quick filter: a gap under ~0.02 in held-out loss is a tie
to confirm at ~2B tokens.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from oazy import term
from oazy.data import default_dir, run_dir
from oazy.report import scores, pct

TIE = 0.02


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--bee5", default="bee5-race")
    ap.add_argument("--bee6", default="bee6", help="the Bee6 runs' prefix: <prefix>-race and <prefix>-hours")
    a = ap.parse_args(argv)
    runs = Path(a.runs or default_dir("runs"))
    names = [(a.bee5, "Bee5", "500M tokens"), (f"{a.bee6}-race", "Bee6", "the same tokens"),
             (f"{a.bee6}-hours", "Bee6", "Bee5's hours")]
    got = {}
    body = []
    for rn, who, how in names:
        d = run_dir(runs, rn)
        if not (d / "metrics.jsonl").exists():
            body.append([f"{who} ({how})", rn, "not run yet", "", "", "", "", ""])
            continue
        s = scores(d)
        st = s["st"]
        tok, sec = st.get("tokens") or 0, st.get("elapsed") or 0
        s["tok"], s["sec"] = tok, sec
        got[rn] = s
        body.append([term.c(f"{who} ({how})", "bold"), rn, f"{tok / 1e6:,.0f}M", term.duration(sec),
                     term.rate(tok / max(sec, 1e-9)), "–" if s["val"] is None else f"{s['val']:.3f}",
                     pct(s["hs"]) + ("" if s["hs_full"] or s["hs"] is None else "*"), pct(s["logres"])])
    term.header("Bee6 vs Bee5", "0.5B-token race on Sado · next-word models")
    term.table(["", "run", "tokens", "training time", "speed", "held-out loss", "HellaSwag", "LogRes"], body,
               right=(2, 3, 4, 5, 6, 7))
    term.info("held-out loss: lower is better. * HellaSwag on part of the questions. "
              f"At 0.5B tokens a held-out gap under {TIE} is a tie (confirm at ~2B).")

    b5, eq, hr = got.get(a.bee5), got.get(f"{a.bee6}-race"), got.get(f"{a.bee6}-hours")
    lines = []

    def verdict(x, y):
        if x["val"] is None or y["val"] is None:
            return "no held-out loss yet"
        d = x["val"] - y["val"]
        word = "better" if d < -TIE else "worse" if d > TIE else "a tie"
        return f"{word} ({x['val']:.3f} vs {y['val']:.3f}, {d:+.3f})"
    if b5 and eq:
        lines.append(("equal tokens", f"Bee6 is {verdict(eq, b5)}, and {b5['sec'] / max(eq['sec'], 1e-9):.2f}x faster "
                                      f"({term.rate(eq['tok'] / max(eq['sec'], 1e-9))} vs {term.rate(b5['tok'] / max(b5['sec'], 1e-9))})"))
    if b5 and hr:
        lines.append(("equal hours", f"in Bee5's {term.duration(b5['sec'])} Bee6 read {hr['tok'] / max(b5['tok'], 1):.2f}x the tokens "
                                     f"and is {verdict(hr, b5)}"))
    if lines:
        term.header("verdict", color="green")
        term.rows(lines)
    return got


if __name__ == "__main__":
    main()
