"""Canvas against Bee-135: does a model that thinks in meanings (the near future all at once, words last) reason better
than the same size of plain next-word transformer, on the same text?

    plain   judgeab-plain-135   Bee-135 (lean + cuDNN attention), the next-word lesson alone. The SAME run as the judge
                                A/B's plain arm (same text, tokens, seed): trained once, read by both A/Bs
    canvas  canvasab-canvas-135 Canvas-135 (canvas/model.py): ear and mouth see 64 pieces back, a mind over 16-piece
                                chunk meanings, a canvas of the next 4 chunks' meanings refined together in 4 steps

Both read the same text (Opal3, Field1 off) for the same number of tokens, the same seed, no mentor. The weights are
about equal (Bee-135 ~132M, Canvas-135 within 3%); the canvas does about half the work per word (12 word layers to
Bee-135's 24; its mind and canvas run once per chunk of 16 pieces).

    bash runpod/canvasab.sh                 # trains plain (if the judge A/B has not), then canvas, then prints the table
    python3 -m canvas.ab                    # the table alone (reads the run folders)

Rule fixed before the run (21 Nov 2026), on the full tests at the end, at equal tokens. João: "natural language looks a
bit worse, but as long as it is more intelligent, it's okay". So the canvas is KEPT only if all three hold:
    1. reasoning: the mean of ARC-Easy, ARC-Challenge, CommonsenseQA and Winogrande is 1.5 points or more above plain's
    2. language: HellaSwag is no more than 3.0 points under plain's
    3. the thought is used: the held-out loss with the canvas switched off is 0.05 or more above it with the canvas on
Each arm gets the better of its two usual scorings (per token, per letter) on each test.

Also shown, not part of the rule: the held-out loss by refining steps (1, 2, 4, 8): if more steps give a lower loss,
thinking longer helps, which a plain transformer cannot do at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnosis2.judgeab import ARMS as JARMS, RUNS as JRUNS, TESTS, read as jread

ARMS = {"plain": JARMS["plain"], "canvas": {}}
MODEL = {"plain": "gnosis2", "canvas": "canvas"}
SEED = {"plain": 0, "canvas": 0}
RUNS = {"plain": JRUNS["plain"], "canvas": "canvasab-canvas-135"}
REASON = ("arc_easy", "arc_challenge", "csqa", "winogrande")
NEED_REASON, HS_SLACK, NEED_GAP = 0.015, 0.03, 0.05
TUNE = "compile_mode=max-autotune-no-cudagraphs"


def set_words(arm, tune=False):
    return [f"{k}={v}" for k, v in ARMS[arm].items()] + ([TUNE] if tune else [])


def _mean(x, keys):
    v = [x.get(k) for k in keys]
    return None if any(a is None for a in v) else sum(v) / len(v)


def extras(d):
    """The canvas run's own checks: (the last canvas-off gap, {steps: held-out loss}) or (None, None)."""
    f = Path(d) / "metrics.jsonl"
    if not f.exists():
        return None, None
    core = loop = None
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") == "core":
            core = r.get("core_gap")
        elif r.get("kind") == "loop":
            loop = r.get("pass_loss")
    return core, loop


def verdict(got, gap):
    P, C = got.get("plain"), got.get("canvas")
    if not (P and C):
        return [("verdict", "none yet: both arms must have run")]
    if not (P["done"] and C["done"] and P["full"] and C["full"]) or abs(P["tok"] - C["tok"]) > 0.01 * P["tok"]:
        return [("verdict", f"none yet: plain is at {P['tok'] / 1e6:,.0f}M tokens, canvas at {C['tok'] / 1e6:,.0f}M "
                            f"(both must finish, with the full tests at the end)")]
    rp, rc = _mean(P["word"], REASON), _mean(C["word"], REASON)
    hp, hc = P["word"].get("hs"), C["word"].get("hs")
    if None in (rp, rc, hp, hc):
        return [("verdict", "none: a test was not scored (python3 -m oazy.evals --prepare, then score the runs again)")]
    ok_r, ok_h = rc - rp >= NEED_REASON - 1e-9, hc - hp >= -HS_SLACK - 1e-9
    ok_g = gap is not None and gap >= NEED_GAP - 1e-9
    lines = [("reasoning", f"{'passes' if ok_r else 'fails'}: canvas {rc * 100:.1f}% against plain {rp * 100:.1f}% = "
                           f"{(rc - rp) * 100:+.1f} points (ARC-E, ARC-C, CSQA, Winogrande; rule: +{NEED_REASON * 100:.1f} or more)"),
             ("language", f"{'passes' if ok_h else 'fails'}: HellaSwag canvas {hc * 100:.1f}% against plain {hp * 100:.1f}% = "
                          f"{(hc - hp) * 100:+.1f} points (rule: no more than {HS_SLACK * 100:.1f} under)"),
             ("thought used", f"{'passes' if ok_g else 'fails'}: held-out loss {gap:+.3f} higher with the canvas off (rule: +{NEED_GAP:.2f} or more)"
              if gap is not None else "fails: the canvas-off check was not logged"),
             ("verdict", "KEEP the canvas: all three parts of the rule hold" if ok_r and ok_h and ok_g else
                         "DROP the canvas: the rule fixed before the run does not hold")]
    lines.append(("held-out loss", f"canvas {C['val'] - P['val']:+.3f} against plain"))
    sp, sc = P["tok"] / max(P["sec"], 1e-9), C["tok"] / max(C["sec"], 1e-9)
    lines.append(("speed", f"the canvas ran {sc / sp:.2f}x the plain arm's speed over the whole run (tests and compile included)"))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--set", choices=list(ARMS), default=None)
    ap.add_argument("--seed", choices=list(ARMS), default=None, help="print an arm's seed")
    ap.add_argument("--name", choices=list(ARMS), default=None, help="print an arm's run name")
    ap.add_argument("--model", choices=list(ARMS), default=None, help="print an arm's --model")
    ap.add_argument("--tune", action="store_true", help="with --set: add torch's kernel tuning")
    a = ap.parse_args(argv)
    if a.set or a.seed or a.name or a.model:
        print(" ".join(set_words(a.set, a.tune)) if a.set else SEED[a.seed] if a.seed else
              RUNS[a.name] if a.name else MODEL[a.model])
        return
    from oazy import term
    from oazy.data import default_dir, run_dir
    from oazy.report import pct
    runs = Path(a.runs or default_dir("runs"))
    got = {arm: jread(run_dir(runs, rn)) for arm, rn in RUNS.items()}
    gap, loop = extras(run_dir(runs, RUNS["canvas"]))
    body = []
    for arm, label in (("plain", "Bee-135 (plain)"), ("canvas", "Canvas-135")):
        g = got[arm]
        if g is None:
            body.append([label, "not run yet"] + [""] * (3 + len(TESTS)))
            continue
        star = "" if g["full"] else "*"
        body.append([term.c(label, "bold"), f"{g['tok'] / 1e9:.2f}B" + ("" if g["done"] else " (running)"),
                     term.duration(g["sec"]), term.rate(g["tok"] / max(g["sec"], 1e-9)), f"{g['val']:.3f}"]
                    + [pct(g["word"][k]) + (star if g["word"][k] is not None else "") for k, _ in TESTS])
    term.header("Canvas against Bee-135: the A/B", "Opal3, Field1 off · equal tokens · the same seed · no mentor")
    term.table(["", "tokens", "training time", "speed", "held-out loss"] + [t for _, t in TESTS], body,
               right=tuple(range(1, 5 + len(TESTS))))
    if any(g is not None and not g["full"] for g in got.values()):
        print("  * a quick check during the run, not the full test at the end")
    if loop:
        print("  thinking longer (held-out loss by refining steps): "
              + "  ".join(f"{k}: {float(v):.3f}" for k, v in sorted(loop.items(), key=lambda kv: int(kv[0]))))
    term.header("verdict by the rule fixed before the run", color="green")
    term.rows(verdict(got, gap))
    return got


if __name__ == "__main__":
    main()
