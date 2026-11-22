"""Loop-135 against Bee-135: does a loop that thinks over the words' vectors reason better than the plain transformer
it is built from, at equal weights and equal compute? Both NEXT-TOKEN.

    plain   loopab-plain-135   Bee-135 (lean + cuDNN), 24 layers once each
    loop    loopab-loop-135    the same 24 layers and weights (+ one vector of 576): reader 16 once, loop 4 in rounds
                               (2-6 in training, the last 4 with gradient; 4 when reading), mouth 4 (gnosis2/loop.py)

Equal weights: 132M each. Equal compute: a loop word costs 1.433x a Bee word in training (loop.cost), so the plain arm
reads 1.433x as many words (2B for the loop -> 2.87B for plain). Same text for both: Opal3 (Field1 off) with 5% of the
wordings text (oazy/wordings.py: each rule said in many ways, chains of 1-3 steps). Same seed, no mentor.

    bash runpod/loopab.sh        # builds the wordings text and its test if needed, trains plain then loop, prints the table
    python3 -m loopab.ab         # the table and the verdict (reads the run folders)
    python3 -m loopab.ab --rounds    # (GPU) the loop's held-out rule test with 1, 2, 4, 8, 12 rounds

Rule fixed before the run (22 Nov 2026), on the full tests at the end. The loop is KEPT only if all three hold:
    1. new rules: the mean of the held-out rule test's hard parts (new wordings, deeper chains of 4-6 steps) is 5.0
       points or more above plain's
    2. reasoning: the mean of ARC-Easy, ARC-Challenge, CommonsenseQA and Winogrande is no more than 0.5 under plain's
    3. language: HellaSwag no more than 2.0 points under plain's
Each test: the better of its two usual scorings (per token, per letter), for both arms.
Shown, not part of the rule: the "seen" part of the rule test, the held-out loss, and whether thinking longer helps
(the held-out loss by rounds, from the [loop] line; the rule test by rounds, from --rounds).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnosis2.judgeab import LEAN, TESTS, read as jread
from gnosis2.race135 import B135

LOOP = dict(loop_layers=4, loop_mouth=4, loop_steps=4, loop_min=2, loop_max=6, loop_grad=4)
ARMS = {"plain": dict(LEAN), "loop": dict(LEAN, **LOOP)}
RUNS = {"plain": "loopab-plain-135", "loop": "loopab-loop-135"}
SEED = 0
WORDINGS = 0.05
TUNE = "compile_mode=max-autotune-no-cudagraphs"
REASON = ("arc_easy", "arc_challenge", "csqa", "winogrande")
RULES = (("rules_seen", "rules: seen"), ("rules_new_wording", "rules: new wording"), ("rules_deeper", "rules: deeper"))
HARD = ("rules_new_wording", "rules_deeper")
NEED_RULES, REASON_SLACK, HS_SLACK = 0.05, 0.005, 0.02


def ratio():
    """Training compute per word, loop against plain."""
    from gnosis2.model import Gnosis2Config
    from gnosis2.loop import cost
    return cost(Gnosis2Config(**ARMS["loop"]))


def tokens(arm, loop_tokens: float) -> int:
    return int(round(loop_tokens * (ratio() if arm == "plain" else 1.0)))


def set_words(arm, tune=False):
    return [f"{k}={v}" for k, v in ARMS[arm].items()] + ([TUNE] if tune else [])


def _best(*v):
    v = [x for x in v if x is not None]
    return max(v) if v else None


def read(d):
    """judgeab.read plus the rule test's three parts (in r["word"]) and the loop's held-out loss by rounds."""
    r = jread(d)
    if r is None:
        return None
    ev, loop = {}, None
    for line in (Path(d) / "metrics.jsonl").read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if x.get("kind") == "evals" and (x.get("full") or not ev.get("full")):
            ev = x
        elif x.get("kind") == "loop":
            loop = x.get("pass_loss")
    for k, _ in RULES:
        r["word"][k] = _best(ev.get(k), ev.get(k + "_char"))
    r["rounds"] = loop
    return r


def _mean(x, keys):
    v = [x.get(k) for k in keys]
    return None if any(a is None for a in v) else sum(v) / len(v)


def verdict(got, loop_tokens=None):
    P, L = got.get("plain"), got.get("loop")
    if not (P and L):
        return [("verdict", "none yet: both arms must have run")]
    want = ratio()
    if not (P["done"] and L["done"] and P["full"] and L["full"]) or abs(P["tok"] / max(L["tok"], 1) - want) > 0.02 * want:
        return [("verdict", f"none yet: plain at {P['tok'] / 1e6:,.0f}M words, loop at {L['tok'] / 1e6:,.0f}M (both must "
                            f"finish, plain reading {want:.3f}x the loop's words, with the full tests at the end)")]
    rp, rl = _mean(P["word"], HARD), _mean(L["word"], HARD)
    qp, ql = _mean(P["word"], REASON), _mean(L["word"], REASON)
    hp, hl = P["word"].get("hs"), L["word"].get("hs")
    if None in (rp, rl, qp, ql, hp, hl):
        return [("verdict", "none: a test was not scored (python3 -m oazy.wordings test; python3 -m oazy.evals --prepare; "
                            "then python3 -m oazy.evals --run <each run>)")]
    ok1, ok2, ok3 = rl - rp >= NEED_RULES - 1e-9, ql - qp >= -REASON_SLACK - 1e-9, hl - hp >= -HS_SLACK - 1e-9
    return [("new rules", f"{'passes' if ok1 else 'fails'}: loop {rl * 100:.1f}% against plain {rp * 100:.1f}% = "
                          f"{(rl - rp) * 100:+.1f} points (new wordings + deeper chains; rule: +{NEED_RULES * 100:.1f} or more)"),
            ("reasoning", f"{'passes' if ok2 else 'fails'}: loop {ql * 100:.1f}% against plain {qp * 100:.1f}% = "
                          f"{(ql - qp) * 100:+.1f} points (ARC-E, ARC-C, CSQA, Winogrande; rule: no more than "
                          f"{REASON_SLACK * 100:.1f} under)"),
            ("language", f"{'passes' if ok3 else 'fails'}: HellaSwag loop {hl * 100:.1f}% against plain {hp * 100:.1f}% = "
                         f"{(hl - hp) * 100:+.1f} points (rule: no more than {HS_SLACK * 100:.1f} under)"),
            ("verdict", "KEEP the loop: all three parts of the rule hold" if ok1 and ok2 and ok3 else
                        "DROP the loop: the rule fixed before the run does not hold"),
            ("held-out loss", f"loop {L['val'] - P['val']:+.3f} against plain (plain read {want:.2f}x the words)")]


def rounds_table(runs: Path, device="cuda"):
    """The loop's held-out rule test with 1, 2, 4, 8 and 12 rounds (does thinking longer help on new rules?)."""
    import torch
    from oazy import hellaswag, term
    from oazy.data import default_dir, run_dir
    from oazy.logres import _load
    dev = torch.device(device)
    m = _load(run_dir(runs, RUNS["loop"]), dev).eval()
    ev = Path(default_dir("data")) / "evals"
    body = []
    for k in (1, 2, 4, 8, 12):
        m._k_force = k
        row = [f"{k} rounds"]
        for name, _ in RULES:
            p = ev / f"{name}.jsonl"
            r = hellaswag.evaluate(m, p, dev) if p.exists() else None
            row.append("–" if r is None else f"{max(r['acc_norm'], r['acc_char']) * 100:.1f}%")
        body.append(row)
    m._k_force = None
    term.header("Loop-135: the held-out rule test by rounds", "4 rounds = as trained to read; more = thinking longer")
    term.table([""] + [t for _, t in RULES], body, right=(1, 2, 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--set", choices=list(ARMS), default=None, help="print an arm's --set words")
    ap.add_argument("--tune", action="store_true", help="with --set: add torch's kernel tuning")
    ap.add_argument("--name", choices=list(ARMS), default=None, help="print an arm's run name")
    ap.add_argument("--tokens", nargs=2, metavar=("ARM", "LOOP_TOKENS"), default=None,
                    help="print an arm's words to read, given the loop arm's")
    ap.add_argument("--wordings", action="store_true", help="print the wordings share")
    ap.add_argument("--rounds", action="store_true", help="(GPU) the loop's rule test by rounds")
    a = ap.parse_args(argv)
    if a.set or a.name or a.tokens or a.wordings:
        print(" ".join(set_words(a.set, a.tune)) if a.set else RUNS[a.name] if a.name else
              WORDINGS if a.wordings else tokens(a.tokens[0], float(a.tokens[1])))
        return
    from oazy import term
    from oazy.data import default_dir, run_dir
    from oazy.report import pct
    runs = Path(a.runs or default_dir("runs"))
    if a.rounds:
        return rounds_table(runs)
    got = {arm: read(run_dir(runs, rn)) for arm, rn in RUNS.items()}
    cols = TESTS + list(RULES)
    body = []
    for arm, label in (("plain", "Bee-135 (plain)"), ("loop", "Loop-135")):
        g = got[arm]
        if g is None:
            body.append([label, "not run yet"] + [""] * (3 + len(cols)))
            continue
        star = "" if g["full"] else "*"
        body.append([term.c(label, "bold"), f"{g['tok'] / 1e9:.2f}B" + ("" if g["done"] else " (running)"),
                     term.duration(g["sec"]), term.rate(g["tok"] / max(g["sec"], 1e-9)), f"{g['val']:.3f}"]
                    + [("–" if g["word"].get(k) is None else pct(g["word"][k]) + star) for k, _ in cols])
    term.header("Loop-135 against Bee-135: the A/B", f"equal weights · equal compute (plain reads {ratio():.2f}x the words) · "
                f"Opal3 + {WORDINGS:.0%} wordings · NEXT-TOKEN")
    term.table(["", "words", "training time", "speed", "held-out loss"] + [t for _, t in cols], body,
               right=tuple(range(1, 5 + len(cols))))
    if any(g is not None and not g["full"] for g in got.values()):
        print("  * a quick check during the run, not the full test at the end")
    L = got.get("loop")
    if L and L.get("rounds"):
        print("  thinking longer (held-out loss by rounds): "
              + "  ".join(f"{k}: {float(v):.3f}" for k, v in sorted(L["rounds"].items(), key=lambda kv: int(kv[0]))))
    term.header("verdict by the rule fixed before the run", color="green")
    term.rows(verdict(got))
    return got


if __name__ == "__main__":
    main()
