"""Judge first, write second: does a Bee-135 that is mainly taught to JUDGE beat the same Bee-135 taught only to write?

    plain   judgeab-plain-135   Bee-135 (lean + cuDNN attention), the next-word lesson alone
    judge   judgeab-judge-135   the same body, seed and text, with the judge's lesson (`judge=10`): some sentences and
                                sentence endings of every row change places (oazy/judge.py), a yes/no head on every
                                piece answers "does what I am reading belong here?", and the next-word loss stays beside
                                it (it is not asked to write a decoy). Loss = next word + 10 x judge; the 10 comes in
                                over the first 10% of the run (`judge_ramp=0.1`: nothing can be judged before the
                                words are known, and on a CPU toy a judge at full weight from the start slowed the writer)

Both read the same text (Opal3, Field1 off) for the same number of tokens, no mentor. Both are still NEXT-WORD models
(the judge arm keeps its next-word head); the judge arm has 577 more weights (the head).

    bash runpod/judgeab.sh                 # trains plain, then judge, then prints the table
    python3 -m gnosis2.judgeab             # the table alone (reads the run folders)
    python3 -m gnosis2.judgeab --set judge # the --set words of an arm (what the script passes to the trainer)

How the tests are scored. A next-word model picks the ending that surprises it least. The judge arm is ALSO scored by
its judge, at the same cost (one reading of each ending): it picks the ending whose last piece gets the highest "yes,
this belongs" (oazy/hellaswag.py). Its next-word scores are shown too: they say what the judge's lesson did to the writer.

Rule fixed before the run (20 Nov 2026), on the full tests at the end, at equal tokens:
    the judge is KEPT only if
      1. HellaSwag by the judge is 5.0 points or more above the plain arm's HellaSwag by next-word loss, and
      2. ARC (the mean of ARC-Easy and ARC-Challenge) by the judge is no more than 1.0 point under the plain arm's.
    The plain arm is given the better of its two usual scorings (per token, per letter) on each test.

What a KEEP would not yet prove: HellaSwag asks the judge's own kind of question (which ending belongs here?), so a gain
there alone could be the scoring rule, not a better body. ARC, PIQA and CommonsenseQA are the check on that, and so is
the held-out loss of the judge arm's writer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .race135 import B135, curve

LEAN = dict(B135, lean=1, attn_backend="cudnn")
ARMS = {"plain": LEAN, "judge": dict(LEAN, judge=10, judge_ramp=0.1)}
SEED = {"plain": 0, "judge": 0}
RUNS = {"plain": "judgeab-plain-135", "judge": "judgeab-judge-135"}
NEED_HS, ARC_SLACK = 0.05, 0.01
TUNE = "compile_mode=max-autotune-no-cudagraphs"
TESTS = [("hs", "HellaSwag"), ("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Chall."), ("piqa", "PIQA"), ("csqa", "CSQA"),
         ("winogrande", "Winogr.")]


def set_words(arm, tune=False):
    return [f"{k}={v}" for k, v in ARMS[arm].items()] + ([TUNE] if tune else [])


def _best(*v):
    v = [x for x in v if x is not None]
    return max(v) if v else None


def read(d):
    """A run's numbers: dict(tok, sec, done, val, cv, judge: was the judge's lesson on, full: are the tests the final full
    ones, word: {test: by next-word loss, the better of per token and per letter}, yes: {test: by the judge's yes})."""
    d = Path(d)
    cv = curve(d)
    if not cv:
        return None
    recs = []
    for line in (d / "metrics.jsonl").read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass

    def last(kind):
        r = [x for x in recs if x.get("kind") == kind]
        return (([x for x in r if x.get("full")] or r)[-1:] or [{}])[0]
    hs, ev = last("hellaswag"), last("evals")
    st, cf = {}, {}
    for name, into in (("status.json", st), ("config.json", cf)):
        try:
            into.update(json.loads((d / name).read_text()))
        except (OSError, ValueError):
            pass
    word = {"hs": _best(hs.get("acc_norm"), hs.get("acc_char"))}
    yes = {"hs": hs.get("acc_judge")}
    for k, _ in TESTS[1:]:
        word[k], yes[k] = _best(ev.get(k), ev.get(k + "_char")), ev.get(k + "_judge")
    tr = [x for x in recs if x.get("kind") == "train"]
    return dict(cv=cv, val=cv[-1][1], tok=st.get("tokens") or cv[-1][0], sec=st.get("elapsed") or 0, done=st.get("state") == "done",
                full=bool(hs.get("full")) and bool(ev.get("full")), judge=float((cf.get("config") or {}).get("judge") or 0) > 0,
                word=word, yes=yes, seen=(tr[-1] if tr else {}))


def _arc(x):
    a, b = x.get("arc_easy"), x.get("arc_challenge")
    return None if a is None or b is None else (a + b) / 2


def verdict(got):
    """[(label, text)] by the rule in the module's text. got: {arm: read(run folder)}."""
    P, J = got.get("plain"), got.get("judge")
    if not (P and J):
        return [("verdict", "none yet: both arms must have run")]
    if not J["judge"]:
        return [("verdict", "none: the judge arm ran without the judge's lesson (see its log)")]
    if not (P["done"] and J["done"] and P["full"] and J["full"]) or abs(P["tok"] - J["tok"]) > 0.01 * P["tok"]:
        return [("verdict", f"none yet: plain is at {P['tok'] / 1e6:,.0f}M tokens, judge at {J['tok'] / 1e6:,.0f}M "
                            f"(both must finish, with the full tests at the end)")]
    if J["yes"]["hs"] is None or P["word"]["hs"] is None or _arc(J["yes"]) is None or _arc(P["word"]) is None:
        return [("verdict", "none: HellaSwag or ARC was not scored (python3 -m oazy.evals --prepare, then score the runs again)")]
    hs = J["yes"]["hs"] - P["word"]["hs"]
    arc = _arc(J["yes"]) - _arc(P["word"])
    ok_hs, ok_arc = hs >= NEED_HS - 1e-9, arc >= -ARC_SLACK - 1e-9
    lines = [("HellaSwag", f"{'passes' if ok_hs else 'fails'}: the judge {J['yes']['hs'] * 100:.1f}% against plain "
                           f"{P['word']['hs'] * 100:.1f}% = {hs * 100:+.1f} points (rule: +{NEED_HS * 100:.1f} or more)"),
             ("ARC", f"{'passes' if ok_arc else 'fails'}: the judge {_arc(J['yes']) * 100:.1f}% against plain "
                     f"{_arc(P['word']) * 100:.1f}% = {arc * 100:+.1f} points (rule: no more than {ARC_SLACK * 100:.1f} under)"),
             ("verdict", ("KEEP the judge: both parts of the rule hold" if ok_hs and ok_arc else
                          "DROP the judge: the rule fixed before the run does not hold"))]
    w = J["word"]["hs"]
    lines.append(("the writer", f"the judge arm's own next-word head: held-out loss {J['val'] - P['val']:+.3f} against plain"
                                + (f", HellaSwag {(w - P['word']['hs']) * 100:+.1f} points" if w is not None else "")))
    sp, sj = P["tok"] / max(P["sec"], 1e-9), J["tok"] / max(J["sec"], 1e-9)
    lines.append(("speed", f"the judge arm ran {sj / sp:.2f}x the plain arm's speed over the whole run (tests and compile included)"))
    seen = J["seen"]
    if seen.get("judge_yes_real") is not None and seen.get("judge_yes_decoy") is not None:
        lines.append(("in training", f"at the end the judge's yes was {seen['judge_yes_real']:.2f} on real text and "
                                     f"{seen['judge_yes_decoy']:.2f} on decoys (1 and 0 would be perfect)"))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--set", choices=list(ARMS), default=None)
    ap.add_argument("--seed", choices=list(ARMS), default=None, help="print an arm's seed")
    ap.add_argument("--name", choices=list(ARMS), default=None, help="print an arm's run name")
    ap.add_argument("--tune", action="store_true", help="with --set: add torch's kernel tuning")
    a = ap.parse_args(argv)
    if a.set or a.seed or a.name:
        print(" ".join(set_words(a.set, a.tune)) if a.set else SEED[a.seed] if a.seed else RUNS[a.name])
        return
    from oazy import term
    from oazy.data import default_dir, run_dir
    from oazy.report import pct
    runs = Path(a.runs or default_dir("runs"))
    got = {arm: read(run_dir(runs, rn)) for arm, rn in RUNS.items()}
    rows = [("plain", "word", "plain, by next-word loss"), ("judge", "yes", "judge arm, by its judge"),
            ("judge", "word", "judge arm, by next-word loss")]
    body = []
    for arm, how, label in rows:
        g = got[arm]
        if g is None:
            body.append([label, "not run yet"] + [""] * (3 + len(TESTS)))
            continue
        star = "" if g["full"] else "*"
        first = how == "word"
        body.append([term.c(label, "bold"), f"{g['tok'] / 1e9:.2f}B" + ("" if g["done"] else " (running)"),
                     term.duration(g["sec"]) if first else "", term.rate(g["tok"] / max(g["sec"], 1e-9)) if first else "",
                     f"{g['val']:.3f}" if first else ""]
                    + [pct(g[how][k]) + (star if g[how][k] is not None else "") for k, _ in TESTS])
    term.header("Judge first, write second: the A/B",
                "Bee-135 lean + cuDNN attention · Opal3, Field1 off · equal tokens · the same seed · no mentor")
    term.table(["", "tokens", "training time", "speed", "held-out loss"] + [t for _, t in TESTS], body,
               right=tuple(range(1, 5 + len(TESTS))))
    if any(g is not None and not g["full"] for g in got.values()):
        print("  * a quick check during the run, not the full test at the end")
    print("  by next-word loss: the better of per token and per letter · by its judge: the ending whose last piece gets the highest yes")
    term.header("verdict by the rule fixed before the run", color="green")
    term.rows(verdict(got))
    return got


if __name__ == "__main__":
    main()
