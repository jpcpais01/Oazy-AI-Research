"""Does the bf16 stream (`stream16=1`) cost any loss? A short A/B on Bee-135, with a second seed to size the noise.

    A   f32-stream-135    Bee-135 lean + cuDNN attention, the stream in 32-bit
    B   bf16-stream-135   the same, the stream in bf16 (`stream16=1`), the same seed
    A2  f32b-stream-135   A again with another seed: how far two runs of the SAME setting end apart

All three are NEXT-WORD models, read the same text (Opal with Field1 off) for the same number of tokens, no mentor.

    bash runpod/stream16ab.sh                # trains A, B, A2 one after the other, then prints the table
    python3 -m gnosis2.stream16ab            # the table alone (reads the run folders)
    python3 -m gnosis2.stream16ab --set b    # the --set words of an arm (what the script passes to the trainer)

Why. Lean, cuDNN's attention and torch's tuning are the same arithmetic in another order. The bf16 stream is not: every
add on the stream is rounded to 8 bits of mantissa, in every layer, for the whole run. One batch says nothing is lost
(loss 0.00% apart, gradients 0.9998 alike); only training says whether that holds.

Rule fixed before the run (20 Nov 2026): the bf16 stream is kept if B's final held-out loss is no more than
max(0.01, the gap between A and A2) above A's. 0.01 is about what its 4% of speed buys back in tokens at this size
(the loss falls about 0.3 per e-fold of tokens near the end of a run, so 4% more tokens is worth about 0.012).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .race135 import B135, curve

LEAN = dict(B135, lean=1, attn_backend="cudnn")
ARMS = {"a": LEAN, "b": dict(LEAN, stream16=1), "a2": LEAN}
SEED = {"a": 0, "b": 0, "a2": 1}
RUNS = {"a": ("f32-stream-135", "A   32-bit stream"), "b": ("bf16-stream-135", "B   bf16 stream"),
        "a2": ("f32b-stream-135", "A2  32-bit stream, another seed")}
ALLOW = 0.01
TUNE = "compile_mode=max-autotune-no-cudagraphs"


def set_words(arm, tune=False):
    return [f"{k}={v}" for k, v in ARMS[arm].items()] + ([TUNE] if tune else [])


def late_gap(x, y):
    """The mean of y - x over the held-out points of the second half of the run that both curves have (or None)."""
    dx, dy = dict(x), dict(y)
    both = sorted(set(dx) & set(dy))
    both = [t for t in both if t >= both[-1] / 2] if both else []
    return sum(dy[t] - dx[t] for t in both) / len(both) if both else None


def verdict(got):
    """[(label, text)] by the rule in the module's text. got: {arm: dict(val, tok, done, stream16, cv)}."""
    A, B, A2 = got.get("a"), got.get("b"), got.get("a2")
    if not (A and B):
        return [("B against A", "no verdict yet: both A and B must have run")]
    if not B["stream16"]:
        return [("B against A", "no verdict: B did not run the bf16 stream (the check at its start refused it: see its log)")]
    if not (A["done"] and B["done"]) or abs(A["tok"] - B["tok"]) > 0.01 * A["tok"]:
        return [("B against A", f"no verdict yet: A is at {A['tok'] / 1e6:,.0f}M tokens, B at {B['tok'] / 1e6:,.0f}M (both must finish)")]
    gap = B["val"] - A["val"]
    seeds = abs(A2["val"] - A["val"]) if A2 and A2["done"] else None
    allow = max(ALLOW, seeds or 0.0)
    lines = [("seed gap", f"{seeds:.3f} between A and A2, the same setting twice" if seeds is not None
              else "not measured (A2 has not finished): the rule falls back to 0.01")]
    late = late_gap(A["cv"], B["cv"])
    lines.append(("B against A", f"{'KEEP' if gap <= allow else 'DROP'} the bf16 stream: {gap:+.3f} held-out loss at the end "
                                 f"(rule: no more than {allow:.3f} above A)"
                                 + (f"; {late:+.3f} on average over the second half of the run" if late is not None else "")))
    sa, sb = A["tok"] / max(A["sec"], 1e-9), B["tok"] / max(B["sec"], 1e-9)
    lines.append(("speed", f"B ran {sb / sa:.2f}x A's speed over the whole run (tests and compile included)"))
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
        print(" ".join(set_words(a.set, a.tune)) if a.set else SEED[a.seed] if a.seed else RUNS[a.name][0])
        return
    from oazy import term
    from oazy.data import default_dir, run_dir
    from oazy.report import scores, pct
    runs = Path(a.runs or default_dir("runs"))
    got, body = {}, []
    for arm, (rn, label) in RUNS.items():
        d = run_dir(runs, rn)
        cv = curve(d)
        if not cv:
            body.append([label, rn, "not run yet", "", "", "", "", ""])
            continue
        s = scores(d)
        st = s["st"]
        tok, sec = st.get("tokens") or cv[-1][0], st.get("elapsed") or 0
        got[arm] = dict(cv=cv, val=s["val"] if s["val"] is not None else cv[-1][1], tok=tok, sec=sec, done=st.get("state") == "done",
                        stream16=bool((s["cf"].get("config") or {}).get("stream16")))
        body.append([term.c(label, "bold"), rn, f"{tok / 1e9:.2f}B" + ("" if got[arm]["done"] else " (running)"), term.duration(sec),
                     term.rate(tok / max(sec, 1e-9)), f"{got[arm]['val']:.3f}",
                     pct(s["hs"]) + ("" if s["hs_full"] or s["hs"] is None else "*"), pct(s["piqa"])])
    term.header("The bf16 stream's A/B", "Bee-135 lean + cuDNN attention · Opal, Field1 off · equal tokens · next-word models · no mentor")
    term.table(["", "run", "tokens", "training time", "speed", "held-out loss", "HellaSwag", "PIQA"], body, right=(2, 3, 4, 5, 6, 7))
    term.header("verdict by the rule fixed before the run", color="green")
    term.rows(verdict(got))
    return got


if __name__ == "__main__":
    main()
