"""The 135M race: SmolLM2-135M's own body and recipe from random weights, against ours at the same size.

    A  smol-race-135   `smol`: SmolLM2-135M's body (30 x 576, a Llama) and their recipe (AdamW 3e-3, 2,048-word windows)
    B  bee-race-135    Bee-135: the Bee6 body at 24 layers of 576 with Canon layers, our recipe, about 132M counting all
    C  phi-race-135    Phi-135 (Phrase-Hinted Inference): Bee-135 with the phrase hint in place of the two value tables (the same size as B to within 0.1M)

All three are NEXT-WORD models, read the same text (Opal with Field1 off) for the same number of tokens, no mentor.

    bash runpod/race135.sh                 # trains A, B, C one after the other, then prints the table
    python3 -m gnosis2.race135             # the table alone (reads the run folders)
    python3 -m gnosis2.race135 --set b     # the --set words of an arm (what the script passes to the trainer)

Rules fixed before the race (16 Nov 2026):
    B stays our body only if it reaches A's final held-out loss on at least 1.3x fewer tokens.
    C stays only if it beats B by 0.02 held-out loss or more at equal tokens and is less than 10% slower.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

B135 = dict(n_layer=24, d_model=576, attn_heads=9, attn_kv_heads=3, attn_head_dim=64, mlp_hidden=2304,
            mod_layers="", key_offset="3,7,11,15,19,23", fp8=0)
C135 = dict(B135, n_ve=0, hint_rows=73000, hint_dim=32, hint_heads=4, hint_orders="2,3")
ARMS = {"b": B135, "c": C135}
RUNS = [("smol-race-135", "A  SmolLM2-135M body, their recipe"), ("bee-race-135", "B  Bee-135"), ("phi-race-135", "C  Phi-135 (Bee-135 + phrase hint)")]
NEED_FEWER, NEED_GAIN, MAX_SLOWER = 1.3, 0.02, 0.10


def set_words(arm):
    return [f"{k}={v}" for k, v in ARMS[arm].items()]


def curve(d):
    """[(tokens, held-out loss)] of a run, in order."""
    out = []
    f = Path(d) / "metrics.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        v = r.get("val_loss", r.get("val"))
        if v is not None and r.get("tokens") is not None:
            out.append((float(r["tokens"]), float(v)))
    return sorted(out)


def tokens_to_reach(cv, target):
    """Tokens at which a curve first gets to `target` (straight line between its two nearest points), or None."""
    for (t0, v0), (t1, v1) in zip(cv, cv[1:]):
        if v1 <= target < v0:
            return t0 + (t1 - t0) * (v0 - target) / (v0 - v1)
    return cv[0][0] if cv and cv[0][1] <= target else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--set", choices=list(ARMS), default=None)
    a = ap.parse_args(argv)
    if a.set:
        print(" ".join(set_words(a.set)))
        return
    from oazy import term
    from oazy.data import default_dir, run_dir
    from oazy.report import scores, pct
    runs = Path(a.runs or default_dir("runs"))
    got, body = {}, []
    for rn, label in RUNS:
        d = run_dir(runs, rn)
        cv = curve(d)
        if not cv:
            body.append([label, rn, "not run yet", "", "", "", "", "", ""])
            continue
        s = scores(d)
        st = s["st"]
        tok, sec = st.get("tokens") or cv[-1][0], st.get("elapsed") or 0
        got[rn] = dict(cv=cv, val=s["val"] if s["val"] is not None else cv[-1][1], tok=tok, sec=sec)
        body.append([term.c(label, "bold"), rn, f"{(st.get('params') or 0) / 1e6:.1f}M", f"{tok / 1e9:.2f}B", term.duration(sec),
                     term.rate(tok / max(sec, 1e-9)), f"{got[rn]['val']:.3f}",
                     pct(s["hs"]) + ("" if s["hs_full"] or s["hs"] is None else "*"), pct(s["logres"])])
    term.header("The 135M race", "Opal, Field1 off · equal tokens · next-word models · no mentor")
    term.table(["", "run", "size", "tokens", "training time", "speed", "held-out loss", "HellaSwag", "LogT"], body,
               right=(2, 3, 4, 5, 6, 7, 8))
    A, B, C = (got.get(rn) for rn, _ in RUNS)
    lines = []
    if A and B:
        need = tokens_to_reach(B["cv"], A["val"])
        if need is None:
            lines.append(("B against A", f"FAIL: Bee-135 never reaches A's {A['val']:.3f} (it ends at {B['val']:.3f})"))
        else:
            x = A["tok"] / need
            lines.append(("B against A", f"{'PASS' if x >= NEED_FEWER else 'FAIL'}: Bee-135 reaches A's {A['val']:.3f} at "
                                         f"{need / 1e9:.2f}B tokens, {x:.2f}x fewer (rule: {NEED_FEWER}x)"))
    if B and C:
        gain = B["val"] - C["val"]
        slower = 1 - (C["tok"] / max(C["sec"], 1e-9)) / (B["tok"] / max(B["sec"], 1e-9))
        ok = gain >= NEED_GAIN and slower < MAX_SLOWER
        lines.append(("C against B", f"{'KEEP' if ok else 'DROP'} the hint (Phi-135): {gain:+.3f} held-out loss (rule: {NEED_GAIN} or more), "
                                     f"{slower * 100:+.0f}% slower (rule: under {MAX_SLOWER * 100:.0f}%)"))
    if lines:
        term.header("verdict by the rules fixed before the race", color="green")
        term.rows(lines)
    return got


if __name__ == "__main__":
    main()
