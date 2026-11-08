"""A finished run's report in the terminal: final scores next to the other models, the loss along the way,
time, speed and cost, the chat fine-tune, and what it writes. Only reads the run folders.

    python3 -m oazy.report --run bee4
    python3 -m oazy.report --run bee4 --vs coral smollm2-135m smollm2-360m openai-gpt2   (the default comparison)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import term
from .data import default_dir, run_dir

GPU_PRICE = (("B300", 9.0), ("B200", 8.0), ("H200", 4.0), ("H100", 3.0), ("5090", 0.9), ("4090", 0.6))
NICE = {"openai-gpt2": "OpenAI GPT-2", "smollm2-135m": "SmolLM2-135M", "smollm2-360m": "SmolLM2-360M",
        "coral": "Coral", "bee4": "Bee4-200M", "coralmini": "Coral Mini", "pegasus": "Pegasus 1", "pegasusmini": "Pegasus 1 Mini", "plain100": "Plain-100M", "bee3": "Bee3", "bee2": "Bee_v2_0.2B"}


def jl(p: Path):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def js(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def scores(d: Path):
    """The run's final numbers: HellaSwag per letter, the three tests, held-out loss, recall, size, cost."""
    m, cf, st = jl(d / "metrics.jsonl"), js(d / "config.json"), js(d / "status.json")
    hs = [x for x in m if x.get("kind") == "hellaswag"]
    hs = ([x for x in hs if x.get("full")] or hs)[-1:] or [{}]
    ev = [x for x in m if x.get("kind") == "evals"]
    ev = ([x for x in ev if x.get("full")] or ev)[-1:] or [{}]
    e = ([x for x in m if x.get("kind") == "eval"] or [{}])[-1]
    mem = ([x for x in m if x.get("kind") == "memory"] or [{}])[-1]
    h = hs[0]
    cost, est = None, False
    if cf.get("trained_tokens") and cf.get("params"):
        cost, est = 6 * cf["params"] * cf["trained_tokens"] / (0.4 * 2.25e15) / 3600 * 8, True
    elif st.get("elapsed") and cf.get("gpu"):
        rate = next((p for k, p in GPU_PRICE if k in cf["gpu"]), 8.0)
        cost = st["elapsed"] / 3600 * rate
    return {"hs": h.get("acc_char", h.get("acc_norm")), "hs_full": bool(h.get("full")), "hs_n": h.get("n"),
            "arc": ev[0].get("arc_easy"), "piqa": ev[0].get("piqa"), "lambada": ev[0].get("lambada"),
            "ev_full": bool(ev[0].get("full")), "val": e.get("val_loss"), "bpb": e.get("val_bpb"), "eval": e,
            "recall": mem.get("recall_acc"), "params": cf.get("params"), "active": cf.get("params_active"),
            "cost": cost, "est": est, "cf": cf, "st": st, "m": m}


def pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


def money(v, est=False):
    if v is None:
        return "–"
    s = f"${v / 1000:.1f}k" if v >= 1000 else f"${v:,.0f}"
    return ("~" + s + " est.") if est else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="bee4")
    ap.add_argument("--vs", nargs="*", default=["coral", "smollm2-360m", "smollm2-135m", "openai-gpt2"])
    ap.add_argument("--runs", default=None)
    a = ap.parse_args()
    runs = Path(a.runs or default_dir("runs"))
    d = run_dir(runs, a.run)
    if not (d / "metrics.jsonl").exists():
        raise SystemExit(f"no run {a.run} in {runs}")
    me = scores(d)
    cf, st, m = me["cf"], me["st"], me["m"]
    name = NICE.get(a.run, a.run)
    tokens = st.get("tokens") or 0
    hours = (st.get("elapsed") or 0) / 3600

    term.header(f"{name} · report", f"{st.get('state', '?')} · {tokens / 1e6:,.0f}M tokens")
    term.rows([
        ("size", f"{(me['params'] or 0) / 1e6:.1f}M parameters" + (f" ({me['active'] / 1e6:.0f}M at work per word)" if me["active"] and me["active"] < (me["params"] or 0) else "")),
        ("trained", f"{tokens / 1e6:,.0f}M tokens in {term.duration(st.get('elapsed') or 0)} on {cf.get('gpu', '?')}"),
        ("speed", f"{tokens / max(1, st.get('elapsed') or 1) / 1e3:,.0f}k tokens/s on average"),
        ("cost", money(me["cost"]) + (f"  (GPU time; ~${me['cost'] / max(tokens / 1e9, 1e-9):.1f} per 1,000M tokens)" if me["cost"] else "")),
        ("mentor", cf.get("mentor")),
    ])

    term.header("Final scores", "higher is better · HellaSwag per letter")
    body = []
    for rn in [a.run] + [x for x in a.vs if x != a.run]:
        rd = run_dir(runs, rn)
        if not (rd / "metrics.jsonl").exists():
            continue
        s = me if rn == a.run else scores(rd)
        mark = lambda full: "" if full else "*"        # noqa: E731
        body.append([term.c(NICE.get(rn, rn), "bold") if rn == a.run else NICE.get(rn, rn),
                     f"{(s['params'] or 0) / 1e6:.0f}M", pct(s["hs"]) + mark(s["hs_full"] or s["hs"] is None),
                     pct(s["arc"]) + mark(s["ev_full"] or s["arc"] is None), pct(s["piqa"]) + mark(s["ev_full"] or s["piqa"] is None),
                     pct(s["lambada"]) + mark(s["ev_full"] or s["lambada"] is None), money(s["cost"], s["est"])])
    term.table(["model", "size", "HellaSwag", "ARC-Easy", "PIQA", "LAMBADA", "training cost"], body, right=(1, 2, 3, 4, 5, 6))
    term.info("* a quick check on part of the questions. Published models' costs are estimates (6 x params x tokens at 40% of a B200).")

    term.header("Held-out loss", "lower is better · our models only (published ones read other text)")
    ev = [x for x in m if x.get("kind") == "eval"]
    total = cf.get("total_tokens") or tokens or 1
    pts = []
    for q in (0.05, 0.1, 0.25, 0.5, 0.7, 0.9, 1.0):
        near = min(ev, key=lambda x: abs(x["tokens"] - q * total)) if ev else None
        if near and (not pts or pts[-1][0] != near["tokens"]):
            pts.append((near["tokens"], near["val_loss"]))
    term.table(["tokens", "held-out loss"], [[f"{t / 1e6:,.0f}M", f"{v:.4f}"] for t, v in pts], right=(0, 1))
    e = me["eval"]
    kinds = {k[9:]: v for k, v in e.items() if k.startswith("val_loss_") and k != "val_loss_own"}
    term.rows([("final", f"{me['val'] or 0:.4f} per GPT-2 token · {me['bpb'] or 0:.3f} bits per byte"),
               ("by kind", "  ".join(f"{k} {v:.3f}" for k, v in kinds.items()) or None),
               ("exact recall", pct(me["recall"]) if me["recall"] is not None else None)])
    cr = run_dir(runs, "coral")
    if a.run != "coral" and (cr / "metrics.jsonl").exists():
        cv = scores(cr)["val"]
        if cv and me["val"]:
            term.info(f"Coral ended at {cv:.4f} held-out ({me['val'] - cv:+.4f} for {name}).")

    sft = js(run_dir(runs, a.run + "-chat") / "sft.json")
    if sft:
        term.header("Chat version", f"{a.run}-chat")
        b, f = sft.get("conv_loss_before"), sft.get("conv_loss_after")
        term.rows([("fine-tune", f"{(sft.get('tokens') or 0) / 1e6:,.0f}M tokens of conversations, {sft.get('steps')} steps, {sft.get('minutes', 0):.1f} min"),
                   ("conversation loss", f"{b:.4f} -> {f:.4f}" if b is not None and f is not None else None)])

    smp = jl(d / "samples.jsonl")
    if smp:
        last = {}
        for x in smp:
            last[x.get("prompt")] = x
        term.header("What it writes", f"after {max(x.get('tokens', 0) for x in last.values()) / 1e6:,.0f}M tokens")
        for p, x in last.items():
            text = " ".join((x.get("text") or "").split())
            term.out("  " + term.c(p, "bold") + " " + text[:300] + ("…" if len(text) > 300 else ""))
            term.out("")


if __name__ == "__main__":
    main()
