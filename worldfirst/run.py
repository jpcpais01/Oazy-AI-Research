"""Stage 1, all of it, in one command: train the Laws model, train its rival, give both the same exams, write the report.

    python3 -m worldfirst.run                      # the real run (a GPU: minutes to an hour; CPU works, slowly)
    python3 -m worldfirst.run --preset tiny --steps 300 --eval-n 20     # a CPU smoke test, a few minutes
    python3 -m worldfirst.run --only report        # re-examine the saved models and rewrite the report

Writes <runs>/<name>/: laws/ and rival/ (checkpoint, metrics.jsonl), results.json (every number), report.md (the
tables, in words). The dashboard shows results.json as the "World-first" card. A run that is stopped resumes nothing:
each model is trained in one go, but a model already trained (its ckpt.pt is there) is not trained again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from . import world as W
from .data import Vocab
from .model import make_pair, n_params, config_of, from_config
from .train import train, evaluate, summarise


def runs_dir():
    try:
        from oazy.data import default_dir
        return default_dir("runs")
    except Exception:
        return os.environ.get("OAZY_RUNS", "/workspace/oazy_runs")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=["tiny", "small", "base"])
    ap.add_argument("--steps", type=int, default=20000, help="training steps for each model")
    ap.add_argument("--batch", type=int, default=128, help="episodes per step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--minutes", type=float, default=None, help="stop each model's training after this long")
    ap.add_argument("--eval-n", type=int, default=200, help="episodes per exam set")
    ap.add_argument("--name", default="worldfirst")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default="all", choices=["all", "laws", "rival", "report"])
    ap.add_argument("--no-curriculum", action="store_true", help="every training level equally often, from the start")
    ap.add_argument("--rival-tags", type=int, default=1, help="1: the rival also gets the 'same thing' marks (fair); 0: plain words only")
    ap.add_argument("--loops", type=int, default=1, help="1: the Laws core loops (the real model); 0: it runs once (the no-loop test)")
    ap.add_argument("--rival-loss", default="all", choices=["all", "answers"],
                    help="all: the rival learns every next word (plain next-word model); answers: only the answer words")
    ap.add_argument("--fresh", action="store_true", help="train again even if a model is saved")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(a.runs or runs_dir()) / a.name
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    def phase(p):
        """<run>/status.json: which part of the run is going on (for the world dashboard)."""
        (out / "status.json").write_text(json.dumps({"phase": p, "preset": a.preset, "steps": a.steps, "batch": a.batch,
                                                     "device": dev, "when": time.strftime("%Y-%m-%d %H:%M")}))

    vocab = Vocab()
    torch.manual_seed(0)
    over = {} if a.loops else {"train_max": 1, "grad_passes": 1, "test_max": 1}
    laws, rival = make_pair(len(vocab), a.preset, **over)
    rival.cfg.loss = a.rival_loss
    say(f"world-first stage 1: {a.preset}, device {dev}, vocab {len(vocab)} words")
    say(f"  Laws model (not next-word): {n_params(laws) / 1e6:.2f}M numbers; rival (next-word): {n_params(rival) / 1e6:.2f}M")
    trained = {}
    for m in (laws, rival):
        d = out / m.kind
        ck = d / "ckpt.pt"
        if ck.exists() and not a.fresh:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            m.load_state_dict(sd["model"])
            trained[m.kind] = sd.get("info", {})
            say(f"{m.kind}: loaded {ck}")
            continue
        if a.only not in ("all", m.kind):
            continue
        say(f"training {m.kind} for {a.steps} steps of {a.batch} episodes")
        phase(m.kind)
        if m.kind == "rival" and not a.rival_tags:
            _no_tags(m)
        info = train(m, vocab, d, steps=a.steps, batch=a.batch, lr=a.lr, device=dev, curriculum=not a.no_curriculum,
                     minutes=a.minutes, say=say, seed=0)
        info["params"] = n_params(m)
        torch.save({"model": m.state_dict(), "config": config_of(m), "vocab": vocab.words, "info": info}, ck)
        (d / "config.json").write_text(json.dumps(config_of(m)))
        trained[m.kind] = info
        say(f"{m.kind}: trained in {info['minutes']:.1f} min, saved {ck}")
    if a.only in ("laws", "rival"):
        return
    if not all(k in trained for k in ("laws", "rival")):
        say("both models are needed for the report")
        return
    rows = {}
    phase("exams")
    for m in (laws, rival):
        t0 = time.time()
        m.to(dev)
        rows[m.kind] = evaluate(m, vocab, n=a.eval_n, device=dev, sweep=bool(a.loops))
        say(f"{m.kind}: examined in {(time.time() - t0) / 60:.1f} min")
    res = {"kind": "worldfirst", "name": a.name, "when": time.strftime("%Y-%m-%d %H:%M"), "device": dev,
           "preset": a.preset, "eval_n": a.eval_n, "rival_tags": bool(a.rival_tags), "rival_loss": a.rival_loss, "loops": bool(a.loops),
           "params": {"laws": n_params(laws), "rival": n_params(rival)}, "train": trained, "rows": rows,
           "summary": {"laws": summarise(rows["laws"], "settle"), "rival": summarise(rows["rival"], "plain")}}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    phase("done")
    rep = report(res)
    (out / "report.md").write_text(rep)
    say("\n" + rep)
    say(f"written: {out / 'results.json'} and {out / 'report.md'}")


def _no_tags(rival):
    """The rival without the 'same thing' marks: its tag table is zeroed and frozen."""
    with torch.no_grad():
        rival.tag.weight.zero_()
    rival.tag.weight.requires_grad_(False)


SPLIT_WORDS = {"seen": "practised kinds, new scenes", "longer": "longer", "deeper": "deeper", "bigger": "bigger falls",
               "new-pairs": "links chained in new ways", "new-mixes": "rules mixed in new ways"}


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def report(res) -> str:
    L, R = res["rows"]["laws"], res["rows"]["rival"]
    get = lambda rows, mode, fam, sp: [r for r in rows if r["mode"] == mode and r["family"] == fam and r["split"] == sp]  # noqa: E731
    lines = []
    sl, sr = res["summary"]["laws"], res["summary"]["rival"]
    lines.append(f"# World-first, stage 1: {res['name']} ({res['when']}, {res['device']})\n")
    loops = res.get("loops", True)
    ans = res.get("rival_loss") == "answers"
    lines.append(f"Laws model {res['params']['laws'] / 1e6:.1f}M ({'loops' if loops else 'NO loops: its core runs once'}; "
                 f"predicts the next scene, NOT a next-word model) vs rival {res['params']['rival'] / 1e6:.1f}M "
                 f"({'a plain transformer learning ONLY the answers' if ans else 'a plain next-word transformer'}), the same world text, "
                 f"{res['eval_n']} new episodes per exam set, a question counts only if its whole answer is right.\n")
    lines.append(f"**Secret exams (never trained on): Laws {_pct(sl.get('secret'))}, rival {_pct(sr.get('secret'))}.** "
                 f"Practised kinds on new scenes: Laws {_pct(sl.get('seen'))}, rival {_pct(sr.get('seen'))}.\n")
    lines.append("| family | exam | levels | Laws | rival | Laws passes |")
    lines.append("|---|---|---|---|---|---|")
    for fam in W.FAMILIES:
        for sp in ["seen"] + list(W.SECRET[fam]):
            lr, rr = get(L, "settle", fam, sp), get(R, "plain", fam, sp)
            if not lr:
                continue
            la = sum(r["right"] for r in lr) / len(lr)
            ra = sum(r["right"] for r in rr) / len(rr) if rr else None
            lv = [r["level"] for r in lr]
            ps = sum(r["passes"] for r in lr) / len(lr)
            secret = "" if sp == "seen" else "**secret**: "
            lines.append(f"| {fam} | {secret}{SPLIT_WORDS.get(sp, sp)} | {min(lv)}-{max(lv)} | {_pct(la)} | {_pct(ra)} | {ps:.1f} |")
    lines.append("\n## Level by level\n")
    lines.append("| family | level | exam | Laws | rival | Laws passes used |")
    lines.append("|---|---|---|---|---|---|")
    for fam in W.FAMILIES:
        for r in [r for r in L if r["mode"] == "settle" and r["family"] == fam]:
            rr = [x for x in R if x["family"] == fam and x["split"] == r["split"] and x["level"] == r["level"]]
            lines.append(f"| {fam} | {r['level']} | {SPLIT_WORDS.get(r['split'], r['split'])} | {_pct(r['right'])} | "
                         f"{_pct(rr[0]['right'] if rr else None)} | {r['passes']:.1f} |")
    ks = sorted({int(r["mode"][1:]) for r in L if r["mode"].startswith("p")})
    if ks:
        lines.append("\n## Does thinking longer help? (Laws model, fixed passes)\n")
        lines.append("| exams | " + " | ".join(f"{k} passes" for k in ks) + " | settles by itself |")
        lines.append("|---|" + "---|" * (len(ks) + 1))
        for part in ("seen", "secret"):
            vals = []
            for mode in [f"p{k}" for k in ks] + ["settle"]:
                rs = [r["right"] for r in L if r["mode"] == mode and ((r["split"] == "seen") == (part == "seen"))]
                vals.append(_pct(sum(rs) / len(rs)) if rs else "-")
            lines.append(f"| {part} | " + " | ".join(vals) + " |")
    lines.append("\n## Reversals (family questions asked the other way round from how the link was told)\n")
    for name, rows, mode in (("Laws", L, "settle"), ("rival", R, "plain")):
        agg = {"True": [0, 0], "False": [0, 0]}
        for r in rows:
            if r["mode"] == mode and r.get("reversed"):
                for k, (ok, n) in r["reversed"].items():
                    agg[k][0] += ok
                    agg[k][1] += n
        f = lambda k: _pct(agg[k][0] / agg[k][1]) if agg[k][1] else "-"  # noqa: E731
        lines.append(f"- {name}: asked the way it was told {f('False')}, asked the other way round {f('True')}")
    tl, tr = res["train"].get("laws", {}), res["train"].get("rival", {})
    lines.append(f"\nTraining: Laws {tl.get('steps')} steps in {tl.get('minutes', 0):.1f} min, rival {tr.get('steps')} steps "
                 f"in {tr.get('minutes', 0):.1f} min, the same number of episodes each. Guessing gets 50% on yes/no, "
                 f"10% on a digit, 1 in the number of things on a 'who/what', and almost nothing on a whole sum.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
