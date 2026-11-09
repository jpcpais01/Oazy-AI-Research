"""Stage 1, all of it, in one command: train the Laws model, train its rival, give both the same exams, write the report.

    python3 -m worldfirst.run                      # the real run (a GPU: minutes to an hour; CPU works, slowly)
    python3 -m worldfirst.run --preset tiny --steps 300 --eval-n 20     # a CPU smoke test, a few minutes
    python3 -m worldfirst.run --only report        # re-examine the saved models and rewrite the report
    python3 -m worldfirst.run --name world-big --preset big --steps 200000 --passes 3   # the long run (~29M each, overnight)

Writes <runs>/<name>/: laws/ and rival/ (checkpoint, metrics.jsonl), results.json (every number), report.md (the
tables, in words). The dashboard shows results.json as the "World-first" card. A stopped run carries on where it
stopped when started again with the same command (each model saves its state every 10 minutes), and a model already
trained (its ckpt.pt is there) is not trained again. On a GPU both models train at the same time, one process each.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import world as W
from .data import Vocab
from .model import make_pair, n_params, config_of, from_config, LR
from .train import train, evaluate, summarise


def runs_dir():
    try:
        from oazy.data import default_dir
        return default_dir("runs")
    except Exception:
        return os.environ.get("OAZY_RUNS", "/workspace/oazy_runs")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=["tiny", "small", "base", "big"])
    ap.add_argument("--steps", type=int, default=20000, help="training steps for each model")
    ap.add_argument("--batch", type=int, default=128, help="episodes per step")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (default: 6e-4, 5e-4 for big)")
    ap.add_argument("--minutes", type=float, default=None, help="stop each model's training after this long")
    ap.add_argument("--eval-n", type=int, default=200, help="episodes per exam set")
    ap.add_argument("--name", default="worldfirst")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default="all", choices=["all", "laws", "rival", "report"])
    ap.add_argument("--no-curriculum", action="store_true", help="every training level equally often, from the start")
    ap.add_argument("--rival-tags", type=int, default=1, help="1: the rival also gets the 'same thing' marks (fair); 0: plain words only")
    ap.add_argument("--loops", type=int, default=1, help="1: the Laws core loops (the real model); 0: it runs once (the no-loop test)")
    ap.add_argument("--passes", type=int, default=0,
                    help="N>0: the Laws core always runs exactly N passes, learning through all of them (no random depth, no early stop)")
    ap.add_argument("--rival-loss", default="all", choices=["all", "answers"],
                    help="all: the rival learns every next word (plain next-word model); answers: only the answer words")
    ap.add_argument("--natural", type=int, default=0,
                    help="1: no marks. The world speaks more naturally (she/he/her/him; sums smallest digit first), the rival "
                         "gets no tags, and the Laws model finds the things itself (--finder slots)")
    ap.add_argument("--teach", type=float, default=0.0,
                    help="with --natural 1: marks as a teacher. The Laws model still reads text with no marks, but in "
                         "training its thing finder is graded against the world's marks (this weight; 1 is a good start)")
    ap.add_argument("--linker", type=int, default=0,
                    help="with --natural 1: the Laws model finds the things with the linker (each mention points back to "
                         "an earlier word of the same thing, or starts a new one) instead of slots; taught by the marks "
                         "(--teach, default 1), never given them")
    ap.add_argument("--rival-loops", type=int, default=1,
                    help="N>1: the rival's last 2 layers run N times with the same weights (a looped transformer, same size)")
    ap.add_argument("--rival-ckpt", type=int, default=0,
                    help="1: the rival recomputes its layers in the backward pass (same maths, much less GPU memory, ~30%% slower)")
    ap.add_argument("--laws-from", default=None,
                    help="another run's name: reuse its trained Logos1 instead of training one here (train only the rival; "
                         "the report compares them once that Logos1 has finished; same world flags needed)")
    ap.add_argument("--finder", type=int, default=None, help="slots for the Laws model's thing finder (default 32 with --natural 1)")
    ap.add_argument("--fresh", action="store_true", help="train again even if a model is saved")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--together", type=int, default=None,
                    help="1: train both models at the same time, one process each (the default on a GPU); 0: one after the other")
    ap.add_argument("--workers", type=int, default=None, help="helper processes building batches (default 3 on a GPU, 0 on CPU)")
    a = ap.parse_args(argv)
    a.lr = a.lr or LR[a.preset]
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
        if os.environ.get("WORLD_CHILD"):           # a helper training one model: the parent keeps the run's status
            return
        (out / "status.json").write_text(json.dumps({"phase": p, "preset": a.preset, "steps": a.steps, "batch": a.batch,
                                                     "device": dev, "when": time.strftime("%Y-%m-%d %H:%M")}))

    if a.natural:
        W.STYLE["natural"] = True
        a.rival_tags = 0
        a.finder = a.finder or (48 if a.linker else 32)
    vocab = Vocab()
    torch.manual_seed(0)
    over = {} if a.loops else {"train_max": 1, "grad_passes": 1, "test_max": 1}
    if a.passes and a.loops:
        over = {"train_max": a.passes, "grad_passes": a.passes, "test_max": a.passes, "fixed": a.passes}
    if a.linker:
        a.teach = a.teach or 1.0
    if a.finder:
        over["finder"] = a.finder
        if a.teach:
            over["teach"] = a.teach
        if a.linker:
            over["finder_kind"] = "link"
    laws, rival = make_pair(len(vocab), a.preset, **over)
    rival.cfg.loss = a.rival_loss
    rival.cfg.loops = a.rival_loops
    rival.cfg.ckpt = a.rival_ckpt
    say(f"world-first stage 1: {a.preset}, device {dev}, vocab {len(vocab)} words")
    say(f"  Logos1, the Laws model (not next-word): {n_params(laws) / 1e6:.2f}M numbers; rival (next-word): {n_params(rival) / 1e6:.2f}M")
    together = (dev == "cuda") if a.together is None else bool(a.together)
    todo = [m.kind for m in (laws, rival) if (a.fresh or not (out / m.kind / "ckpt.pt").exists()) and a.only in ("all", m.kind)
            and not (m.kind == "laws" and a.laws_from)]
    if together and a.only == "all" and len(todo) == 2:
        say("training both models at the same time (one process each)")
        phase("both")
        args = list(argv if argv is not None else sys.argv[1:])
        env = dict(os.environ, WORLD_CHILD="1")
        procs = [subprocess.Popen([sys.executable, "-m", "worldfirst.run", *args, "--only", k, "--together", "0"], env=env)
                 for k in todo]
        codes = [p.wait() for p in procs]
        if any(codes):
            say(f"a training process failed ({codes}); see the log above")
            return
        a.fresh = False
    trained = {}
    for m in (laws, rival):
        d = out / m.kind
        ck = d / "ckpt.pt"
        if m.kind == "laws" and a.laws_from:                    # Logos1 comes from another run, never trained here
            src = out.parent / a.laws_from / "laws" / "ckpt.pt"
            if src.exists():
                sd = torch.load(src, map_location="cpu", weights_only=False)
                m.load_state_dict(sd["model"])
                trained["laws"] = sd.get("info", {})
                say(f"laws: loaded from {a.laws_from} ({src})")
            else:
                say(f"laws: {a.laws_from} has not finished yet; the report waits for it (rerun with --only report later)")
            continue
        if ck.exists() and not a.fresh:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            m.load_state_dict(sd["model"])
            trained[m.kind] = sd.get("info", {})
            say(f"{m.kind}: loaded {ck}")
            continue
        st_file = d / "train_state.pt"
        if a.only == "report" and st_file.exists():       # stopped early: examine its last saved state as it is
            st = torch.load(st_file, map_location="cpu", weights_only=False)
            m.load_state_dict(st["model"])
            trained[m.kind] = {"steps": st["step"], "minutes": st["minutes"], "stopped_early": True,
                               "frontier": {f: W.TRAIN_LEVELS[f][k] for f, k in st["cur"]["front"].items()}}
            say(f"{m.kind}: stopped early; examining its state saved at step {st['step']}")
            continue
        if a.only not in ("all", m.kind):
            continue
        say(f"training {m.kind} for {a.steps} steps of {a.batch} episodes")
        phase(m.kind)
        if a.fresh:
            (d / "train_state.pt").unlink(missing_ok=True)
        if m.kind == "rival" and not a.rival_tags:
            _no_tags(m)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(config_of(m)))      # the dashboard describes it while it trains
        info = train(m, vocab, d, steps=a.steps, batch=a.batch, lr=a.lr, device=dev, curriculum=not a.no_curriculum,
                     minutes=a.minutes, say=say, seed=0, workers=a.workers)
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
           "preset": a.preset, "eval_n": a.eval_n, "rival_tags": bool(a.rival_tags), "rival_loss": a.rival_loss, "loops": bool(a.loops), "passes": a.passes,
           "natural": bool(a.natural), "finder": a.finder or 0, "teach": a.teach if a.finder else 0, "linker": bool(a.linker), "rival_teach": rival.cfg.teach, "rival_loops": rival.cfg.loops,
           "laws_from": a.laws_from,
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
               "new-pairs": "links chained in new ways", "new-mixes": "rules mixed in new ways", "more-moves": "more moves in a row"}


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
    loop_words = (f"always {res['passes']} passes" if res.get("passes") else "loops") if loops else "NO loops: its core runs once"
    lines.append(f"Logos1 (the Laws model) {res['params']['laws'] / 1e6:.1f}M ({loop_words}; "
                 f"predicts the next scene, NOT a next-word model) vs rival {res['params']['rival'] / 1e6:.1f}M "
                 f"({'a plain transformer learning ONLY the answers' if ans else 'a plain next-word transformer'}"
                 + (f", its last 2 layers looped {res['rival_loops']} times" if res.get("rival_loops", 1) > 1 else "")
                 + "), the same world text, "
                 + (f"Logos1 taken from the run {res['laws_from']}, "
                    if res.get("laws_from") else "")
                 + f"{res['eval_n']} new episodes per exam set, a question counts only if its whole answer is right.\n")
    if res.get("finder"):
        fd = [r["found"] for r in L if r["mode"] == "settle" and "found" in r]
        lines.append(f"**No marks**: {'the world speaks naturally (she/he/her/him, sums smallest digit first), ' if res.get('natural') else ''}"
                     f"the rival gets no tags, and Logos1 finds the things itself "
                     + (f"with its linker (each mention points back to an earlier word of the same thing, or starts a new one; up to {res['finder']} things). "
                        if res.get("linker") else f"with {res['finder']} slots. ")
                     + (f"In training only, the world's marks were its finder's answer key (marks as a teacher, weight {res['teach']:g}); "
                        "at the exams it reads plain text like the rival. " if res.get("teach") else "")
                     + ("The rival gets the very same answer key, the same way: a link head on its last layer learns which "
                        "earlier word names the same thing (same loss, same weight, sizes still matched). " if res.get("rival_teach") else "")
                     + f"Things it found cleanly (all of a thing's words in one group of their own): {_pct(sum(fd) / len(fd)) if fd else '-'}.\n")
    lines.append(f"**Secret exams (never trained on): Logos1 {_pct(sl.get('secret'))}, rival {_pct(sr.get('secret'))}.** "
                 f"Practised kinds on new scenes: Logos1 {_pct(sl.get('seen'))}, rival {_pct(sr.get('seen'))}.\n")
    lines.append("| family | exam | levels | Logos1 | rival | Logos1 passes |")
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
    lines.append("| family | level | exam | Logos1 | rival | Logos1 passes used |")
    lines.append("|---|---|---|---|---|---|")
    for fam in W.FAMILIES:
        for r in [r for r in L if r["mode"] == "settle" and r["family"] == fam]:
            rr = [x for x in R if x["family"] == fam and x["split"] == r["split"] and x["level"] == r["level"]]
            lines.append(f"| {fam} | {r['level']} | {SPLIT_WORDS.get(r['split'], r['split'])} | {_pct(r['right'])} | "
                         f"{_pct(rr[0]['right'] if rr else None)} | {r['passes']:.1f} |")
    ks = sorted({int(r["mode"][1:]) for r in L if r["mode"].startswith("p")})
    if ks:
        lines.append("\n## Does thinking longer help? (Logos1, fixed passes)\n")
        lines.append("| exams | " + " | ".join(f"{k} passes" for k in ks) + " | " + (f"its usual {res['passes']} |" if res.get("passes") else "settles by itself |"))
        lines.append("|---|" + "---|" * (len(ks) + 1))
        for part in ("seen", "secret"):
            vals = []
            for mode in [f"p{k}" for k in ks] + ["settle"]:
                rs = [r["right"] for r in L if r["mode"] == mode and ((r["split"] == "seen") == (part == "seen"))]
                vals.append(_pct(sum(rs) / len(rs)) if rs else "-")
            lines.append(f"| {part} | " + " | ".join(vals) + " |")
    lines.append("\n## Reversals (family questions asked the other way round from how the link was told)\n")
    for name, rows, mode in (("Logos1", L, "settle"), ("rival", R, "plain")):
        agg = {"True": [0, 0], "False": [0, 0]}
        for r in rows:
            if r["mode"] == mode and r.get("reversed"):
                for k, (ok, n) in r["reversed"].items():
                    agg[k][0] += ok
                    agg[k][1] += n
        f = lambda k: _pct(agg[k][0] / agg[k][1]) if agg[k][1] else "-"  # noqa: E731
        lines.append(f"- {name}: asked the way it was told {f('False')}, asked the other way round {f('True')}")
    lines.append("\n## False beliefs (where does someone think a thing is, when it was moved while they were away)\n")
    for name, rows, mode in (("Logos1", L, "settle"), ("rival", R, "plain")):
        agg = {"True": [0, 0], "False": [0, 0]}
        for r in rows:
            if r["mode"] == mode and r.get("belief"):
                for k, (ok, n) in r["belief"].items():
                    agg[k][0] += ok
                    agg[k][1] += n
        f = lambda k: _pct(agg[k][0] / agg[k][1]) if agg[k][1] else "-"  # noqa: E731
        lines.append(f"- {name}: they saw it last where it is {f('False')}, they believe something false {f('True')}")
    tl, tr = res["train"].get("laws", {}), res["train"].get("rival", {})
    early = [k for k, t in (("Logos1", tl), ("rival", tr)) if t.get("stopped_early")]
    lines.append(f"\nTraining: Logos1 {tl.get('steps')} steps in {tl.get('minutes', 0):.1f} min, rival {tr.get('steps')} steps "
                 f"in {tr.get('minutes', 0):.1f} min" + (f" (stopped early: {' and '.join(early)}, examined at the last saved state)" if early
                 else ", the same number of episodes each") + ". Guessing gets 50% on yes/no, "
                 f"10% on a digit, 1 in the number of things on a 'who/what', and almost nothing on a whole sum.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
