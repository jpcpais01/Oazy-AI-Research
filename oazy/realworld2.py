"""Real World 2 (João, Nov 11): the Real World and Real World Pro exams, marked fairly for models that answer in
whole phrases.

Why. Real World answers a thing bare ("Where is the shell? Answer: chest."), while Field1, the world text Opal runs
read, always says "Answer: the chest." The old marking only read the bare word right after "Answer:", among choices
that mix things and people, so a model that learned Field1's habit was marked down for it, more as it learned it.
And in Real World Pro's lines, the 80 names shared 20 vehicles: two people of one queue could both be "the camper"
(28% of those questions could not be answered).

What changes. The same scenes, wording and choices (the same seeds), but:
  - each choice may be said in its natural forms, "chest" or "the chest" (people, numbers, yes and no: as they are),
    and counts with the form the model finds likeliest; every choice gets the same chance, so nothing is given away
  - Pro's lines give every person a vehicle of their own
The old Real World and Pro lines stay as they were (older runs keep comparing with them); these are new lines,
"real2" and "pro2", on the dashboard's Real World 2 chart.

    python3 -m oazy.realworld2 --run gnosis-opal-10b smollm2-360m       # a checkpoint, or a yardstick
The trainer takes it on the world exams' schedule (runs started or resumed since Nov 11).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

VERSION = 1


def exam_sets():
    """The practised world scenes, as Real World 2 asks them: (family, level, "real2" | "pro2")."""
    from worldfirst import world as W
    return [(f, lv, k) for k in ("real2", "pro2") for f, lv, sp in W.exam_sets() if sp == "train"]


def forms(c: str):
    """The ways a choice may be said: a thing with or without "the"; a person, a number, yes or no as it is."""
    if c.isalpha() and c.islower() and c not in ("yes", "no"):
        return [c, "the " + c]
    return [c]


def _episode(s, i):
    from .worldmix import _make, episode_parts
    return episode_parts(_make((s[0], s[1], "pro" if s[2] == "pro2" else "real"), i), real="pro2" if s[2] == "pro2" else True)


def questions(enc, sets, n):
    """Every answer word to mark: (set index, episode, question, context pieces, [[form pieces] per choice], right)."""
    from .worldmix import cut
    out = []
    for si, s in enumerate(sets):
        for ei in range(n):
            text, spans = _episode(s, ei)
            pieces = cut(text, spans)
            ids_p = enc.encode_ordinary_batch(pieces)
            ids = [enc.eot_token]
            for j, p in enumerate(ids_p):
                if j % 2 == 1:
                    _, _, kind, w, cands, qi = spans[j // 2]
                    fs = [enc.encode_ordinary_batch([" " + f for f in forms(c)]) for c in cands]
                    out.append((si, ei, qi, list(ids), fs, cands.index(w)))
                ids += p
    return out


def mark(model, qs, device, pad_ok=False, budget=16384):
    """Each answer word's pick: the choice whose likeliest form has the highest log-probability. One reading serves
    every form that shares its text up to the form's last piece (yes/no, digits, names, "the" + a thing)."""
    import torch
    from .field1.levels import _forward
    need = {}                                   # reading (tuple) -> index
    plan = []                                   # per question: [[(reading, start, pieces) per form] per choice]
    for _, _, _, ctx, fs, _ in qs:
        per = []
        for cf in fs:
            opts = []
            for fp in cf:
                key = tuple(ctx + fp[:-1])
                if key not in need:
                    need[key] = len(need)
                opts.append((need[key], len(ctx), fp))
            per.append(opts)
        plan.append(per)
    seqs = [list(k) for k in need]
    wants = {}                                  # reading -> [(position, piece)]
    for per in plan:
        for opts in per:
            for r, L, fp in opts:
                for j, t in enumerate(fp):
                    wants.setdefault(r, set()).add((L - 1 + j, t))
    got = {}
    for g, logits in _forward(model, seqs, device, pad_ok, budget):
        for b, r in enumerate(g):
            ws = sorted(wants.get(r, ()))
            if not ws:
                continue
            pos = sorted({p for p, _ in ws})
            lp = torch.log_softmax(logits[b, pos].float(), -1).cpu()
            at = {p: k for k, p in enumerate(pos)}
            for p, t in ws:
                got[(r, p, t)] = float(lp[at[p], t])
    picks = []
    for per in plan:
        best = [max(sum(got[(r, L - 1 + j, t)] for j, t in enumerate(fp)) for r, L, fp in opts) for opts in per]
        picks.append(int(np.argmax(best)))
    return picks


def exam(model, device, n=100, pad_ok=False, sets=None):
    """-> {"real2", "pro2", "families": {family/split: acc}, "n", "version"}: each set weighs the same, a question
    counts when every word of its answer is right (as the world exams)."""
    from . import tok
    sets = sets or exam_sets()
    was = model.training
    model.eval()
    try:
        enc = tok.for_model(model)
        qs = questions(enc, sets, n)
        with torch_no_grad():
            picks = mark(model, qs, device, pad_ok)
    finally:
        if was:
            model.train()
    right = {}
    for (si, ei, qi, _, _, r), p in zip(qs, picks):
        right[(si, ei, qi)] = right.get((si, ei, qi), True) and p == r
    rows, fam, part = [], {}, {}
    for si, s in enumerate(sets):
        v = [ok for (a, _, _), ok in right.items() if a == si]
        acc = sum(v) / max(1, len(v))
        rows.append({"family": s[0], "level": s[1], "split": s[2], "n_q": len(v), "right": acc})
        fam.setdefault(f"{s[0]}/{s[2]}", []).append(acc)
        part.setdefault(s[2], []).append(acc)
    out = {"version": VERSION, "n": n, "families": {k: sum(v) / len(v) for k, v in fam.items()}, "rows": rows}
    for k, v in part.items():
        out[k] = sum(v) / len(v)
    return out


class torch_no_grad:
    def __enter__(self):
        import torch
        self.g = torch.no_grad()
        return self.g.__enter__()

    def __exit__(self, *a):
        return self.g.__exit__(*a)


def run(name, runs: Path, device, n=100, say=print):
    """Examine a run's checkpoint (or a SmolLM2 yardstick) and log a "realworld2" line for the dashboard."""
    from .data import run_dir
    from .worldmix import REFS, ref_files
    d = run_dir(runs, name)
    t0 = time.time()
    if name in REFS:
        from .compare import smol, SMOL_TOKENS
        model, meta, pad_ok = smol(name, device), {"tokens": SMOL_TOKENS[name], "step": 0, "elapsed": 0}, True
        ref_files(d)
    else:
        from .train import load_model
        model, meta = load_model(d / "ckpt.pt", device, mmap=True)
        pad_ok = False
    r = exam(model, device, n=n, pad_ok=pad_ok)
    rec = {"kind": "realworld2", "step": meta.get("step"), "tokens": meta.get("tokens"), "elapsed": meta.get("elapsed"),
           **{k: v for k, v in r.items() if k != "rows"}, "full": True, "seconds": time.time() - t0}
    with open(d / "metrics.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    say(f"{name}: Real World 2 {r['real2'] * 100:.1f}%, Pro 2 {r['pro2'] * 100:.1f}% (guessing ~23%; {n} scenes per set, "
        f"{time.time() - t0:.0f} s)")
    return r


def main(argv=None):
    from .data import default_dir, few_cpu_threads
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--n", type=int, default=100, help="scenes per exam set")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    import torch
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(dev)
    for rn in a.run:
        run(rn, Path(a.runs or default_dir("runs")), dev, a.n)


if __name__ == "__main__":
    main()
