"""Field1's level table (João, Nov 11): how a model does on every family of Field1, split by how hard the scene is, so
the levels a model of this size only guesses at can be cut or read less.

    python3 -m oazy.field1.levels --run smollm2-360m                 # Hugging Face's trained SmolLM2-360M (GPU: ~5 min)
    python3 -m oazy.field1.levels --run gnosis-opal-10b              # one of ours (its ckpt.pt), any time in its run
    python3 -m oazy.field1.levels --run smollm2-360m --n 40 --shots 1 --device cpu   # a quick look on a CPU pod
    python3 -m oazy.field1.levels --show lines                       # what is asked, with its choices (no model)

What it asks. Fresh Field1 scenes, made the way the builder makes them (same simulators, same walls) but from seeds
the builder never uses, so no run has read them. Each scene is told as a story, then its questions are asked one at a
time as plain text, "<question> Answer:", with the earlier questions of the scene and their answers given before it.
Before the scene come --shots solved scenes of the same family (default 2), so a model that never read Field1 (like
SmolLM2) knows the form of the answer. Pieces are SmolLM2's, scenes are joined by <|endoftext|>, as in Opal.

How it is marked. The model picks among choices by the log-probability of each choice's pieces after "Answer:": yes
or no (and "we can't tell" where the family has it), numbers near the true one (in the same digits or words), the
people or things of the scene that could be the answer (in a "who is taller, Ana or Bo?" question, just those two).
Chance is 1 over the number of choices, question by question; the floor a row must beat is chance or what always
giving that row's commonest answer would score, whichever is higher (yes and no are not always half and half).

The levels. Every family writes down how hard each scene is (oazy/field1/core.note: how many people, moves, links,
digits ...) and what each question asks (core.ask: its kind and its own knobs, e.g. how many links between the two
people compared). The table gives, per family, the score against chance for every value of every knob, and for every
kind of question. A row is "at chance" when the score is within two standard errors of guessing.

What it writes: <runs>/<run>/field1_levels.json (every question's result) and field1_levels.md (the table), and a
"field1" line in the run's metrics.jsonl, which the dashboard draws (the Field1 chart and the by-family card).

During a run. Runs started since Nov 11 on Opal take a short version of it (60 scenes per family) on the tests'
schedule, every 200M tokens. A run that started before keeps one checkpoint only (overwritten every few minutes), so its
past cannot be examined; the watcher examines each new checkpoint from now on, every 200M tokens, beside the training:

    python3 -m oazy.field1.levels --watch gnosis-opal-10b             # leave it in its own tmux window
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

SEED = "levels"                   # the builder's seeds are (0, job, i) and (7, job, i): these are never read in training
SHOT_SEED = "levels-shots"

# ---------------------------------------------------------------- scenes
def scene(family: str, i, seed=SEED) -> dict:
    """Scene i of a family -> {"story", "qas" [(question, answer, kind, knobs)], "knobs"}; the walls as in the builder."""
    from . import GEN, VERSION, TRIES
    from . import core, guard
    for t in range(TRIES):
        core.take_notes()
        r = random.Random(f"field1|{VERSION}|{family}|{(seed, i)}|{t}")
        st, qas = GEN[family](r)
        knobs = core.take_notes()
        story = st.tell()
        qas = [q for q in qas if q.q]
        qs = [(core.end(core.cap(q.q), "?"), q.a, q.kind, dict(q.knobs)) for q in qas]
        text = story + "\n" + "\n".join(f"{q} Answer: {a}." for q, a, *_ in qs)
        if not guard.leaks(text):
            return {"family": family, "story": story, "qas": qs, "knobs": knobs}
    raise RuntimeError(f"{family} {seed} {i}: every try said something held back")


def told(sc: dict, upto=None) -> str:
    """A scene as text: the story, then its questions answered (all of them, or the first `upto`)."""
    qs = sc["qas"] if upto is None else sc["qas"][:upto]
    return sc["story"] + "".join(f"\n{q} Answer: {a}." for q, a, *_ in qs)


# ---------------------------------------------------------------- the choices of a question
def _vocab():
    from . import core, fam_order, fam_space, fam_people, fam_logic
    names = set(core.FEMALE) | set(core.MALE) | set(fam_logic.PET)
    events = [fam_order.EV_NICE.get(e, e) for e in fam_order.EVENTS]
    cats = [
        sorted(set(core.CONTAINERS) | set(fam_people.SPOTS) | {"cupboard"}),
        list(core.ROOMS),
        list(fam_space.COVERS),
        list(fam_space.PILE),
        list(fam_order.ROW_THINGS),
        list(core.ANIMALS),
        list(core.SMALL),
        events,
        list(core.FURNITURE) + ["floor"],
    ]
    cats += [list(g[1]) for g in fam_logic.GRID]
    return names, cats


_V = None


def vocab():
    global _V
    if _V is None:
        _V = _vocab()
    return _V


DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
DIRS = ["north", "east", "south", "west"]
CANT = "we can't tell"


def _num(s):
    from .core import NUM_WORDS
    if re.fullmatch(r"\d+", s):
        return int(s), "digits"
    if s in NUM_WORDS:
        return NUM_WORDS.index(s), "words"
    return None, None


def _say(n, form):
    from .core import num_word
    return num_word(n) if form == "words" and n <= 20 else str(n)


def _near(n, form, lo=0, hi=None, k=2):
    out = [m for m in range(n - k, n + k + 1) if m >= lo and (hi is None or m <= hi)]
    while len(out) < 2 * k + 1 and (hi is None or out[-1] < hi):       # near the floor: more above instead
        out.append(out[-1] + 1)
    return [_say(m, form) for m in out]


def _digit_slips(ans: str, rng: random.Random, k=4):
    """A big written number with one digit off by one (the slips of a sum): '1,482' -> '1,492', '1,382', ..."""
    spots = [i for i, ch in enumerate(ans) if ch.isdigit()]
    out = {ans}
    for _ in range(60):
        if len(out) > k:
            break
        i = rng.choice(spots)
        d = int(ans[i]) + rng.choice((-1, 1))
        if not 0 <= d <= 9 or (d == 0 and i == spots[0]):
            continue
        out.add(ans[:i] + str(d) + ans[i + 1:])
    return sorted(out)


def _phrases(text, words, the=True):
    """Every '(the) (colour) word' of the text for these words, as said there."""
    from .core import COLOURS
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    col = "|".join(COLOURS)
    pat = (r"\b[Tt]he " if the else r"\b") + rf"(?:(?:{col}) )?(?:{alt})\b"
    return [("the " + m.group(0)[4:]) if the else m.group(0) for m in re.finditer(pat, text)]


def choices(family: str, q: str, a: str, story: str, rng=None, knobs=None):
    """The choices a question is marked among (the true answer one of them), or None when it cannot be marked fairly.
    knobs["_max"]: the most a place can be ("fifth" is no choice in a row of three)."""
    rng = rng or random.Random(f"{family}|{q}|{a}")
    names, cats = vocab()
    al = a.lower()
    if al in ("yes", "no", CANT):
        return ["yes", "no", CANT] if (family in ("rules", "sets") or al == CANT) else ["yes", "no"]
    if a in DAYS:
        return list(DAYS)
    if a in DIRS:
        return list(DIRS)
    from .core import ORD_WORDS, COLOURS
    if a in ORD_WORDS[1:]:
        top = min(len(ORD_WORDS) - 1, (knobs or {}).get("_max", 99))
        lo = max(1, min(ORD_WORDS.index(a) - 2, top - 4))
        return ORD_WORDS[lo:min(top, lo + 4) + 1]
    n, form = _num(a)
    if n is not None:
        if form == "digits" and n >= 20:
            return _digit_slips(a, rng)
        return _near(n, form, hi=20 if form == "words" else None)
    if re.fullmatch(r"[\d ,]+", a) and any(ch.isdigit() for ch in a):     # a sum written '1 4 8 2' or '1,482'
        return _digit_slips(a, rng)
    if " and " in a and a not in names:                                  # 'the key and the coin': not marked
        return None
    if a == "nothing" or (family == "shells" and q.lower().startswith("what")):
        from .core import SMALL
        out = ["nothing"] + sorted(set(_phrases(story, SMALL)))
        return out if a in out and len(out) > 1 else None
    if a in names:
        pool = sorted({w for w in re.findall(r"\b[A-Z][a-z]+\b", story) if w in names})
    elif a in COLOURS and family == "clues":
        pool = sorted({c for c in COLOURS if re.search(rf"\b{c}\b", story)})
    else:
        bare = not al.startswith("the ")
        head = a if bare else a[4:]
        words = {w for c in cats if any(head == w or head.endswith(" " + w) for w in c) for w in c}
        if not words:
            return None
        pool = sorted(set(_phrases(story, words, the=not bare)))
    if a not in pool:
        pool.append(a)
    heads = {c.split()[-1] for c in pool if len(c.split()) > 2}       # "the bucket" of "the bucket game" is no choice
    pool = [c for c in pool if not (len(c.split()) == 2 and c.startswith("the ") and c.split()[1] in heads) or c == a]
    inq = [c for c in pool if re.search(rf"\b{re.escape(c)}\b", q)]
    if a in inq and len(inq) >= 2:                  # "of Ana and Bo, who is taller?": those two
        pool = inq
    elif len(pool) - len(inq) >= 2:                 # "who is right behind Ana?": not Ana
        pool = [c for c in pool if c not in inq]
    return sorted(set(pool)) if len(set(pool)) >= 2 else None


# ---------------------------------------------------------------- the exam
def items(families, n, shots, enc, max_len):
    """Every question to mark: {"family", "i", "j", "kind", "knobs", "ctx" (pieces), "cands" (pieces each), "right",
    "choices"}; and how many questions could not be marked."""
    eot = enc.eot_token
    out, unmarked = [], {}
    for f in families:
        shot_txt = [told(scene(f, s, SHOT_SEED)) for s in range(shots)]
        shot_ids = []
        for t in shot_txt:
            shot_ids += [eot] + enc.encode_ordinary(t)
        for i in range(n):
            sc = scene(f, i)
            for j, (q, a, kind, qk) in enumerate(sc["qas"]):
                ch = choices(f, q, a, sc["story"], knobs=qk)
                if ch is None:
                    unmarked[f] = unmarked.get(f, 0) + 1
                    continue
                ctx = shot_ids + [eot] + enc.encode_ordinary(told(sc, j) + f"\n{q} Answer:")
                cands = enc.encode_ordinary_batch([" " + c for c in ch])
                one = all(len(c) == 1 for c in cands) and len({c[0] for c in cands}) == len(cands)
                if not one:                                         # one piece each: one reading marks them all; else
                    # each choice is read, with its full stop, so "1" is not a start of "12"
                    cands = enc.encode_ordinary_batch([" " + c + "." for c in ch])

                room = max_len - max(len(c) for c in cands) - 1
                if len(ctx) > room:                                 # too long for the model: the oldest shots go first
                    ctx = ctx[len(ctx) - room:]
                out.append({"family": f, "i": i, "j": j, "kind": kind, "knobs": {**sc["knobs"], **qk},
                            "ctx": ctx, "cands": cands, "one": one, "right": ch.index(a), "choices": ch})
    return out, unmarked


def _forward(model, seqs, device, pad_ok, budget):
    """Log-softmax rows of every sequence (a list of tensors on the CPU, each (len, vocab) would be too big: we keep
    only what is asked). Yields (index, logits) in groups of the same length (or padded on the right when the model
    is plainly causal, which leaves every real place as it was)."""
    import torch
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k]))
    k0 = 0
    while k0 < len(order):
        T0 = len(seqs[order[k0]])
        k1 = k0
        while k1 < len(order):
            T = len(seqs[order[k1]])
            if (not pad_ok and T != T0) or (k1 - k0 + 1) * T > budget and k1 > k0:
                break
            k1 += 1
        g = order[k0:k1]
        T = max(len(seqs[k]) for k in g)
        x = torch.zeros((len(g), T), dtype=torch.long)
        for b, k in enumerate(g):
            x[b, :len(seqs[k])] = torch.tensor(seqs[k])
        x = x.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model._body(x)
        yield g, logits
        k0 = k1


def mark(model, its, device, budget=16384, pad_ok=False, say=print):
    """Each item's choice: the one whose pieces the model finds likeliest after the context."""
    import torch
    seqs, owner = [], []                        # (item, choice or -1 for a one-reading item)
    for k, it in enumerate(its):
        if it["one"]:
            seqs.append(it["ctx"])
            owner.append((k, -1))
        else:
            for c, cc in enumerate(it["cands"]):
                seqs.append(it["ctx"] + cc)
                owner.append((k, c))
    scores = [[0.0] * len(it["cands"]) for it in its]
    t0, done = time.time(), 0
    for g, logits in _forward(model, seqs, device, pad_ok, budget):
        for b, s in enumerate(g):
            k, c = owner[s]
            it = its[k]
            L = len(it["ctx"])
            if c < 0:
                lp = torch.log_softmax(logits[b, L - 1].float(), -1)
                for ci, cc in enumerate(it["cands"]):
                    scores[k][ci] = float(lp[cc[0]])
            else:
                cc = it["cands"][c]
                lp = torch.log_softmax(logits[b, L - 1:L - 1 + len(cc)].float(), -1)
                scores[k][c] = float(lp.gather(1, torch.tensor(cc, device=lp.device)[:, None]).sum())
        done += len(g)
        if say and (done == len(seqs) or time.time() - t0 > 30 and done % 500 < len(g)):
            say(f"  read {done:,} of {len(seqs):,} ({time.time() - t0:.0f} s)")
    for it, sc in zip(its, scores):
        it["picked"] = max(range(len(sc)), key=lambda c: sc[c])
        it["ok"] = it["picked"] == it["right"]
    return its


# ---------------------------------------------------------------- the table
def _row(xs):
    """Score, chance (1 / choices), and the floor: chance, or what always giving the same answer would score when the
    answers lean one way (in "rules", "no" is right more often than "yes"), whichever is higher."""
    n = len(xs)
    acc = sum(x["ok"] for x in xs) / n
    ch = sum(1 / len(x["choices"]) for x in xs) / n
    same = {}
    for x in xs:
        same[x["choices"][x["right"]]] = same.get(x["choices"][x["right"]], 0) + 1
    floor = max(ch, max(same.values()) / n)
    se = math.sqrt(sum((1 / len(x["choices"])) * (1 - 1 / len(x["choices"])) for x in xs)) / n
    if floor > ch:
        se = math.sqrt(floor * (1 - floor) / n)
    z = (acc - floor) / se if se > 0 else 0.0
    lift = (acc - floor) / (1 - floor) if floor < 1 else 0.0
    return {"n": n, "acc": acc, "chance": ch, "floor": floor, "lift": lift, "z": z, "at_chance": z < 2}


def table(its, min_n=12):
    """{family: {"all": row, "knobs": {knob: {value: row}}, "kinds": {kind: row}}}."""
    out = {}
    for f in dict.fromkeys(it["family"] for it in its):
        xs = [it for it in its if it["family"] == f]
        fam = {"all": _row(xs), "knobs": {}, "kinds": {}}
        for k in sorted({k for it in xs for k in it["knobs"] if not k.startswith("_")}):
            vals = {}
            for it in xs:
                if k in it["knobs"]:
                    vals.setdefault(it["knobs"][k], []).append(it)
            if len(vals) < 2 and k != "sub":
                continue
            fam["knobs"][k] = {v: _row(g) for v, g in sorted(vals.items(), key=lambda kv: str(kv[0]).zfill(4))
                               if len(g) >= min_n}
        for kind in sorted({it["kind"] for it in xs}):
            g = [it for it in xs if it["kind"] == kind]
            if len(g) >= min_n:
                fam["kinds"][kind] = _row(g)
        out[f] = fam
    return out


def _cell(r):
    flag = " ·" if r["at_chance"] else ""
    return f"{r['acc'] * 100:5.1f} vs {r['floor'] * 100:4.1f}{flag}"


def markdown(name, tab, unmarked, meta) -> str:
    lines = [f"# Field1 levels: {name}", "",
             f"{meta['questions']:,} questions, {meta['n']} scenes per family, {meta['shots']} solved scenes before each, "
             f"{meta['seconds']:.0f} s. Score vs the floor in % (the floor: guessing, or always giving the commonest answer "
             f"when that scores more); a dot = within two standard errors of the floor. Lift = how much of the way from the "
             f"floor to 100%.", "",
             "| family | score vs floor | lift | questions |", "|---|---|---|---|"]
    for f, t in tab.items():
        a = t["all"]
        lines.append(f"| {f} | {_cell(a)} | {a['lift'] * 100:+.0f}% | {a['n']} |")
    for f, t in tab.items():
        lines += ["", f"## {f}", ""]
        for k, vals in t["knobs"].items():
            if vals:
                lines.append(f"- **{k}**: " + ";  ".join(f"{v}: {_cell(r)} (n {r['n']})" for v, r in vals.items()))
        if t["kinds"]:
            lines.append("- **question**: " + ";  ".join(f"{v}: {_cell(r)} (n {r['n']})" for v, r in t["kinds"].items()))
        if unmarked.get(f):
            lines.append(f"- ({unmarked[f]} questions with a list for an answer were not marked)")
    return "\n".join(lines) + "\n"


def show_text(tab, say=print):
    from .. import term
    say("")
    say(term.c(f"{'family':10s} {'score':>6s} {'floor':>7s} {'lift':>6s} {'n':>5s}", "dim"))
    for f, t in tab.items():
        a = t["all"]
        col = "dim" if a["at_chance"] else ("green" if a["lift"] > 0.3 else "yellow")
        say(term.c(f"{f:10s} {a['acc'] * 100:5.1f}% {a['floor'] * 100:6.1f}% {a['lift'] * 100:+5.0f}% {a['n']:5d}", col))
    for f, t in tab.items():
        say("\n" + term.c(f"── {f} ", "cyan") + term.c("─" * max(0, 60 - len(f)), "dim"))
        rows = [(k, v, r) for k, vals in t["knobs"].items() for v, r in vals.items()] + \
               [("question", v, r) for v, r in t["kinds"].items()]
        for k, v, r in rows:
            col = "dim" if r["at_chance"] else ("green" if r["lift"] > 0.3 else "yellow")
            say(term.c(f"  {k:>13s} = {str(v):<16s} {r['acc'] * 100:5.1f}% vs {r['floor'] * 100:4.1f}%  "
                       f"lift {r['lift'] * 100:+4.0f}%  n {r['n']:4d}" + ("   at chance" if r["at_chance"] else ""), col))


# ---------------------------------------------------------------- the short version, for charts
def record(its, tab, n, shots) -> dict:
    """A metrics line from a marking: every family weighs the same; at_chance lists each family's levels at chance."""
    fams = {f: {"acc": round(t["all"]["acc"], 4), "floor": round(t["all"]["floor"], 4), "lift": round(t["all"]["lift"], 4),
                "at_chance": bool(t["all"]["at_chance"]), "n": t["all"]["n"]} for f, t in tab.items()}
    flat = {f: [f"{k} {v}" for k, vals in t["knobs"].items() for v, r in vals.items() if r["at_chance"]] +
               [f"asks {v}" for v, r in t["kinds"].items() if r["at_chance"]] for f, t in tab.items()}
    k = max(1, len(fams))
    return {"kind": "field1", "score": sum(x["acc"] for x in fams.values()) / k,
            "floor": sum(x["floor"] for x in fams.values()) / k, "lift": sum(x["lift"] for x in fams.values()) / k,
            "families": fams, "at_chance": flat, "n": n, "shots": shots, "questions": len(its)}


def quick(model, device, n=60, shots=2, pad_ok=False, families=None, budget=16384):
    """The table for a model in hand (the trainer's check): -> (metrics record, table). Leaves the model as it was."""
    from .. import tok
    from . import FAMILIES
    was = model.training
    model.eval()
    try:
        enc = tok.for_model(model)
        max_len = int(getattr(model.cfg, "max_len", 2048) or 2048)
        its, _ = items(list(families or FAMILIES), n, shots, enc, max_len)
        mark(model, its, device, budget=budget, pad_ok=pad_ok, say=None)
        tab = table(its, min_n=max(4, n // 8))
        return record(its, tab, n, shots), tab
    finally:
        if was:
            model.train()


def _last_logged(d: Path):
    """The tokens of the run's newest field1 line (None: none yet)."""
    last = None
    try:
        for line in open(d / "metrics.jsonl"):
            if '"field1"' in line:
                try:
                    x = json.loads(line)
                    if x.get("kind") == "field1":
                        last = x.get("tokens")
                except ValueError:
                    pass
    except OSError:
        pass
    return last


def _log(d: Path, rec):
    with open(d / "metrics.jsonl", "a") as f:     # one short write: safe beside the trainer's own lines
        f.write(json.dumps(rec) + "\n")


def watch(name, runs: Path, device, every=200e6, n=60, shots=2, poll=60, budget=16384, say=print):
    """Examine each new checkpoint of a run as it trains (every `every` tokens), and log it for the dashboard."""
    import torch
    from ..data import run_dir
    from ..train import load_model
    d = run_dir(runs, name)
    ck = d / "ckpt.pt"
    last, seen = _last_logged(d), None
    say(f"watching {ck}: Field1 levels every {every / 1e6:.0f}M tokens ({n} scenes per family)"
        + (f", last check at {last / 1e9:.2f}B" if last else ""))
    while True:
        try:
            mt = ck.stat().st_mtime
        except OSError:
            mt = None
        try:
            st = json.loads((d / "status.json").read_text())
        except Exception:
            st = {}
        done = st.get("state") == "done"
        near = st.get("tokens")                     # the run's tokens now: no need to open a checkpoint too early
        due = last is None or done or near is None or near - last >= every
        if mt is not None and mt != seen and due:
            seen = mt
            try:
                model, meta = load_model(ck, device, mmap=True)
                tokens = meta.get("tokens") or 0
                if last is None or tokens - last >= every or (done and tokens > last):
                    t0 = time.time()
                    rec, tab = quick(model, device, n, shots, pad_ok=False, budget=budget)
                    rec.update(step=meta.get("step"), tokens=tokens, elapsed=meta.get("elapsed"), full=done,
                               by="watcher")
                    _log(d, rec)
                    last = tokens
                    worst = [f for f, x in rec["families"].items() if x["at_chance"]]
                    say(f"[field1] {tokens / 1e9:.2f}B: {rec['score'] * 100:.1f}% (floor {rec['floor'] * 100:.1f}%), "
                        f"at chance: {', '.join(worst) or 'none'} ({time.time() - t0:.0f} s)", flush=True)
                del model
            except Exception as e:                      # a checkpoint half written, a GPU short of memory: next time
                say(f"[field1] skipped this checkpoint ({type(e).__name__}: {str(e)[:160]})", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if done and last is not None and mt == seen:
            say("the run is finished: done watching")
            return
        time.sleep(poll)


# ---------------------------------------------------------------- python3 -m oazy.field1.levels
def run(name, runs: Path, device, n=200, shots=2, families=None, budget=16384, say=print):
    from .. import tok
    from . import FAMILIES
    from ..worldmix import REFS, ref_files
    from ..data import run_dir
    d = run_dir(runs, name)
    if name in REFS:
        from ..compare import smol
        model = smol(name, device)
        pad_ok = True
    else:
        from ..train import load_model
        if not (d / "ckpt.pt").exists():
            raise SystemExit(f"{name}: no checkpoint in {d}")
        model, _ = load_model(d / "ckpt.pt", device, mmap=True)
        model.to(device)
        pad_ok = False
    model.eval()
    if tok.name_of(model) != "smollm2":
        raise SystemExit(f"{name} reads GPT-2 pieces: Field1's table is for SmolLM2-piece models")
    enc = tok.for_model(model)
    max_len = int(getattr(model.cfg, "max_len", 2048) or 2048)
    fams = families or list(FAMILIES)
    t0 = time.time()
    say(f"{name}: making {n} fresh scenes of each of {len(fams)} Field1 families ({shots} solved ones before each)")
    its, unmarked = items(fams, n, shots, enc, max_len)
    say(f"  {len(its):,} questions to mark (longest {max(len(x['ctx']) for x in its)} pieces, model reads {max_len})")
    mark(model, its, device, budget=budget, pad_ok=pad_ok, say=say)
    tab = table(its)
    rec = record(its, tab, n, shots)
    meta = {"run": name, "n": n, "shots": shots, "questions": len(its), "seconds": time.time() - t0,
            "families": fams, "made": time.strftime("%Y-%m-%d %H:%M")}
    d.mkdir(parents=True, exist_ok=True)
    if name in REFS:
        ref_files(d)
    keep = [{k: v for k, v in it.items() if k not in ("ctx", "cands")} for it in its]
    (d / "field1_levels.json").write_text(json.dumps({**meta, "table": tab, "unmarked": unmarked, "items": keep}, default=str))
    (d / "field1_levels.md").write_text(markdown(name, tab, unmarked, meta))
    if name in REFS:                             # a yardstick: its line on the dashboard's Field1 chart
        from ..compare import SMOL_TOKENS
        _log(d, {**rec, "step": 0, "tokens": SMOL_TOKENS.get(name, 0), "elapsed": 0, "full": True})
    else:
        _log(d, {**rec, "step": None, "tokens": _ckpt_tokens(d), "elapsed": None, "full": True, "by": "levels"})
    show_text(tab, say)
    say(f"\nwritten {d / 'field1_levels.md'} ({time.time() - t0:.0f} s)")
    return tab


def _ckpt_tokens(d: Path):
    try:
        st = json.loads((d / "status.json").read_text())
        return st.get("tokens")
    except Exception:
        return None


def show(family, n=3):
    for i in range(n):
        sc = scene(family, i)
        print(f"── {family} {i}  {sc['knobs']}")
        print(sc["story"])
        for q, a, kind, qk in sc["qas"]:
            print(f"   {q} -> {a}   [{kind} {qk}]  choices: {choices(family, q, a, sc['story'], knobs=qk)}")
        print()


def main(argv=None):
    from ..data import default_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", default=["smollm2-360m"])
    ap.add_argument("--runs", default=None)
    ap.add_argument("--n", type=int, default=200, help="scenes per family")
    ap.add_argument("--shots", type=int, default=2, help="solved scenes of the same family before each one")
    ap.add_argument("--family", nargs="+", default=None, help="only these families")
    ap.add_argument("--device", default=None)
    ap.add_argument("--budget", type=int, default=16384, help="pieces read at once (lower it if the GPU is short of memory)")
    ap.add_argument("--watch", default=None, metavar="RUN", help="examine each new checkpoint of a run as it trains")
    ap.add_argument("--every", type=float, default=200e6, help="--watch: tokens between checks")
    ap.add_argument("--show", default=None, metavar="FAMILY", help="print a few scenes, their questions and choices")
    a = ap.parse_args(argv)
    if a.show:
        show(a.show)
        return
    import torch
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from ..data import few_cpu_threads
    few_cpu_threads(dev)
    runs = Path(a.runs or default_dir("runs"))
    if a.watch:
        watch(a.watch, runs, dev, a.every, n=a.n if a.n != 200 else 60, shots=a.shots, budget=a.budget)
        return
    for rn in a.run:
        run(rn, runs, dev, a.n, a.shots, a.family, a.budget)


if __name__ == "__main__":
    main()
