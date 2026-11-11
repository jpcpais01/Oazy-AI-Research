"""Opal2: Opal taught to think (João, Nov 11: "the best shot we have right now").

Opal is our text (books, web, maths, code, chat, a little religious text) plus Field1, the world told in natural
speech. Opal2 keeps all of it and adds three new parts, each aimed at the same goal: a model that writes clean English
and tries to think its way to an answer.

  proc    the logic warm-up (oazy/proc.py): brackets, automata, sorting, stacks, sums. Pure structure, no words.
          Read alone for the first 0.2% of the run, then 0.5% until 10%. ~30M pieces, CPU, a minute.
  work    Field1 worked out (oazy/field1/work.py): every scene's answer reached step by step by the simulator itself
          (the clues gathered, the events put in order, what is true after each one), so the reasoning is never wrong.
          8% of the run. 1.0B pieces, CPU, ~5 minutes.
  think   text that shows its thinking (oazy/think.py): Opal passages followed by the thinking a careful reader does,
          written by an open model. 12% of the run. 400M pieces, GPU, ~2-3 hours on an H200 (or overnight on the 5090).

and Field1 itself drops from 20% to 7%: the worked part now carries most of the world, and the 20% runs showed that
more world text did not move the English tests (bee6-w20 HellaSwag ~ Bee6's).

The mix over the run (oazy/mix.py OPAL2_MIX; the text part keeps Opal's own proportions inside its share):

      0 ... 0.2%    proc 100%                                  the warm-up: structure before words
   0.2% ... 10%     text 72.5%, think 12%, work 8%, field1 7%, proc 0.5%
    10% ... 70%     text 73%,   think 12%, work 8%, field1 7%
    70% ... 100%    the cool-down: Opal's cool-down text (more chat and maths) 73%, think 12%, work 8%, field1 7%

A part that is not built is skipped and the rest grows to fill its share, so a run can start before `think` is
written (it will say so at the start).

    nice -n 10 python3 -m oazy.opal2 build                 # CPU pod: proc + work into <data>/opal/ (Field1 as before)
    python3 -m oazy.think --tokens 400e6                    # a GPU: the thinking part (see oazy/think.py)
    python3 -m oazy.opal2 show --part work --family lines   # read some
    python3 -m oazy.opal2 check                             # the walls on worked episodes and questions
    bash runpod/train.sh gnosis2 --preset big --tokens 10e9 --micro 32 --data-sub opal2 --mentor none --name gnosis2-10b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PER_JOB = 256
PARTS = {"proc": dict(folder="proc", tokens=30e6, val=0.5e6),
         "work": dict(folder="field1work", tokens=1.0e9, val=2e6)}
VAL_SEED, TRAIN_SEED = 7, 0

_ENC = None


def _init(tk):
    global _ENC
    from . import tok
    _ENC = tok.get(tk)


def _texts(part, seed, k):
    if part == "proc":
        from . import proc
        return proc.texts(seed, k, PER_JOB), {}
    from . import field1 as F
    from .field1 import work
    fams = F.pick(("work", seed), k, PER_JOB)
    out, tries = [], {}
    for i, f in enumerate(fams):
        t, n = work.episode(f, (seed, k, i))
        out.append(t)
        tries[f] = tries.get(f, 0) + (n - 1)
    return out, tries


def job(args):
    part, seed, k, tk = args
    if _ENC is None:
        _init(tk)
    texts, tries = _texts(part, seed, k)
    flat = []
    for x in _ENC.encode_ordinary_batch(texts):
        flat += x
        flat.append(_ENC.eot_token)
    return np.asarray(flat, dtype=np.uint16), tries, len(texts)


def version(part):
    if part == "proc":
        from . import proc
        return proc.VERSION
    from .field1 import work
    from . import field1 as F
    return f"{F.VERSION}.{work.VERSION}"


def build_part(data: Path, part: str, tokens=None, val_tokens=None, workers=0, tk="smollm2", say=print):
    """Write <data>/opal/<folder>/{val,train}.bin and meta.json. A finished build of this version is left alone."""
    from multiprocessing import get_context
    from . import term
    from .opal import opal_dir
    from .sado import cpu_limit
    spec = PARTS[part]
    tokens, val_tokens = tokens or spec["tokens"], val_tokens or spec["val"]
    out = opal_dir(Path(data)) / spec["folder"]
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if old.get("version") == version(part) and old.get("train_tokens", 0) >= tokens:
            say(f"[opal2] {part} is already built in {out}: {old['train_tokens'] / 1e6:.0f}M pieces")
            return out
    workers = workers or max(1, cpu_limit() - 1)
    t0, sizes, tries, docs = time.time(), {}, {}, 0
    say(f"[opal2] building {part} into {out}: {term.tokens(int(tokens))} pieces, {workers} workers")
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(tk,)) as pool:
        for split, seed, want in (("val", VAL_SEED, val_tokens), ("train", TRAIN_SEED, tokens)):
            n, k = 0, 0
            with open(out / f"{split}.bin.tmp", "wb") as f:
                while n < want:
                    jobs = [(part, seed, k + i, tk) for i in range(4 * workers)]
                    k += len(jobs)
                    for arr, tr, nd in pool.imap(job, jobs):
                        f.write(arr.tobytes())
                        n += arr.size
                        docs += nd
                        for a, b in tr.items():
                            tries[a] = tries.get(a, 0) + b
                        if n >= want:
                            break
                    if split == "train":
                        dt = max(time.time() - t0, 1e-6)
                        term.out(f"  {term.bar(min(n / want, 1), 24, 'blue')} {min(n / want, 1) * 100:5.1f}%  "
                                 f"{term.tokens(n)} / {term.tokens(int(want))} pieces  "
                                 + term.c(f"· {term.rate(n / dt)} · {term.duration((want - n) / max(n / dt, 1))} left", "dim"),
                                 flush=True)
            (out / f"{split}.bin.tmp").replace(out / f"{split}.bin")
            sizes[split] = n
    meta = {"dataset": f"Opal2 {part} (oazy/opal2.py)", "tokenizer": tk, "version": version(part), "documents": docs,
            "train_tokens": sizes["train"], "val_tokens": sizes["val"], "remade_for_walls": tries,
            "built": time.strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - t0) / 60, 1)}
    mp.write_text(json.dumps(meta, indent=1))
    say(term.c("ok", "green") + f" {out}: {sizes['train'] / 1e6:.0f}M training pieces of {part}, {docs:,} documents, "
        f"{meta['minutes']:.0f} min")
    return out


def build(data: Path, parts=("proc", "work"), workers=0, work_tokens=None, say=print):
    from .opal import opal_dir
    opal = opal_dir(Path(data))
    if not (opal / "field1" / "train.bin").exists():
        say("[opal2] Field1 is not built yet: building it first (python3 -m oazy.opal build)")
        from .opal import build as build_field1
        build_field1(Path(data), workers=workers)
    for p in parts:
        build_part(data, p, tokens=work_tokens if p == "work" else None, workers=workers, say=say)
    think = opal / "think" / "train.bin"
    say("[opal2] " + (f"think is built ({think})" if think.exists() else
                      "think is not built yet: on a GPU run  python3 -m oazy.think --tokens 400e6  (a run without it reads "
                      "the other parts a little more)"))


def check(n=5000, say=print):
    """The walls on n worked episodes and n reasoning questions (every family); and every question grades its own
    worked answer as right. -> (leaks, wrong)"""
    from . import field1 as F
    from .field1 import guard, work
    seen, wrong = {}, 0
    for i in range(n):
        f = F.FAMILIES[i % len(F.FAMILIES)]
        t, _ = work.episode(f, ("check", i))
        q = work.question(f, ("check", i))
        for x in guard.leaks(t) + guard.leaks(q["prompt"] + " " + q["work"]):
            seen[x] = seen.get(x, 0) + 1
        wrong += work.grade(q["work"], q["answer"]) < 1
    say(f"{n:,} worked episodes and {n:,} questions checked: {sum(seen.values())} say anything held back, "
        f"{wrong} questions whose own worked answer does not grade as right")
    return seen, wrong


def show(part, family=None, n=2, seed=0, data=None):
    from . import term
    if part == "proc":
        from . import proc
        for i in range(n):
            print(term.c(f"── proc {i} ", "cyan") + term.c("─" * 50, "dim"))
            print(proc.document((seed, i)) + "\n")
    elif part == "work":
        from . import field1 as F
        from .field1 import work
        for f in ([family] if family else F.FAMILIES):
            for i in range(n):
                print(term.c(f"── {f} ", "cyan") + term.c("─" * max(0, 60 - len(f)), "dim"))
                print(work.episode(f, (seed, i))[0] + "\n")
    elif part == "ask":
        from . import field1 as F
        from .field1 import work
        for f in ([family] if family else F.FAMILIES):
            q = work.question(f, (seed, 0))
            print(term.c(f"── {f} (answer: {q['answer']}) ", "cyan") + term.c("─" * 30, "dim"))
            print(q["prompt"] + term.c(q["work"], "dim") + "\n")
    else:
        from .think import think_dir
        path = think_dir(Path(data)) / "texts.jsonl"
        if not path.exists():
            raise SystemExit(f"nothing written yet in {path}")
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                d = json.loads(line)
                print(term.c(f"── think {d['i']} ({d['src']}) ", "cyan") + term.c("─" * 40, "dim"))
                print(d["text"] + "\n")


def main():
    from .data import default_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["build", "show", "check"])
    ap.add_argument("--part", default=None, help="build: proc or work (default both); show: proc, work, ask or think")
    ap.add_argument("--data", default=None, help="the data folder (Opal is under it); default $OAZY_DATA or the lab's")
    ap.add_argument("--work-tokens", type=float, default=None, help="pieces of worked Field1 (default 1.0e9)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--family", default=None)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5000, help="check: episodes and questions each")
    a = ap.parse_args()
    data = Path(a.data or default_dir("data"))
    if a.what == "show":
        show(a.part or "work", a.family, a.n, a.seed, data)
    elif a.what == "check":
        seen, wrong = check(a.episodes)
        for k, v in sorted(seen.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {v:6d}  {k}")
        raise SystemExit(1 if (seen or wrong) else 0)
    else:
        build(data, (a.part,) if a.part else ("proc", "work"), a.workers, a.work_tokens)


if __name__ == "__main__":
    main()
