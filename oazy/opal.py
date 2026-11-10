"""Opal: our training set (João, Nov 10). Our text (books, web, maths, code, chat, a little religious text) plus
Field1, the world told in natural speech (oazy/field1/).

    python3 -m oazy.opal show --family minds --n 3     # read some Field1 episodes (no tokenizer needed)
    python3 -m oazy.opal check                          # the walls: none of Real World, Pro or LogT in 20,000 episodes
    nice -n 10 python3 -m oazy.opal build               # write Field1 into Opal (CPU pod; ~2.5B tokens)
    python3 -m oazy.opal link                           # only name the folder: <data>/sado becomes <data>/opal
    bash runpod/train.sh bee6 --preset big --tokens 10e9 --micro 32 --data-sub opal --mentor none --name bee6-opal-10b

On disk. Opal lives in <data>/opal. The text part was built by oazy/sado.py under its old name, so `build` and `link`
rename <data>/sado to <data>/opal and leave a link named sado behind: runs started before the rename (Emerald's,
bee6moe-emerald-10b) find their text where they left it. Field1 goes to <data>/opal/field1/ (train.bin, val.bin,
meta.json). Emerald's old world text stays in <data>/opal/world/ for those older runs only.

How much. --data-sub opal reads Field1 at 20% of every phase (cool-down included), as the reference run
bee6-w20-emerald-10b read the old world; --opal-plan changes it ("0.30:0.1,0.15", "flat:0.10"). A 10B run at 20% reads
2B tokens of Field1; the default build writes 2.5B, all different (every episode is made fresh from its own seed).

The exams the trainer already takes on world runs (secret, practised, Real World, Real World Pro) run on Opal too.
Field1 never uses the world's old one-line wording, so on Opal the "practised" exam is also a test of other wording;
Real World and Real World Pro are the ones to watch.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

VAL_SEED, TRAIN_SEED = 7, 0                 # held-out episodes come from their own seed
DEFAULT_TOKENS = 2.5e9
PER_JOB = 512                               # episodes per helper job


def opal_dir(data: Path, rename=True) -> Path:
    """<data>/opal, renamed from <data>/sado the first time (a link named sado is left for older runs)."""
    data = Path(data)
    opal, sado = data / "opal", data / "sado"
    if opal.exists():
        if not sado.exists() and not sado.is_symlink():
            sado.symlink_to("opal")
        return opal
    if sado.is_dir() and not sado.is_symlink() and rename:
        sado.rename(opal)
        sado.symlink_to("opal")
        print(f"renamed {sado} -> {opal} (a link named sado stays, for the runs started before)")
        return opal
    if not rename:
        return sado if sado.exists() else opal
    opal.mkdir(parents=True, exist_ok=True)
    return opal


# ---------------------------------------------------------------- one helper job: episodes -> pieces
_ENC = None


def _init(tk):
    global _ENC
    from . import tok
    _ENC = tok.get(tk)


def job(args):
    """PER_JOB Field1 episodes of job k, as SmolLM2 pieces, each followed by <|endoftext|>. -> (pieces, tries by family)."""
    seed, k, tk = args
    from . import field1 as F
    if _ENC is None:
        _init(tk)
    fams = F.pick(seed, k, PER_JOB)
    texts, tries = [], {}
    for i, f in enumerate(fams):
        t, n = F.episode(f, (seed, k, i))
        texts.append(t)
        tries[f] = tries.get(f, 0) + (n - 1)
    ids = _ENC.encode_ordinary_batch(texts)
    eot = _ENC.eot_token
    flat = []
    for x in ids:
        flat += x
        flat.append(eot)
    return np.asarray(flat, dtype=np.uint16), tries, len(texts)


def build(data: Path, tokens: float = DEFAULT_TOKENS, val_tokens: float = 2e6, workers: int = 0, tk: str = "smollm2"):
    """Write <data>/opal/field1/{val,train}.bin and meta.json. A finished build of this Field1 version is left alone."""
    from multiprocessing import get_context
    from . import term, field1 as F
    from .sado import cpu_limit
    opal = opal_dir(data)
    out = opal / "field1"
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if old.get("field1") == F.VERSION and old.get("train_tokens", 0) >= tokens:
            print(f"Field1 is already built in {out}: {old['train_tokens'] / 1e9:.2f}B tokens (Field1 v{F.VERSION})")
            return out
    workers = workers or max(1, cpu_limit() - 1)
    t0 = time.time()
    sizes, tries, eps = {}, {}, 0
    print(f"building Field1 v{F.VERSION} into {out}: {term.tokens(int(tokens))} tokens, {workers} workers, "
          f"{len(F.FAMILIES)} families")
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(tk,)) as pool:
        for split, seed, want in (("val", VAL_SEED, val_tokens), ("train", TRAIN_SEED, tokens)):
            n, k = 0, 0
            with open(out / f"{split}.bin.tmp", "wb") as f:
                while n < want:
                    jobs = [(seed, k + i, tk) for i in range(4 * workers)]
                    k += len(jobs)
                    for arr, tr, ne in pool.imap(job, jobs):
                        f.write(arr.tobytes())
                        n += arr.size
                        eps += ne
                        for a, b in tr.items():
                            tries[a] = tries.get(a, 0) + b
                        if n >= want:
                            break
                    if split == "train":
                        dt = max(time.time() - t0, 1e-6)
                        term.out(f"  {term.bar(min(n / want, 1), 24, 'blue')} {min(n / want, 1) * 100:5.1f}%  "
                                 f"{term.tokens(n)} / {term.tokens(int(want))} tokens  "
                                 + term.c(f"· {term.rate(n / dt)} · {term.duration((want - n) / max(n / dt, 1))} left", "dim"),
                                 flush=True)
            (out / f"{split}.bin.tmp").replace(out / f"{split}.bin")
            sizes[split] = n
    meta = {"dataset": "Field1: the world told in natural speech (oazy/field1/), part of Opal", "tokenizer": tk,
            "field1": F.VERSION, "families": dict(F.WEIGHTS), "episodes": eps,
            "train_tokens": sizes["train"], "val_tokens": sizes["val"], "train_seed": TRAIN_SEED, "val_seed": VAL_SEED,
            "remade_for_walls": tries, "built": time.strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - t0) / 60, 1)}
    mp.write_text(json.dumps(meta, indent=1))
    try:
        from .sado import token_ratios
        token_ratios(opal, names=("field1",))
    except Exception as e:
        print(f"  (token ratio not written: {type(e).__name__}: {str(e)[:100]})")
    print(term.c("ok", "green") + f" {out}: {sizes['train'] / 1e9:.2f}B training tokens of Field1, {sizes['val'] / 1e6:.1f}M held out, "
          f"{eps:,} episodes, {meta['minutes']:.0f} min. Runs read Opal with --data-sub opal.")
    return out


def check(n=20_000, say=print):
    """The walls on n fresh episodes (every family): nothing of Real World, Pro or LogT. -> {leak: count}."""
    from . import field1 as F
    from .field1 import core, guard
    import random
    seen, made = {}, 0
    for i in range(n):
        f = F.FAMILIES[i % len(F.FAMILIES)]
        t, _ = F.episode(f, ("check", i))
        made += 1
        for x in guard.leaks(t):
            seen[x] = seen.get(x, 0) + 1
    # and the first try of each, before any remake: how often the walls are needed at all
    first = 0
    for i in range(2000):
        f = F.FAMILIES[i % len(F.FAMILIES)]
        r = random.Random(f"field1|{F.VERSION}|{f}|('check', {i})|0")
        st, qas = F.GEN[f](r)
        first += bool(guard.leaks(core.frame(r, st.tell(), qas)))
    say(f"{made:,} Field1 episodes checked: {sum(seen.values())} say anything held back "
        f"(first tries that needed a remake: {first} of 2,000)")
    return seen


def main():
    from .data import default_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["build", "show", "check", "link"])
    ap.add_argument("--data", default=None, help="the data folder (Opal is under it); default $OAZY_DATA or the lab's")
    ap.add_argument("--tokens", type=float, default=DEFAULT_TOKENS, help="training tokens of Field1 (2.5e9 = enough for 20%% of 10B, and more)")
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--workers", type=int, default=0, help="processes making and cutting the text (default: every core but one)")
    ap.add_argument("--family", default=None, help="show: only this family")
    ap.add_argument("--n", type=int, default=2, help="show: episodes per family")
    ap.add_argument("--episodes", type=int, default=20_000, help="check: episodes in all")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data or default_dir("data"))
    if a.what == "show":
        from . import field1 as F
        from . import term
        for f, t in F.show(a.family, a.n, a.seed):
            print(term.c(f"── {f} ", "cyan") + term.c("─" * max(0, 60 - len(f)), "dim"))
            print(t + "\n")
    elif a.what == "check":
        found = check(a.episodes)
        for k, v in sorted(found.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {v:6d}  {k}")
        raise SystemExit(1 if found else 0)
    elif a.what == "link":
        print(opal_dir(data))
    else:
        build(data, a.tokens, a.val_tokens, a.workers)


if __name__ == "__main__":
    main()
