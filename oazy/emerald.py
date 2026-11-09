"""Emerald: our third dataset. It is Sado with the world inside it (João, Nov 9).

    python3 -m oazy.emerald build                         # write the world's text into Sado (<data>/sado/world/), ~2.5B tokens
    bash runpod/train.sh <model> ... --data-sub emerald   # a run that reads Emerald: Sado + 10% world
    bash runpod/train.sh <model> ... --data-sub emerald --emerald 0.05   # the same with 5% world (0.20 = 20%, 0 = plain Sado)
    python3 -m oazy.worldmix exam --run <run> ...         # the world's secret exams, asked as plain text

What it adds. Sado's text stays exactly as it is. The new part is the world from worldfirst/world.py (World 2: fifteen
families, sums, lines, boxes, family, falling, machines, cups, minds, walks, carry, time, cause, wants, count and the
mixed "house"), each episode a scene, an action, questions about the
scene after it, and their answers, as plain sentences ("There is a line. Vasco is right behind Cora. ... Is Vasco ahead
of Simao? Answer: no."). Only practice levels, with fresh random names; the secret exams are never written into it, so
`oazy.worldmix exam` still asks only what the model never read. A model reading Emerald is still a plain next-word model.

How it differs from `--world` (oazy/worldmix.py): --world makes the episodes on the fly with helper processes during the
run; Emerald writes them once into a file, so a run reads the world like any other part of the dataset (no helpers, the
same reader, the held-out loss of the world shows up next to the other sources'), and every model of every size can use
it with the same one flag.

How much (the `emerald` mix, oazy/mix.py): the world gets 3x its average share for the first 20% of the run, then half
of it to the end, cool-down included, as --world does. Average 10% -> 30% for the first 20%, then 5%. The other sources
keep their Sado proportions. A 20B run at 10% reads 2B tokens of world; the default build writes 2.5B (no rereading).

Build time: the episodes cost almost nothing; cutting them into SmolLM2's pieces is the work, on every CPU core but one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

VAL_SEED, TRAIN_SEED = 7, 0          # held-out episodes come from their own seed, never the training one
DEFAULT_TOKENS = 2.5e9


def build(data: Path, tokens: float = DEFAULT_TOKENS, val_tokens: float = 2e6, workers: int = 0, tk: str = "smollm2"):
    """Write <data>/sado/world/{val,train}.bin (+ meta.json). A finished build is left alone; a bigger --tokens restarts it."""
    import os
    from multiprocessing import get_context
    from . import term, worldmix
    out = data / "sado" / "world"
    out.mkdir(parents=True, exist_ok=True)
    from worldfirst import world as W
    W.use(W.LATEST)
    mp = out / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if old.get("world", 1) != W.VERSION:            # built from an older world: built again from this one
            print(f"Emerald's world in {out} was built from World {old.get('world', 1)}: building it again from World {W.VERSION}")
        elif old.get("train_tokens", 0) >= tokens:
            print(f"Emerald's world is already built in {out}: {old['train_tokens'] / 1e9:.2f}B tokens of World {W.VERSION}")
            if not old.get("marks"):
                add_marks(data, workers, tk)
            return out
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    sizes = {}
    with get_context("spawn").Pool(workers, initializer=worldmix._init, initargs=(tk, W.VERSION)) as pool:
        for split, seed, want in (("val", VAL_SEED, val_tokens), ("train", TRAIN_SEED, tokens)):
            n, k = 0, 0
            with open(out / f"{split}.bin.tmp", "wb") as f, open(out / f"{split}.marks.tmp", "wb") as fm:
                while n < want:
                    jobs = [(seed, k + i, tk) for i in range(4 * workers)]
                    k += len(jobs)
                    for arr, mk in pool.imap(worldmix.job_marks, jobs):
                        f.write(arr.astype(np.uint16).tobytes())          # whole episodes, each ending with <|endoftext|>
                        fm.write(mk.tobytes())                            # one byte per piece: the thing it names (255: none)
                        n += arr.size
                        if n >= want:
                            break
                    if split == "train":
                        dt = max(time.time() - t0, 1e-6)
                        term.out(f"  {term.bar(min(n / want, 1), 24, 'blue')} {min(n / want, 1) * 100:5.1f}%  "
                                 f"{term.tokens(n)} / {term.tokens(int(want))} tokens  "
                                 + term.c(f"· {term.rate(n / dt)} · {term.duration((want - n) / max(n / dt, 1))} left", "dim"),
                                 flush=True)
            (out / f"{split}.bin.tmp").replace(out / f"{split}.bin")
            (out / f"{split}.marks.tmp").replace(out / f"{split}.marks")
            sizes[split] = n
    meta = {"dataset": "world: practice episodes of worldfirst/world.py as plain text (oazy/emerald.py)", "tokenizer": tk,
            "world": W.VERSION, "families": list(W.FAMILIES),
            "train_tokens": sizes["train"], "val_tokens": sizes["val"], "train_seed": TRAIN_SEED, "val_seed": VAL_SEED, "marks": True,
            "built": time.strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - t0) / 60, 1)}
    mp.write_text(json.dumps(meta, indent=1))
    try:
        from .sado import token_ratios
        token_ratios(data / "sado", names=("world",))
    except Exception as e:
        print(f"  (token ratio not written: {type(e).__name__}: {str(e)[:100]})")
    print(f"{out}: {sizes['train'] / 1e9:.2f}B training tokens of World {W.VERSION}, {sizes['val'] / 1e6:.1f}M held out, "
          f"{meta['minutes']:.0f} min. Runs read it with --data-sub emerald.")
    return out


def add_marks(data: Path, workers: int = 0, tk: str = "smollm2"):
    """An Emerald built before the marks: make the same episodes again (the jobs are fixed by their seeds), check that
    their pieces are the very ones on disk, and write only the marks beside them. Only Logos1-200M reads the marks (as
    its linker's answer key); every other model reads the same pieces as before."""
    import os
    from multiprocessing import get_context
    from . import term, worldmix
    from worldfirst import world as W
    out = data / "sado" / "world"
    meta = json.loads((out / "meta.json").read_text())
    W.use(meta.get("world", 1))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    print(f"adding the world's marks to {out} (the same episodes again; the pieces are checked against the file)")
    with get_context("spawn").Pool(workers, initializer=worldmix._init, initargs=(tk, W.VERSION)) as pool:
        for split, seed in (("val", VAL_SEED), ("train", TRAIN_SEED)):
            have = np.memmap(out / f"{split}.bin", dtype=np.uint16, mode="r")
            n, k = 0, 0
            with open(out / f"{split}.marks.tmp", "wb") as fm:
                while n < have.size:
                    jobs = [(seed, k + i, tk) for i in range(4 * workers)]
                    k += len(jobs)
                    for arr, mk in pool.imap(worldmix.job_marks, jobs):
                        m = min(arr.size, have.size - n)
                        if not np.array_equal(arr[:m].astype(np.uint16), have[n:n + m]):
                            raise SystemExit(f"{split}: the episodes made again differ from the file at piece {n:,}; "
                                             f"build Emerald again instead (delete {out} and run: python3 -m oazy.emerald build)")
                        fm.write(mk[:m].tobytes())
                        n += m
                        if n >= have.size:
                            break
                    if split == "train":
                        dt = max(time.time() - t0, 1e-6)
                        term.out(f"  {term.bar(n / have.size, 24, 'blue')} {n / have.size * 100:5.1f}%  "
                                 + term.c(f"· {term.rate(n / dt)} · {term.duration((have.size - n) / max(n / dt, 1))} left", "dim"),
                                 flush=True)
            (out / f"{split}.marks.tmp").replace(out / f"{split}.marks")
    meta["marks"] = True
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"marks written beside {out}/train.bin and val.bin ({(time.time() - t0) / 60:.0f} min)")


def main():
    from .data import default_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["build", "marks"], help="build: write the world (with its marks); "
                    "marks: add only the marks to a world built before them")
    ap.add_argument("--data", default=None, help="the data folder (Sado is under it); default $OAZY_DATA or the lab's")
    ap.add_argument("--tokens", type=float, default=DEFAULT_TOKENS, help="training tokens of world (2.5e9 = enough for 10%% of 20B)")
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--workers", type=int, default=0, help="processes cutting the text (default: every core but one)")
    a = ap.parse_args()
    if a.what == "marks":
        add_marks(Path(a.data or default_dir("data")), a.workers)
    else:
        build(Path(a.data or default_dir("data")), a.tokens, a.val_tokens, a.workers)


if __name__ == "__main__":
    main()
