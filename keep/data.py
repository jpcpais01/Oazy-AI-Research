"""Keep's reading matter: endless streams where something is told, other text comes between, and only then is it asked.

A stream is one long run of tokens, made as it is read:
    <|endoftext|> a scene, told once                                      (a ledger of plain facts, or a Field1 story)
    <|endoftext|> ordinary text from Opal's text part, for G tokens       (G = 0 for a quarter of the scenes, else 32..gap_max)
    <|endoftext|> the scene's questions, each with its answer             ("Where does Rui keep the coin? Answer: the cellar.")
    <|endoftext|> more ordinary text ...
Two kinds of scene, half and half:
    ledger   keep/ledger.py: plain facts. Answering needs no reasoning, so a miss after a gap is a fact that was not kept.
    Field1   oazy/field1: a small simulated world told as a story. Answering needs reasoning too: the harder row.
One scene is open at a time, so "Rui" in a question means the Rui of the one scene still waiting for its questions.
Each row of a batch is its own stream and goes on from window to window: a model with a state can carry it.

The exam (exam_set) is the same thing with the gap fixed: story, exactly G tokens of text, then the questions. A
question is marked by which of its choices the model scores highest as the next piece, at the first place where the
choices part ("the cellar" / "the attic": after "the"). Its scenes come from seeds no training stream uses.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

EXAM_SEED = "keep-exam"
TRAIN_SEED = "keep-train"


# ---------------------------------------------------------------- ordinary text
class Text:
    """Opal's text part as flat token files (oazy/mix.py's sources that are there). span(): n tokens from a random place."""

    def __init__(self, data: Path, split: str = "train", log=print):
        from oazy.mix import OPAL_TEXT_MIX, parse_mix, resolve_mix, source_path, WORLD_SOURCES
        names = sorted({k for _, w in parse_mix(resolve_mix(OPAL_TEXT_MIX)) for k in w if k not in WORLD_SOURCES})
        self.parts = []
        for n in names:
            p = source_path(Path(data), n, split)
            if p.exists() and p.stat().st_size > 8192:
                self.parts.append(np.memmap(p, dtype=np.uint16, mode="r"))
        if not self.parts and split != "train":                     # no held-out file: the tail of the training text
            t = Text(data, "train", log)
            self.parts = [p[-max(4096, len(p) // 100):] for p in t.parts]
        if not self.parts:
            raise SystemExit(f"no text found under {data} (looked for {', '.join(names)}): pass --data and --sub, e.g. --data /workspace/data --sub opal")
        self.sizes = np.array([len(p) for p in self.parts], dtype=np.float64)
        self.p = self.sizes / self.sizes.sum()

    def span(self, rng: random.Random, n: int):
        part = self.parts[rng.choices(range(len(self.parts)), weights=self.p)[0]]
        n = min(n, len(part) - 1)
        i = rng.randrange(0, len(part) - n)
        return part[i:i + n].astype(np.int64).tolist()


# ---------------------------------------------------------------- scenes
LEDGER = "ledger"
P_LEDGER = 0.5
TEXT, STORY, ASKED = 0, 1, 2


def _families():
    from oazy.field1 import FAMILIES, WEIGHTS
    return list(FAMILIES), [WEIGHTS[f] for f in FAMILIES]


def pick(rng: random.Random, fams, wts, p_ledger=P_LEDGER) -> str:
    return LEDGER if rng.random() < p_ledger else rng.choices(fams, weights=wts)[0]


def scene(family: str, i, seed) -> dict:
    """-> {"story", "lines": [(lead, answer, choices or None, kind)]}: lead + " " + answer + "." is a line of the block."""
    if family == LEDGER:
        from . import ledger
        return ledger.scene(i, seed)
    from oazy.field1 import levels
    sc = levels.scene(family, i, seed)
    return {"story": sc["story"], "lines": [(f"{q} Answer:", a, levels.choices(family, q, a, sc["story"], knobs=qk), family)
                                            for q, a, _, qk in sc["qas"]]}


def scene_parts(enc, family: str, i, seed):
    """A scene -> (story tokens, question-block tokens, the scene)."""
    sc = scene(family, i, seed)
    story = enc.encode_ordinary(sc["story"])
    block = enc.encode_ordinary("\n".join(f"{lead} {a}." for lead, a, *_ in sc["lines"]))
    return story, block, sc


def filler(rng: random.Random, text: Text, eot: int, n: int):
    """Exactly n tokens of ordinary text, in pieces of 200..800 as a stream has them, each behind an <|endoftext|>."""
    out = []
    while len(out) < n:
        m = min(rng.randint(200, 800), n - len(out))
        out += [eot] + text.span(rng, m - 1) if m > 1 else [eot]
    return out[:n]


def stream(rng: random.Random, text: Text, enc, gap_max: int, p_scene: float = 0.6, p_ledger: float = P_LEDGER, tag=0):
    """Yields (tokens, what they are), for ever: scenes, text between, questions after the gap.
    What they are: TEXT, STORY or ASKED (the question block)."""
    eot = enc.eot_token
    fams, wts = _families()
    n, wait, block, left = 0, False, None, 0
    while True:
        if wait and left <= 0:
            yield [eot] + block, ASKED
            wait = False
        elif not wait and rng.random() < p_scene:
            story, block, _ = scene_parts(enc, pick(rng, fams, wts, p_ledger), n, (TRAIN_SEED, tag))
            n += 1
            yield [eot] + story, STORY
            left = 0 if rng.random() < 0.25 else int(math.exp(rng.uniform(math.log(32), math.log(max(33, gap_max)))))
            wait = True
        else:
            sp = filler(rng, text, eot, min(rng.randint(200, 800), left) if wait else rng.randint(200, 800))
            left -= len(sp)
            yield sp, TEXT


class Rows:
    """`rows` streams cut into windows: next() -> tokens (rows, T + 1) int64 and what each is (rows, T + 1) uint8 (TEXT,
    STORY, ASKED). Each window is one token past the last (x = [:, :-1], y = [:, 1:]), each row going on where it stopped."""

    def __init__(self, text: Text, enc, rows: int, T: int, gap_max: int, seed: int, first_row: int = 0, **how):
        self.T = T
        self.gens = [stream(random.Random(f"{seed}|{first_row + r}"), text, enc, gap_max, tag=first_row + r, **how) for r in range(rows)]
        self.buf = [([], []) for _ in range(rows)]

    def next(self):
        out = np.empty((len(self.gens), self.T + 1), dtype=np.int64)
        what = np.empty((len(self.gens), self.T + 1), dtype=np.uint8)
        for r, g in enumerate(self.gens):
            b, k = self.buf[r]
            while len(b) < self.T + 1:
                t, kind = next(g)
                b.extend(t)
                k.extend([kind] * len(t))
            out[r], what[r] = b[:self.T + 1], k[:self.T + 1]
            self.buf[r] = (b[self.T:], k[self.T:])
        return out, what


def _worker(q, data, split, rows, T, gap_max, seed, first_row, how):
    from oazy import tok
    enc = tok.get("smollm2")
    r = Rows(Text(Path(data), split, log=lambda *a: None), enc, rows, T, gap_max, seed, first_row, **how)
    while True:
        q.put(r.next())


class Feed:
    """Batches of windows from `workers` processes (0 = made here). Row r is always the same stream.
    how: stream()'s p_scene (how often a scene starts when none is open) and p_ledger (the ledger's share of the scenes)."""

    def __init__(self, data: Path, enc, batch: int, T: int, gap_max: int, seed: int = 0, workers: int = 8, split="train", **how):
        self.procs, self.qs, self.local = [], [], None
        workers = min(workers, batch)
        if workers <= 0:
            self.local = Rows(Text(data, split), enc, batch, T, gap_max, seed, **how)
            return
        Text(data, split)                                # fail here, with the message, not in a worker
        per = [batch // workers + (1 if i < batch % workers else 0) for i in range(workers)]
        ctx = mp.get_context("fork")
        first = 0
        for n in per:
            q = ctx.Queue(maxsize=6)
            p = ctx.Process(target=_worker, args=(q, str(data), split, n, T, gap_max, seed, first, how), daemon=True)
            p.start()
            self.procs.append(p)
            self.qs.append(q)
            first += n

    def next(self):
        if self.local is not None:
            return self.local.next()
        got = [q.get() for q in self.qs]
        return np.concatenate([g[0] for g in got], 0), np.concatenate([g[1] for g in got], 0)

    def close(self):
        for p in self.procs:
            p.terminate()


# ---------------------------------------------------------------- the exam
def group(kind: str) -> str:
    """The report's rows: "facts" (the ledger), "changed" (ledger facts that were changed), "field1" (every Field1 family)."""
    return kind if kind in ("facts", "changed") else "field1"


def _mark(enc, lead_len, ans, choices, answer):
    """Where the choices part and which pieces stand there -> (pieces into the answer, candidate pieces, which is right),
    or None when the choices cannot be told apart on one piece (or the answer is not said as its choice is)."""
    if not choices or answer not in choices:
        return None
    toks = [enc.encode_ordinary(" " + c) for c in choices]
    k = 0
    while all(len(t) > k for t in toks) and len({t[k] for t in toks}) == 1:
        k += 1
    if any(len(t) <= k for t in toks):
        return None
    first = [t[k] for t in toks]
    right = choices.index(answer)
    if len(set(first)) != len(first) or ans[:k + 1] != toks[right][:k + 1]:
        return None
    return k, first, right


def exam_set(enc, text: Text, gap: int, n: int, q_max: int = 320, seed: int = 0):
    """n scenes (ledger and Field1 in turn) with exactly `gap` tokens of text between the story and its questions.
    -> prefix (n, P) tokens [text to pad on the left, <eot> story, <eot> text x gap], block (n, Q) [<eot> questions,
       then <eot> padding], marks: list of (scene row, place in the block whose next piece is asked, candidate pieces,
       which is right, kind), spans: (n, 2) where each story starts and ends in its prefix row."""
    eot = enc.eot_token
    rng = random.Random(f"{EXAM_SEED}|{seed}|{gap}")
    which = random.Random(f"{EXAM_SEED}|{seed}")                     # the same scenes at every gap
    fams, wts = _families()
    pre, blocks, marks, lens = [], [], [], []
    for r in range(n):
        f = LEDGER if r % 2 == 0 else which.choices(fams, weights=wts)[0]
        sc = scene(f, r, (EXAM_SEED, seed))
        story = [eot] + enc.encode_ordinary(sc["story"])
        lens.append(len(story))
        pre.append(story + filler(rng, text, eot, gap))
        b = [eot]
        for j, (lead, a, ch, kind) in enumerate(sc["lines"]):
            ask = enc.encode_ordinary(("\n" if j else "") + lead)
            ans = enc.encode_ordinary(f" {a}.")
            if len(b) + len(ask) + len(ans) > q_max:
                break
            b += ask
            m = _mark(enc, len(b), ans, ch, a)
            if m is not None:
                marks.append((r, len(b) - 1 + m[0], m[1], m[2], kind))
            b += ans
        blocks.append(b)
    P = max(len(p) for p in pre)
    prefix = np.empty((n, P), dtype=np.int64)
    spans = np.empty((n, 2), dtype=np.int64)
    for r, p in enumerate(pre):
        pad = P - len(p)
        prefix[r] = filler(rng, text, eot, pad) + p
        spans[r] = (pad, pad + lens[r])
    Q = max(len(b) for b in blocks)
    block = np.full((n, Q), eot, dtype=np.int64)
    for r, b in enumerate(blocks):
        block[r, :len(b)] = b
    return prefix, block, marks, spans
