"""Think: is reasoning depth bought with weights, or with reading time?  (NEXT-TOKEN)

Three models of one size are trained on scenes alone. By default "circuits", this file's own plain rung: switches
and machines that follow them, in bare sentences, every part asked; trained to depth 3, examined to depth 8.
With --families switches,rules: Field1 scenes (oazy/field1: small simulated worlds told as stories; these two
families are chains of rules told in any order: "switches", where a machine follows machines that follow switches,
and "rules", if-then chains). Every scene is told R times in a row (Field1: R = 1..--reads; a circuit: its depth
+ 1 or + 2) and then asked:

    <|endoftext|> story \\n\\n story \\n\\n story \\n question Answer: yes.\\n question Answer: no.

  feed   Keep with its stores written from the TOP of the stack (keep/model.py, feed): what the last layer works out
         at one token is what the first layers read at later tokens, so every further reading is one more turn of the
         whole stack on what the readings before concluded
  keep   Keep as it was: the stores read and written at one layer (the control without that wire)
  full   the plain twin: attention over the whole window, no stores

The exam: scenes never trained on, told 1, 2, 3, 4, 6, 8, 12 times, the questions marked on the answer's first piece
among the scene's choices. Scores by family, by the question's level (the knob a family gives its questions, e.g.
depth) and by the number of readings. What to look for: does the score rise with the readings (past the --reads of
training too), and at which level does each model fall to the floor?

    python3 -m keep.think --preset base --name think                    # GPU: the three models on circuits, then the exam
    python3 -m keep.think --preset base --name field --families switches,rules      # the same on Field1's two families
    python3 -m keep.think --preset base --name think --only feed        # one of them (they can train side by side)
    python3 -m keep.think --preset base --name think --only report      # the exam and the report from the saved models

Needs no text data: only the SmolLM2 tokenizer and the Field1 generator. --families circuits is the quickest answer.
Writes <runs>/<name>/: feed/, keep/, full/ (ckpt.pt, log.jsonl), think.json, think.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .model import Keep, KeepConfig
from . import data as D
from .run import PRESETS, amp, runs_dir

ARMS = ("feed", "keep", "full")
NAMES = {"feed": "Keep, stores written from the top", "keep": "Keep", "full": "Twin"}
FAMILIES = ("circuits",)                   # the default: this file's own plain rung (below)
FIELD1 = ("switches", "rules")             # Field1's two families that are chains of rules told in any order (their questions say how deep)
OTHERS = ("places", "compare", "count", "family", "schedule")      # more families whose questions say how many steps they take
# what differs from run.py's presets: the training window (a scene told several times has to fit in it), the stores
# of the fed model (where they are read), how far back its top may point, the tokens
THINK = {
    "tiny": dict(cfg=dict(max_len=512), feed=dict(stores=(1,), point=4), tokens=2e5, reads=3, exam_n=12, exam_reads=(1, 2, 3)),
    "pilot": dict(cfg=dict(max_len=512), feed=dict(stores=(1,), point=16, chunk=16), tokens=1.2e7, reads=3, exam_n=100, exam_reads=(1, 2, 3, 4, 6, 8)),
    # base, big: as the pilots that worked, scaled to sentences: attention reaches about three sentences (32 tokens: a scene is
    # NOT in view, so what is known of it has to go through the stores), a chunk is one or two (16), windows of 512 (a scene told
    # 5 times fits), small stores (a window's scenes need few rows; the fed tables are dense), many rows a step (a fed step is
    # 32 chunks one after the other, so its speed comes from the batch: a fed step is some 100,000 small operations whatever
    # the batch, 23k tok/s at 64 rows on a 5090. An address of 4 rows and values of 32 numbers, as in the pilots, keep the
    # fed tables at about 0.07 GB a row instead of 0.2)
    "base": dict(cfg=dict(max_len=512), keep=dict(window=32, side=64, k=4, dv=32), feed=dict(stores=(2, 5), point=16, chunk=16), batch=128, tokens=3e8, reads=4,
                 exam_n=600, exam_reads=(1, 2, 3, 4, 6, 8, 12)),
    "big": dict(cfg=dict(max_len=512), keep=dict(window=32, side=64, k=4, dv=32), feed=dict(stores=(3, 7), point=16, chunk=16), batch=128, tokens=8e8, reads=4,
                exam_n=600, exam_reads=(1, 2, 3, 4, 6, 8, 12)),
}
TRAIN_SEED, EXAM_SEED = "think-train", "think-exam"
# --align C: a telling is ONE chunk. Every telling is padded on its left to C tokens and so is the question block, and
# the fed model's chunk is C: a telling goes through the stack in one parallel pass, what its top worked out lands in
# the stores when it ends, and the next telling reads it. A training step is then ALIGN_ROOM passes one after the
# other instead of 32 (512 tokens in chunks of 16), each over C tokens of every row. (Token pilot clock/fb4.py: it
# learns as fast, and goes as far past the depth it was trained on.) One more telling is one more step of reasoning
# for the WHOLE scene, in whatever order its sentences stand.
ALIGN_ROOM = 6                              # chunks to a training window: a circuit of depth 3 told 5 times, and its questions

# ---------------------------------------------------------------- circuits: the plain rung
# Field1's switches in bare sentences: no setting, no people, no flips, one wording for each rule, and EVERY part asked
# (the switches too: a plain lookup, learned before computing). Between the token pilots (clock/fb2.py) and Field1.
#   The red switch is on.  The lamp is on if the pump and the fan are on.  If the red switch is off, the pump is on.
#   Is the lamp on? Answer: yes.
# Training: circuits no deeper than CIRCUIT_TRAIN, told (its depth + 1 or + 2) times. Exam: circuits built to depth
# 1, 2, 3, 4, 6, 8 (each machine follows the one before it), so that the levels past CIRCUIT_TRAIN were never trained.
CIRCUIT_TRAIN = 3
CIRCUIT_EXAM = (1, 2, 3, 4, 6, 8)
C_SWITCH = ["red", "blue", "green", "grey", "white", "black", "brown", "pink", "orange", "purple", "yellow", "silver"]
C_MACHINE = ["lamp", "pump", "fan", "bell", "radio", "motor", "heater", "siren", "buzzer", "alarm", "light", "clock", "door", "gate", "mill",
             "drill", "press", "valve", "crane", "oven", "kettle", "robot", "drone", "engine"]


def circuit(i, seed, top=None):
    """Scene i as levels.scene gives one: {"story", "qas": [(question, answer, _, {"depth": d})], "depth": its deepest}.
    A seed that starts with EXAM_SEED gives the exam's circuits. top: no deeper than this (training's curriculum)."""
    r = random.Random(f"circuits|{seed}|{i}")
    exact = CIRCUIT_EXAM[i % len(CIRCUIT_EXAM)] if (seed[0] if isinstance(seed, tuple) else seed) == EXAM_SEED else None
    while True:
        nsw = r.randint(2, 3)
        nm = exact + r.randint(0, 1) if exact else r.randint(2, 5)
        sw = [f"the {c} switch" for c in r.sample(C_SWITCH, nsw)]
        ms = [f"the {m}" for m in r.sample(C_MACHINE, nm)]
        val = {s: r.random() < 0.5 for s in sw}
        depth = {s: 0 for s in sw}
        parts, said = list(sw), [f"{s.capitalize()} is {'on' if val[s] else 'off'}." for s in sw]
        for k, m in enumerate(ms):
            if exact:                               # past the chain: a machine on an earlier part, so that none is deeper than asked for
                first = parts[-1] if k < exact else r.choice(parts[:-1])
            else:
                first = parts[-1] if (k and r.random() < 0.6) else r.choice(parts)
            ins = [first]
            if r.random() < 0.4:
                ins.append(r.choice([p for p in (parts[:-1] if exact and k >= exact else parts) if p != first]))
            op = r.choice(["same", "not"]) if len(ins) == 1 else r.choice(["and", "or"])
            xs = [val[p] for p in ins]
            val[m] = {"same": xs[0], "not": not xs[0], "and": all(xs), "or": any(xs)}[op]
            depth[m] = 1 + max(depth[p] for p in ins)
            cond = {"same": f"{ins[0]} is on", "not": f"{ins[0]} is off", "and": f"{ins[0]} and {ins[-1]} are on", "or": f"{ins[0]} or {ins[-1]} is on"}[op]
            said.append(f"{m.capitalize()} is on if {cond}." if r.random() < 0.5 else f"If {cond}, {m} is on.")
            parts.append(m)
        if exact or max(depth.values()) <= (top or CIRCUIT_TRAIN):
            break
    r.shuffle(said)
    asked = parts if not exact else sw + ms[exact - 1:] + ms[:exact - 1]
    if not exact:
        r.shuffle(asked)
    return {"story": " ".join(said), "depth": max(depth.values()),
            "qas": [(f"Is {p} on?", "yes" if val[p] else "no", None, {"depth": depth[p]}) for p in asked]}


def make(arm: str, preset: dict, vocab: int, point=None, stores=None, align=0, share=None) -> Keep:
    cfg = dict(preset["cfg"], vocab_size=vocab, **THINK[preset["name"]].get("cfg", {}))
    if align:                                       # a window: ALIGN_ROOM chunks, a telling to each, the questions in the last
        cfg["max_len"] = ALIGN_ROOM * align
    if arm in ("keep", "feed"):
        cfg.update(preset["keep"])
        cfg.update(THINK[preset["name"]].get("keep", {}))
    if arm == "feed":
        cfg.update(feed=True, **THINK[preset["name"]]["feed"])
        if point is not None:
            cfg["point"] = point
        if stores:
            cfg["stores"] = stores
        if align:
            cfg["chunk"] = align
        if share:
            cfg["share"] = share
    return Keep(KeepConfig(**cfg))


# ---------------------------------------------------------------- scenes told R times
def level(knobs: dict):
    """The level of a question: its depth when the family gives one, else the first knob it gives, else None."""
    for k in ("depth", "hops", "nest", "gap", "changes", "rel", "false_belief"):
        if k in knobs:
            return f"{k} {int(knobs[k]) if isinstance(knobs[k], bool) else knobs[k]}"
    for k, v in knobs.items():
        if not k.startswith("_") and isinstance(v, (int, bool, str)):
            return f"{k} {int(v) if isinstance(v, bool) else v}"
    return None


def told(enc, family: str, i, seed, top=None):
    """Scene i -> (one reading as tokens, what stands between two readings, the question lines
    [(lead tokens, answer tokens, choices, answer, level)])."""
    from oazy.field1 import levels
    sc = circuit(i, seed, top) if family == "circuits" else levels.scene(family, i, seed)
    lines = []
    for q, a, _, qk in sc["qas"]:
        lines.append((enc.encode_ordinary(f"\n{q} Answer:"), enc.encode_ordinary(f" {a}."),
                      ["yes", "no"] if family == "circuits" else levels.choices(family, q, a, sc["story"], knobs=qk), a, level(qk)))
    return enc.encode_ordinary(sc["story"]), enc.encode_ordinary("\n\n"), lines, sc.get("depth")


def scene_tokens(enc, family, i, seed, reads, room: int, rng=None, most=1, align=0, top=None):
    """A scene told `reads` times and asked, as it stands in a stream: tokens, what each is (D.STORY / D.ASKED), and
    the marks [(place whose next piece is asked, candidate pieces, which is right, level, the answer)].
    Told fewer times if it would not fit in `room` tokens; None if one telling does not fit. reads None: drawn (training).
    align C: every telling fills a chunk of C tokens (<|endoftext|> on its left), the questions end the next one."""
    story, sep, lines, deep = told(enc, family, i, seed, top)
    if reads is None:                               # training: Field1 1..R times; a circuit as often as its depth asks (+ 1 or 2)
        reads = rng.randint(1, most) if deep is None else deep + rng.randint(1, 2)
    block = [t for lead, ans, *_ in lines for t in lead + ans]
    if align:                                       # a telling to a chunk, the questions at the end of the next
        if len(story) > align or len(block) > align or not lines:
            return None
        reads = min(reads, room // align - 1)
        if reads < 1:
            return None
        toks = ([enc.eot_token] * (align - len(story)) + story) * reads + [enc.eot_token] * (align - len(block))
    else:
        while reads > 1 and 1 + reads * len(story) + (reads - 1) * len(sep) + len(block) > room:
            reads -= 1
        toks = [enc.eot_token] + story
        for _ in range(reads - 1):
            toks += sep + story
        if len(toks) + len(block) > room or not lines:
            return None
    kinds = [D.STORY] * len(toks)
    marks = []
    for lead, ans, ch, a, lv in lines:
        toks += lead
        m = D._mark(enc, len(toks), ans, ch, a)
        if m is not None:
            marks.append((len(toks) - 1 + m[0], m[1], m[2], lv, a))
        toks += ans
    return toks, kinds + [D.ASKED] * (len(toks) - len(kinds)), marks, reads


def stream(rng: random.Random, enc, T: int, families, reads: int, tag=0, align=0, cap=None):
    """Yields (tokens, what they are) for ever: one scene to a training window, as in the exam: <|endoftext|> up to
    the scene, then the scene told again and again and its questions at the window's end. (A second scene in the
    window would find the first one's switches still in the stores: the same names, other states.)"""
    from oazy.field1 import WEIGHTS
    wts = [WEIGHTS.get(f, 5) for f in families]
    n, none = 0, 0
    while True:
        f = rng.choices(families, weights=wts)[0]
        sc = scene_tokens(enc, f, n, (TRAIN_SEED, tag), None, T, rng=rng, most=reads, align=align, top=cap.value if cap is not None else None)
        n += 1
        if sc is None:
            none += 1
            if none > 200 and none > 0.9 * n:
                raise RuntimeError(f"scenes of {', '.join(families)} do not fit in a window of {T} tokens")
            continue
        toks, kinds = sc[0], sc[1]
        if len(toks) < T:
            yield [enc.eot_token] * (T - len(toks)), D.TEXT
        yield (toks, kinds)


class Rows:
    """`rows` streams cut into windows, as data.Rows: next() -> tokens (rows, T + 1), what each is (rows, T + 1)."""

    def __init__(self, enc, rows, T, seed, families, reads, first_row=0, align=0, cap=None):
        self.T, self.eot = T, enc.eot_token
        assert not align or T % align == 0, "a window is whole chunks"
        self.gens = [stream(random.Random(f"{seed}|{first_row + r}"), enc, T, families, reads, tag=first_row + r, align=align, cap=cap) for r in range(rows)]
        self.buf = [([], []) for _ in range(rows)]

    def next(self):
        out = np.empty((len(self.gens), self.T + 1), dtype=np.int64)
        what = np.empty((len(self.gens), self.T + 1), dtype=np.uint8)
        for r, g in enumerate(self.gens):
            b, k = self.buf[r]
            while len(b) < self.T + 1:
                t, kind = next(g)
                b.extend(t)
                k.extend(kind if isinstance(kind, list) else [kind] * len(t))
            out[r], what[r] = b[:self.T + 1], k[:self.T + 1]
            self.buf[r] = (b[self.T:], k[self.T:])
        return out, what


def _worker(q, rows, T, seed, families, reads, first_row, align=0, cap=None):
    from oazy import tok
    r = Rows(tok.get("smollm2"), rows, T, seed, families, reads, first_row, align, cap)
    while True:
        q.put(r.next())


class Feed:
    """Batches of windows from `workers` processes (0 = made here). Row r is always the same stream."""

    def __init__(self, enc, batch, T, seed, families, reads, workers=8, align=0, ramp=False):
        import multiprocessing as mp
        self.procs, self.qs, self.local = [], [], None
        workers = min(workers, batch)
        ctx = mp.get_context("fork")
        self.cap = cap = ctx.Value("i", 1) if ramp else None        # the curriculum: circuits no deeper than this (the trainer raises it)
        if workers <= 0:
            self.local = Rows(enc, batch, T, seed, families, reads, align=align, cap=cap)
            return
        first = 0
        for i in range(workers):
            n = batch // workers + (1 if i < batch % workers else 0)
            q = ctx.Queue(maxsize=6)
            p = ctx.Process(target=_worker, args=(q, n, T, seed, families, reads, first, align, cap), daemon=True)
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
@torch.no_grad()
def exam(model: Keep, enc, families, reads_list, n, device, batch=32, seed=0, say=None, align=0):
    """-> {family: {level: {reads: [right, asked, commonest answer's count]}}} on scenes never trained on.
    A model with stores reads a scene from its beginning, window after window, the stores carried. The twin sees the
    last max_len tokens (at many readings: the last tellings and the questions).
    align: a telling to a chunk, as in training; the chunk is as long as the exam's longest telling or question block
    asks (its scenes are deeper, so longer, than training's), and the fed model takes a scene in one go."""
    model.eval()
    T = model.cfg.max_len
    eot = enc.eot_token
    res = {}
    was = model.cfg.chunk
    for f in families:
        res[f] = {}
        C = 0
        if align:
            for i in range(n):
                story, _, lines, _ = told(enc, f, i, (EXAM_SEED, seed))
                C = max(C, len(story), sum(len(lead) + len(ans) for lead, ans, *_ in lines))
            C = min(T, max(align, -(-C // 16) * 16))
            if model.cfg.feed:
                model.cfg.chunk = C
        for R in reads_list:
            rows = []
            for i in range(n):
                sc = scene_tokens(enc, f, i, (EXAM_SEED, seed), R, 1 << 30, align=C)
                if sc is not None and sc[3] == R and sc[2]:
                    rows.append(sc)
            for lo in range(0, len(rows), batch):
                part = rows[lo:lo + batch]
                L = max(len(s[0]) for s in part)
                W = C if C and model.cfg.feed else T
                L = min(L, T) if not model.store else -(-L // W) * W      # stores: whole windows, so that the last one holds the scene's end (and most scenes whole)
                if C and model.cfg.feed:
                    W = L                                                 # in one go, as a training window: a chunk's first tokens see the end of the chunk before
                x = np.full((len(part), L), eot, dtype=np.int64)
                offs = []
                for r, s in enumerate(part):
                    t = s[0][-L:]
                    x[r, L - len(t):] = t
                    offs.append(L - len(s[0]))
                x = torch.from_numpy(x).to(device)
                with amp(device):
                    if model.store:
                        st, hs = model.init_state(len(part), device), []
                        for c in range(0, L, W):
                            h, st, _ = model.hidden(x[:, c:c + W], st)
                            hs.append(h)
                        h = torch.cat(hs, 1)
                    else:
                        h = model.hidden(x)[0]
                for r, s in enumerate(part):
                    for p, cands, right, lv, a in s[2]:
                        if offs[r] + p < 0:
                            continue
                        sc_ = model.scores_at(h, torch.tensor([r], device=device), torch.tensor([offs[r] + p], device=device))[0]
                        ok = int(sc_[torch.tensor(cands, device=device)].argmax()) == right
                        for key in (lv, "all"):
                            if key is None:
                                continue
                            cell = res[f].setdefault(key, {}).setdefault(R, [0, 0, {}])
                            cell[0] += ok
                            cell[1] += 1
                            cell[2][a] = cell[2].get(a, 0) + 1
        for lv in res[f]:
            for R in res[f][lv]:
                c = res[f][lv][R]
                res[f][lv][R] = [c[0], c[1], max(c[2].values())]
        if say:                                     # all the questions, then a middle level and the deepest (of those asked 10 times or more)
            lvs = sorted((k for k in res[f] if k != "all" and max(c[1] for c in res[f][k].values()) >= 10), key=_order)
            for key in ["all"] + sorted({lvs[len(lvs) // 2], lvs[-1]}, key=_order) if lvs else ["all"]:
                row = res[f].get(key, {})
                say(f"    {f + (', ' + key if key != 'all' else ''):20s} " + "  ".join(f"{R}x {100 * c[0] / max(1, c[1]):.0f}%" for R, c in row.items())
                    + (f"   (commonest answer {100 * max(c[2] for c in row.values()) / max(1, max(c[1] for c in row.values())):.0f}%)" if row else ""))
    model.cfg.chunk = was
    model.train()
    return res


# ---------------------------------------------------------------- training
def train(arm, preset, th, args, enc, device, out: Path, say):
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt.pt"
    if ck.exists():
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        if sd.get("done"):
            say(f"[{arm}] already trained ({sd['tokens']:,} tokens): not trained again")
            m = Keep(KeepConfig.from_dict(sd["config"]))
            m.load_state_dict(sd["model"])
            return m.to(device), sd.get("summary", {})
    if args.only == "report":
        return None, {}
    torch.manual_seed(args.seed)
    model = make(arm, preset, enc.n_vocab, point=args.point,
                 stores=tuple(int(i) for i in args.stores.split(",")) if args.stores else None, align=args.align, share=args.share).to(device)
    T, B = model.cfg.max_len, args.batch or th.get("batch") or preset["batch"]
    total = int(args.tokens or th["tokens"])
    steps = max(1, total // (B * T))
    lr = args.lr or preset["lr"]
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": rest, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    warm = max(10, steps // 50)
    reads = args.reads or th["reads"]
    feed = Feed(enc, B, T, args.seed, args.families, reads, workers=args.workers, align=args.align, ramp=bool(args.ramp))
    yn = [enc.encode_ordinary(" yes.")[0], enc.encode_ordinary(" no.")[0]]       # the curriculum's measure: the share of yes/no answers right
    right, since = None, 0
    say(f"[{arm}] {model.n_params() / 1e6:.1f}M weights, attention reaches {model.cfg.window or T} tokens, "
        + (f"{len(model.store)} stores of {model.cfg.side ** 2:,} rows" + (f" written from the top (it may file under one of the {model.cfg.point} tokens before)"
                                                                         if model.cfg.feed else "") if model.store else "no stores")
        + f"; {steps:,} steps of {B} x {T} = {steps * B * T / 1e6:.0f}M tokens; scenes told 1 to {reads} times: {', '.join(args.families)}"
        + (f"; a telling to a chunk of {args.align} tokens" if args.align else "") + (f"; writes share rows by {model.cfg.share}" if model.cfg.feed and model.cfg.share != "all" else ""))
    if args.align:                                  # a scene whose telling or questions are longer than the chunk is left out: say how many
        fit = [scene_tokens(enc, f, i, (TRAIN_SEED, 0), 1, T, align=args.align) is not None for f in args.families for i in range(300)]
        say(f"[{arm}] {100 * sum(fit) / len(fit):.0f}% of the scenes fit a chunk of {args.align} tokens" + ("" if sum(fit) > 0.9 * len(fit) else ": raise --align"))
    if args.compile and device.type == "cuda" and not model.cfg.feed:
        model.speed_up()
    log = open(out / "log.jsonl", "a")
    state = model.init_state(B, device) if model.store else None
    t0 = time.time()
    last, seen = t0, (0, t0)
    run = run_a = None
    mids = {int(steps * f) for f in (0.25, 0.5, 0.75)} if steps >= 40 else set()
    aw = preset.get("asked_weight", 1.0)
    for step in range(1, steps + 1):
        cur = lr * (step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = cur
        w, what = feed.next()
        w = torch.from_numpy(w).to(device, non_blocking=True)
        asked = torch.from_numpy(what[:, 1:] == D.ASKED).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        try:
            with amp(device):
                loss, new, info = model.loss(w[:, :-1], w[:, 1:], state, asked=asked, asked_weight=aw, pair=yn if args.ramp else None)
            loss.backward()
        except Exception as e:                      # a compile that fails must not cost the run: the same step again, plain
            if model._fast is None:
                raise
            say(f"[{arm}] torch.compile failed ({type(e).__name__}: {str(e)[:200]}); going on without it")
            model.speed_up(False)
            opt.zero_grad(set_to_none=True)
            with amp(device):
                loss, new, info = model.loss(w[:, :-1], w[:, 1:], state, asked=asked, asked_weight=aw)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)             # (the stores are empty at every window: a window is one scene)
        opt.step()
        if feed.cap is not None and "right" in info:                 # depth 1 first; the next depth when this one is answered right
            right = float(info["right"]) if right is None else 0.95 * right + 0.05 * float(info["right"])
            if right > args.ramp and feed.cap.value < CIRCUIT_TRAIN and step > since + 40:
                feed.cap.value += 1
                since, right = step, None
                say(f"[{arm}] step {step:,}: {100 * args.ramp:.0f}% of the answers right, circuits now up to depth {feed.cap.value}")
        lv = float(info["lm"])
        if not math.isfinite(lv):
            raise SystemExit(f"[{arm}] the loss is not a number at step {step}: lower --lr")
        run = lv if run is None else 0.98 * run + 0.02 * lv
        if "asked" in info:
            run_a = float(info["asked"]) if run_a is None else 0.98 * run_a + 0.02 * float(info["asked"])
        if step % args.every == 0 or step == steps or time.time() - last > 60:
            last = time.time()
            rate = (step - seen[0]) * B * T / max(1e-9, time.time() - seen[1])
            seen = (step, time.time())
            mem = f"  {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if device.type == "cuda" else ""
            wr = " ".join(f"{float(v):.2f}" for v in info["write"])
            say(f"[{arm}] step {step:,}/{steps:,}  {step * B * T / 1e6:.1f}M tokens  loss {run:.3f}  asked {run_a or 0:.3f}  {rate / 1e3:.0f}k tok/s"
                + (f"  write strength {wr}" if wr else "") + mem + f"  left {(steps - step) * B * T / rate / 60:.0f} min")
            log.write(json.dumps({"kind": "train", "step": step, "tokens": step * B * T, "loss": run, "asked": run_a, "tok_s": rate}) + "\n")
            log.flush()
        if step in mids:
            say(f"[{arm}] a short exam at {step * B * T / 1e6:.0f}M tokens (share right by times told):")
            mid = exam(model, enc, args.families, [r for r in (1, 2, 4, 8) if r in th["exam_reads"]] or th["exam_reads"][:4],
                       max(8, th["exam_n"] // 2), device, seed=1, say=say, align=args.align)
            log.write(json.dumps({"kind": "exam", "step": step, "tokens": step * B * T, "exam": mid}) + "\n")
            log.flush()
            seen = (step, time.time())
    feed.close()
    secs = time.time() - t0
    summary = {"params": model.n_params(), "tokens": steps * B * T, "secs": round(secs), "tok_s": round(steps * B * T / secs),
               "loss": run, "asked": run_a, "reads": reads}
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(), "kind": "keep", "arm": arm, "done": True,
                "tokens": steps * B * T, "summary": summary}, ck)
    say(f"[{arm}] trained: {summary['tokens'] / 1e6:.0f}M tokens in {secs / 60:.0f} min ({summary['tok_s'] / 1e3:.0f}k tok/s)")
    return model, summary


# ---------------------------------------------------------------- the report
def _order(lv: str):
    k, _, v = lv.rpartition(" ")
    return (k, int(v)) if v.lstrip("-").isdigit() else (lv, 0)


def report(res: dict) -> str:
    reads = [str(r) for r in res["exam_reads"]]
    arms = [a for a in ARMS if a in res["exam"]]
    L = ["# Think: is reasoning depth bought with weights, or with reading time?", "",
         f"Scenes ({', '.join(res['families'])}), each told again and again before its questions. Training told a Field1 scene "
         f"1 to {res['reads']} times and a circuit (trained to depth {CIRCUIT_TRAIN} only) its depth + 1 or + 2 times; the exam tells scenes "
         f"never trained on {', '.join(reads)} times. Share of questions right (%)."
         + (f" A telling fills one chunk of {res['align']} tokens (longer in the exam): the fed model takes a telling in one pass, and one more "
            "telling is one more step for the whole scene, so depth d asks for d + 1 tellings." if res.get("align") else ""), ""]
    L += ["| Model | Weights | Tokens | Speed |", "|---|---|---|---|"]
    for a in arms:
        s = res["train"].get(a, {})
        if s:
            L.append(f"| {NAMES[a]} | {s['params'] / 1e6:.1f}M | {s['tokens'] / 1e6:.0f}M | {s['tok_s'] / 1e3:.0f}k tok/s |")
    for f in res["families"]:
        levels = sorted({lv for a in arms for lv in res["exam"][a].get(f, {}) if lv != "all"}, key=_order) + ["all"]
        L += ["", f"## {f}", "", "| Level | Model | " + " | ".join(f"told {r}x" for r in reads) + " | Asked | Commonest answer |",
              "|---|---|" + "---|" * (len(reads) + 2)]
        for lv in levels:
            for a in arms:
                row = res["exam"][a].get(f, {}).get(lv)
                if not row:
                    continue
                cells = [f"{100 * row[r][0] / max(1, row[r][1]):.0f}" if r in row else "" for r in reads]
                n = max(c[1] for c in row.values())
                floor = 100 * max(c[2] / max(1, c[1]) for c in row.values())
                L.append(f"| {lv} | {NAMES[a]} | " + " | ".join(cells) + f" | {n} | {floor:.0f} |")
    L += ["", "How to read it: a model that reasons by reading again rises from left to right, also past the number of "
          "tellings it was trained with, and holds at the deeper levels. \"Commonest answer\" is what always giving the "
          "most frequent answer of that row would score: the floor."]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=list(THINK))
    ap.add_argument("--runs", default=None)
    ap.add_argument("--name", default="think")
    ap.add_argument("--families", default=",".join(FAMILIES), help="families, with commas (default: circuits, this file's plain rung; Field1's: switches,rules; "
                    f"others with a level: {', '.join(OTHERS)}; 'all': every family)")
    ap.add_argument("--reads", type=int, default=None, help="training: a scene is told 1 to this many times (default: the preset's)")
    ap.add_argument("--tokens", type=float, default=None, help="training tokens for each model")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--point", type=int, default=None, help="feed: the top may file a write under one of this many tokens before it (0: the token before)")
    ap.add_argument("--stores", default=None, help="feed: a store read after each of these blocks, e.g. 2 or 2,5")
    ap.add_argument("--align", type=int, default=0, help="a telling to a chunk of this many tokens (96 for circuits): the fed model takes a whole "
                    "telling in one parallel pass, so a step is 6 passes one after the other instead of 32. 0: tellings follow each other, chunks of 16")
    ap.add_argument("--ramp", type=float, default=0.0, help="circuits by a curriculum, as the token pilots were trained: depth 1 only at first, the next depth "
                    "when this share of the yes/no answers is right (0.95). 0: every depth from the first step")
    ap.add_argument("--share", default=None, choices=["all", "strength"], help="feed: how a chunk's writes share a row (strength: a write that is held back "
                    "weakens no other; meant for --align, where a whole telling's writes land together)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--exam-n", type=int, default=None, help="scenes per family and number of tellings in the final exam")
    ap.add_argument("--only", default="all", choices=["all", "report"] + list(ARMS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    from oazy import tok
    from oazy.field1 import FAMILIES as ALL
    enc = tok.get("smollm2")
    args.families = list(ALL) if args.families == "all" else [f for f in args.families.split(",") if f]
    for f in args.families:
        if f not in ALL and f != "circuits":
            raise SystemExit(f"no Field1 family {f!r}; there are: {', '.join(ALL)}")
    preset = dict(PRESETS[args.preset], name=args.preset)
    th = THINK[args.preset]
    out = Path(args.runs or runs_dir()) / args.name
    out.mkdir(parents=True, exist_ok=True)
    say = lambda s: print(s, flush=True)
    res_path = out / "think.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res.update({"preset": args.preset, "families": args.families, "exam_reads": list(th["exam_reads"]), "reads": args.reads or th["reads"], "align": args.align})
    res.setdefault("train", {})
    res.setdefault("exam", {})
    for arm in [a for a in ARMS if args.only in ("all", "report", a)]:
        model, summary = train(arm, preset, th, args, enc, device, out / arm, say)
        if model is None:
            continue
        if summary:
            res["train"][arm] = summary
        say(f"[{arm}] the exam (share right by times told):")
        ex = exam(model, enc, args.families, th["exam_reads"], args.exam_n or th["exam_n"], device, say=say, align=args.align)
        if res_path.exists():                                       # another model may have been trained beside this one
            there = json.loads(res_path.read_text())
            for part in ("train", "exam"):
                res[part] = {**there.get(part, {}), **res[part]}
        res["exam"][arm] = json.loads(json.dumps(ex))               # the keys as json has them
        res_path.write_text(json.dumps(res, indent=1))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if res["exam"]:
        (out / "think.md").write_text(report(res))
        say("")
        say(report(res))
        say(f"written: {out / 'think.md'}")


if __name__ == "__main__":
    main()
