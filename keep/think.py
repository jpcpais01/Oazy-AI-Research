"""Think: is reasoning depth bought with weights, or with reading time?  (NEXT-TOKEN)

Three models of one size are trained on Field1 scenes alone (oazy/field1: small simulated worlds told as stories;
by default the two families that are chains of rules told in any order: "switches", where a machine follows
machines that follow switches, and "rules", if-then chains). Every scene is told R times in a row (R = 1..--reads)
and then asked:

    <|endoftext|> story \\n\\n story \\n\\n story \\n question Answer: yes.\\n question Answer: no.

  feed   Keep with its stores written from the TOP of the stack (keep/model.py, feed): what the last layer works out
         at one token is what the first layers read at later tokens, so every further reading is one more turn of the
         whole stack on what the readings before concluded
  keep   Keep as it was: the stores read and written at one layer (the control without that wire)
  full   the plain twin: attention over the whole window, no stores

The exam: scenes never trained on, told 1, 2, 3, 4, 6, 8 times, the questions marked on the answer's first piece
among the scene's choices. Scores by family, by the question's level (the knob a family gives its questions, e.g.
depth) and by the number of readings. What to look for: does the score rise with the readings (past the --reads of
training too), and at which level does each model fall to the floor?

    python3 -m keep.think --preset base --name think                    # GPU: the three models, then the exam
    python3 -m keep.think --preset base --name think --only feed        # one of them (they can train side by side)
    python3 -m keep.think --preset base --name think --only report      # the exam and the report from the saved models

Needs no text data: only the SmolLM2 tokenizer and the Field1 generator.
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
FAMILIES = ("switches", "rules")           # the two families that are chains of rules told in any order (their questions say how deep)
OTHERS = ("places", "compare", "count", "family", "schedule")      # more families whose questions say how many steps they take
# what differs from run.py's presets: the training window (a scene told several times has to fit in it), the stores
# of the fed model (where they are read), how far back its top may point, the tokens
THINK = {
    "tiny": dict(cfg=dict(max_len=512), feed=dict(stores=(1,), point=4), tokens=2e5, reads=3, exam_n=12, exam_reads=(1, 2, 3)),
    "pilot": dict(cfg=dict(max_len=512), feed=dict(stores=(1,), point=16), tokens=1.2e7, reads=3, exam_n=100, exam_reads=(1, 2, 3, 4, 6)),
    "base": dict(feed=dict(stores=(2, 5), point=32), tokens=3e8, reads=4, exam_n=300, exam_reads=(1, 2, 3, 4, 6, 8)),
    "big": dict(feed=dict(stores=(3, 7), point=32), tokens=8e8, reads=4, exam_n=300, exam_reads=(1, 2, 3, 4, 6, 8)),
}
TRAIN_SEED, EXAM_SEED = "think-train", "think-exam"


def make(arm: str, preset: dict, vocab: int, point=None, stores=None) -> Keep:
    cfg = dict(preset["cfg"], vocab_size=vocab, **THINK[preset["name"]].get("cfg", {}))
    if arm in ("keep", "feed"):
        cfg.update(preset["keep"])
    if arm == "feed":
        cfg.update(feed=True, **THINK[preset["name"]]["feed"])
        if point is not None:
            cfg["point"] = point
        if stores:
            cfg["stores"] = stores
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


def told(enc, family: str, i, seed):
    """Scene i -> (one reading as tokens, what stands between two readings, the question lines
    [(lead tokens, answer tokens, choices, answer, level)])."""
    from oazy.field1 import levels
    sc = levels.scene(family, i, seed)
    lines = []
    for q, a, _, qk in sc["qas"]:
        lines.append((enc.encode_ordinary(f"\n{q} Answer:"), enc.encode_ordinary(f" {a}."),
                      levels.choices(family, q, a, sc["story"], knobs=qk), a, level(qk)))
    return enc.encode_ordinary(sc["story"]), enc.encode_ordinary("\n\n"), lines


def scene_tokens(enc, family, i, seed, reads: int, room: int):
    """A scene told `reads` times and asked, as it stands in a stream: tokens, what each is (D.STORY / D.ASKED), and
    the marks [(place whose next piece is asked, candidate pieces, which is right, level, the answer)].
    Told fewer times if it would not fit in `room` tokens; None if one telling does not fit."""
    story, sep, lines = told(enc, family, i, seed)
    block = [t for lead, ans, *_ in lines for t in lead + ans]
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


def stream(rng: random.Random, enc, T: int, families, reads: int, tag=0):
    """Yields (tokens, what they are) for ever: scene after scene, each told 1..reads times. A scene never lies
    across two training windows (the twin would lose its story there): the window is filled up with <|endoftext|>."""
    from oazy.field1 import WEIGHTS
    wts = [WEIGHTS[f] for f in families]
    pos, n, none = 0, 0, 0
    while True:
        f = rng.choices(families, weights=wts)[0]
        sc = scene_tokens(enc, f, n, (TRAIN_SEED, tag), rng.randint(1, reads), T)
        n += 1
        if sc is None:
            none += 1
            if none > 200 and none > 0.9 * n:
                raise RuntimeError(f"scenes of {', '.join(families)} do not fit in a window of {T} tokens")
            continue
        toks, kinds = sc[0], sc[1]
        if pos + len(toks) > T:
            yield [enc.eot_token] * (T - pos), D.TEXT
            pos = 0
        yield (toks, kinds)
        pos = (pos + len(toks)) % T


class Rows:
    """`rows` streams cut into windows, as data.Rows: next() -> tokens (rows, T + 1), what each is (rows, T + 1)."""

    def __init__(self, enc, rows, T, seed, families, reads, first_row=0):
        self.T, self.eot = T, enc.eot_token
        self.gens = [stream(random.Random(f"{seed}|{first_row + r}"), enc, T, families, reads, tag=first_row + r) for r in range(rows)]
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


def _worker(q, rows, T, seed, families, reads, first_row):
    from oazy import tok
    r = Rows(tok.get("smollm2"), rows, T, seed, families, reads, first_row)
    while True:
        q.put(r.next())


class Feed:
    """Batches of windows from `workers` processes (0 = made here). Row r is always the same stream."""

    def __init__(self, enc, batch, T, seed, families, reads, workers=8):
        import multiprocessing as mp
        self.procs, self.qs, self.local = [], [], None
        workers = min(workers, batch)
        if workers <= 0:
            self.local = Rows(enc, batch, T, seed, families, reads)
            return
        ctx = mp.get_context("fork")
        first = 0
        for i in range(workers):
            n = batch // workers + (1 if i < batch % workers else 0)
            q = ctx.Queue(maxsize=6)
            p = ctx.Process(target=_worker, args=(q, n, T, seed, families, reads, first), daemon=True)
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
def exam(model: Keep, enc, families, reads_list, n, device, batch=32, seed=0, say=None):
    """-> {family: {level: {reads: [right, asked, commonest answer's count]}}} on scenes never trained on.
    A model with stores reads a scene from its beginning, window after window, the stores carried. The twin sees the
    last max_len tokens (at many readings: the last tellings and the questions)."""
    model.eval()
    T = model.cfg.max_len
    eot = enc.eot_token
    res = {}
    for f in families:
        res[f] = {}
        for R in reads_list:
            rows = []
            for i in range(n):
                sc = scene_tokens(enc, f, i, (EXAM_SEED, seed), R, 1 << 30)
                if sc is not None and sc[3] == R and sc[2]:
                    rows.append(sc)
            for lo in range(0, len(rows), batch):
                part = rows[lo:lo + batch]
                L = max(len(s[0]) for s in part)
                if not model.store:
                    L = min(L, T)
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
                        for c in range(0, L, T):
                            h, st, _ = model.hidden(x[:, c:c + T], st)
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
        if say:
            row = res[f].get("all", {})
            say(f"    {f:10s} " + "  ".join(f"{R}x {100 * c[0] / max(1, c[1]):.0f}%" for R, c in row.items())
                + (f"   (commonest answer {100 * max(c[2] for c in row.values()) / max(1, max(c[1] for c in row.values())):.0f}%)" if row else ""))
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
                 stores=tuple(int(i) for i in args.stores.split(",")) if args.stores else None).to(device)
    T, B = model.cfg.max_len, args.batch or preset["batch"]
    total = int(args.tokens or th["tokens"])
    steps = max(1, total // (B * T))
    lr = args.lr or preset["lr"]
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": rest, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    warm = max(10, steps // 50)
    reads = args.reads or th["reads"]
    feed = Feed(enc, B, T, args.seed, args.families, reads, workers=args.workers)
    say(f"[{arm}] {model.n_params() / 1e6:.1f}M weights, attention reaches {model.cfg.window or T} tokens, "
        + (f"{len(model.store)} stores of {model.cfg.side ** 2:,} rows" + (f" written from the top (it may file under one of the {model.cfg.point} tokens before)"
                                                                         if model.cfg.feed else "") if model.store else "no stores")
        + f"; {steps:,} steps of {B} x {T} = {steps * B * T / 1e6:.0f}M tokens; scenes told 1 to {reads} times: {', '.join(args.families)}")
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
                loss, new, info = model.loss(w[:, :-1], w[:, 1:], state, asked=asked, asked_weight=aw)
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
        state = model.detach(new) if state is not None else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
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
            mid = exam(model, enc, args.families, th["exam_reads"][:4], max(8, th["exam_n"] // 4), device, seed=1, say=say)
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
         f"Field1 scenes ({', '.join(res['families'])}), each told again and again before its questions. Training told a scene "
         f"1 to {res['reads']} times; the exam tells scenes never trained on {', '.join(reads)} times. Share of questions right (%).", ""]
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
    ap.add_argument("--families", default=",".join(FAMILIES), help="Field1 families, with commas (default: switches,rules; "
                    f"others with a level: {', '.join(OTHERS)}; 'all': every family)")
    ap.add_argument("--reads", type=int, default=None, help="training: a scene is told 1 to this many times (default: the preset's)")
    ap.add_argument("--tokens", type=float, default=None, help="training tokens for each model")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--point", type=int, default=None, help="feed: the top may file a write under one of this many tokens before it (0: the token before)")
    ap.add_argument("--stores", default=None, help="feed: a store read after each of these blocks, e.g. 2 or 2,5")
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
        if f not in ALL:
            raise SystemExit(f"no Field1 family {f!r}; there are: {', '.join(ALL)}")
    preset = dict(PRESETS[args.preset], name=args.preset)
    th = THINK[args.preset]
    out = Path(args.runs or runs_dir()) / args.name
    out.mkdir(parents=True, exist_ok=True)
    say = lambda s: print(s, flush=True)
    res_path = out / "think.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res.update({"preset": args.preset, "families": args.families, "exam_reads": list(th["exam_reads"]), "reads": args.reads or th["reads"]})
    res.setdefault("train", {})
    res.setdefault("exam", {})
    for arm in [a for a in ARMS if args.only in ("all", "report", a)]:
        model, summary = train(arm, preset, th, args, enc, device, out / arm, say)
        if model is None:
            continue
        if summary:
            res["train"][arm] = summary
        say(f"[{arm}] the exam (share right by times told):")
        ex = exam(model, enc, args.families, th["exam_reads"], args.exam_n or th["exam_n"], device, say=say)
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
