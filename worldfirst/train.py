"""Training and examining one model of stage 1 (the Laws model or its rival) on the world.

Both learn from the world only: random scenes and random actions, and the world's true answer after each. Each has
its own curriculum (world.Curriculum), which always serves the edge of what *that* learner can do, by the same rule.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import torch

from . import world as W
from .data import Vocab, laws_batch, rival_batch

TRAIN_SEED0 = 1_000_000          # training episodes use seeds from here up; exam episodes use seeds below it


def batch_for(model, eps, vocab, device, rng=None):
    return laws_batch(eps, vocab, device) if model.kind == "laws" else rival_batch(eps, vocab, device, rng=rng)


def question_hits(hits, meta):
    """Item hits -> per question: right only if every word of its answer is right."""
    q = {}
    for h, m in zip(hits.tolist(), meta):
        q[m["q"]] = q.get(m["q"], True) and h
    return q


def autocast(device):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


# ---------------------------------------------------------------- making batches in helper processes
_VOCAB = None


def _init_worker(words):
    global _VOCAB
    _VOCAB = Vocab(words)
    torch.set_num_threads(1)


def _np(b):
    return {k: (v.numpy() if torch.is_tensor(v) else _np(v) if isinstance(v, dict) else v) for k, v in b.items()}


def _build(job):
    """One training batch, on the CPU: the episodes of one step, already turned into numbers."""
    kind, picks, seed0, tag_seed, vocab = job
    vocab = vocab or _VOCAB
    eps = [W.make(f, lv, "train", seed0 + i) for i, (f, lv) in enumerate(picks)]
    if kind == "laws":
        return laws_batch(eps, vocab)
    return rival_batch(eps, vocab, rng=random.Random(tag_seed))


def _to(b, device):
    out = {}
    for k, v in b.items():
        if k == "meta":
            out[k] = v
        elif isinstance(v, dict):
            out[k] = _to(v, device)
        else:
            t = torch.from_numpy(v) if not torch.is_tensor(v) else v
            out[k] = t.to(device, non_blocking=True)
    return out


class Feeder:
    """Builds the next few steps' batches ahead of time in helper processes, so the GPU never waits for the world.

    The curriculum picks each step's (family, level)s when the step is queued, a few steps early: a lag of a few
    steps in a run of thousands changes nothing. workers=0 builds each batch in place (CPU runs, tests)."""

    def __init__(self, kind, vocab, cur, batch, seed, workers=0, ahead=6):
        self.kind, self.vocab, self.cur, self.batch, self.seed = kind, vocab, cur, batch, seed
        self.pool = None
        if workers:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(workers, initializer=_init_worker, initargs=(vocab.words,))
        self.ahead = ahead if workers else 1
        self.q = deque()
        self.next = None

    def _job(self, k):
        picks = [self.cur.pick() for _ in range(self.batch)]
        return (self.kind, picks, TRAIN_SEED0 + k * self.batch, self.seed * 1_000_003 + k,
                None if self.pool else self.vocab)

    def get(self, k):
        """The batch of step k (steps are asked for in order)."""
        if self.next is None:
            self.next = k
        while len(self.q) < self.ahead:
            job = self._job(self.next)
            self.q.append(self.pool.apply_async(_build, (job,)) if self.pool else job)
            self.next += 1
        item = self.q.popleft()
        return item.get() if self.pool else _build(item)

    def close(self):
        if self.pool:
            self.pool.terminate()


def train(model, vocab, out: Path, steps=20000, batch=128, lr=6e-4, warmup=None, device="cpu", curriculum=True,
          minutes=None, log_every=50, eval_every=None, eval_n=24, seed=0, say=print, workers=None, save_minutes=10):
    """Train one model. A long run is safe to stop: every `save_minutes` it saves train_state.pt, and starting again
    with the same settings carries on from there (same steps, same curriculum, same place in the schedule)."""
    out.mkdir(parents=True, exist_ok=True)
    cuda = str(device).startswith("cuda")
    warmup = warmup or max(300, steps // 100)
    eval_every = eval_every if eval_every is not None else max(1000, steps // 100)
    workers = (3 if cuda else 0) if workers is None else workers
    model.to(device).train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": rest, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=cuda)
    sched = lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))  # noqa: E731
    cur = W.Curriculum(on=curriculum, seed=seed)
    rng = random.Random(seed)
    start, done_minutes = 0, 0.0
    state_file = out / "train_state.pt"
    if state_file.exists():
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        if st.get("steps") == steps and st.get("batch") == batch:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            cur.front, cur.acc, cur.seen = st["cur"]["front"], st["cur"]["acc"], st["cur"]["seen"]
            cur.rng.setstate(st["cur"]["rng"])
            rng.setstate(st["rng"])
            start, done_minutes = st["step"], st["minutes"]
            _trim_metrics(out / "metrics.jsonl", start)
            say(f"{model.kind}: carrying on from step {start} ({done_minutes:.0f} min already done)")
        else:
            say(f"{model.kind}: a saved state is there but for other settings; starting fresh")
    elif (out / "metrics.jsonl").exists():
        (out / "metrics.jsonl").unlink()
    feed = Feeder(model.kind, vocab, cur, batch, seed, workers=workers)
    metrics = open(out / "metrics.jsonl", "a")
    t0 = time.time() - done_minutes * 60
    t_run, s_run = time.time(), start
    last_save = time.time()
    run_loss, run_acc, run_n = 0.0, 0.0, 0
    step = start - 1
    try:
        for step in range(start, steps):
            for g in opt.param_groups:
                g["lr"] = lr * sched(step)
            b = _to(feed.get(step), device)
            kw = {"passes": model.cfg.fixed or rng.randint(1, model.cfg.train_max)} if model.kind == "laws" else {}
            with autocast(device):
                loss, hits, _ = model.loss_and_hits(b, **kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            qh = question_hits(hits, b["meta"])
            by_ep = {m["q"]: (m["family"], m["level"]) for m in b["meta"]}
            for qid, ok in qh.items():
                f, lv = by_ep[qid]
                cur.update(f, lv, float(ok))
            run_loss += loss.item()
            run_acc += sum(qh.values()) / max(1, len(qh))
            run_n += 1
            el = time.time() - t0
            if (step + 1) % log_every == 0 or step == start:
                rate = (step + 1 - s_run) / max(1e-6, time.time() - t_run)
                left = (steps - step - 1) / max(1e-6, rate) / 3600
                rec = {"kind": "train", "step": step + 1, "loss": run_loss / run_n, "acc": run_acc / run_n,
                       "episodes": (step + 1) * batch, "minutes": el / 60, "lr": lr * sched(step), "frontier": cur.state()}
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                _status(out, {"state": "training", "kind": model.kind, "step": step + 1, "steps": steps, "batch": batch,
                              "minutes": el / 60, "loss": rec["loss"], "acc": rec["acc"], "frontier": cur.state(),
                              "steps_per_s": rate, "hours_left": left,
                              "params": sum(p.numel() for p in model.parameters())})
                say(f"{model.kind} step {step + 1}/{steps}  loss {rec['loss']:.4f}  right {100 * rec['acc']:.1f}%  "
                    f"{el / 60:.1f} min  {rate:.1f} steps/s  {left:.1f} h left  frontier "
                    + " ".join(f"{k}:{v}" for k, v in cur.state().items()))
                run_loss, run_acc, run_n = 0.0, 0.0, 0
            if eval_every and (step + 1) % eval_every == 0 and step + 1 < steps:
                res = evaluate(model, vocab, n=eval_n, device=device, sweep=False)
                sm = summarise(res)
                metrics.write(json.dumps({"kind": "exam", "step": step + 1, "minutes": el / 60, **sm}) + "\n")
                say(f"{model.kind} short exam at step {step + 1}:  secret {100 * sm.get('secret', 0):.1f}%  "
                    f"practised kinds {100 * sm.get('seen', 0):.1f}%")
                metrics.flush()
                model.train()
            if save_minutes and time.time() - last_save > save_minutes * 60 and step + 1 < steps:
                _save_state(state_file, model, opt, cur, rng, step + 1, el / 60, steps, batch)
                last_save = time.time()
            if minutes and el > minutes * 60:
                say(f"{model.kind}: time is up after {step + 1} steps")
                break
    finally:
        feed.close()
    info = {"steps": step + 1, "episodes": (step + 1) * batch, "minutes": (time.time() - t0) / 60, "frontier": cur.state()}
    last = None
    try:
        last = [json.loads(x) for x in open(out / "metrics.jsonl") if '"train"' in x][-1]
    except Exception:
        pass
    _status(out, {"state": "done", "kind": model.kind, "step": step + 1, "steps": steps, "batch": batch,
                  "minutes": info["minutes"], "acc": (last or {}).get("acc"), "frontier": cur.state(), "params": sum(p.numel() for p in model.parameters())})
    if state_file.exists():
        state_file.unlink()
    return info


def _save_state(path, model, opt, cur, rng, step, minutes, steps, batch):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "minutes": minutes,
                "steps": steps, "batch": batch, "rng": rng.getstate(),
                "cur": {"front": cur.front, "acc": cur.acc, "seen": cur.seen, "rng": cur.rng.getstate()}}, tmp)
    tmp.replace(path)


def _trim_metrics(path, step):
    """Drop log lines from after the saved step (they will be written again)."""
    if not path.exists():
        return
    keep = [x for x in open(path) if x.strip() and json.loads(x).get("step", 0) <= step]
    path.write_text("".join(keep))


def _status(out: Path, st: dict):
    """status.json next to metrics.jsonl: what the world dashboard reads for the progress bar."""
    tmp = out / "status.json.tmp"
    tmp.write_text(json.dumps(st))
    tmp.replace(out / "status.json")


@torch.no_grad()
def evaluate(model, vocab, n=200, device="cpu", sweep=True, bs=100, sets=None):
    """Every exam set (seen levels on fresh episodes, and every secret exam), n episodes each.

    Returns rows: {family, split, level, mode, n_q, right, passes (Laws only, mean used)}. For the Laws model, mode
    "settle" is the real test (thinks until the picture stops moving); with sweep=True it is also run at fixed passes,
    which shows whether thinking longer helps."""
    model.eval()
    rows = []
    modes = [("settle", None)]
    if model.kind == "laws" and sweep:
        modes += [(f"p{k}", k) for k in (1, 2, 4, 8, 16, 32)]
    if model.kind == "rival":
        modes = [("plain", None)]
    rng = random.Random(7)
    for fam, lv, sp in (sets or W.exam_sets()):
        eps = [W.make(fam, lv, sp, s) for s in range(n)]
        for mode, k in modes:
            right, nq, used_sum, rev, bel = 0, 0, 0.0, defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
            for i in range(0, n, bs):
                chunk = eps[i:i + bs]
                b = batch_for(model, chunk, vocab, device, rng)
                with autocast(device):
                    if model.kind == "laws":
                        _, hits, used = model.loss_and_hits(b, passes=k, adaptive=(k is None))
                        used_sum += used.float().sum().item()
                    else:
                        _, hits, _ = model.loss_and_hits(b)
                qh = question_hits(hits, b["meta"])
                right += sum(qh.values())
                nq += len(qh)
                tags = {m["q"]: m.get("reversed") for m in b["meta"]}
                btags = {m["q"]: m.get("belief") for m in b["meta"]}
                for qid, ok in qh.items():
                    if tags[qid] is not None:
                        rev[tags[qid]][0] += ok
                        rev[tags[qid]][1] += 1
                    if btags[qid] is not None:
                        bel[btags[qid]][0] += ok
                        bel[btags[qid]][1] += 1
            row = {"family": fam, "split": "seen" if sp == "train" else sp, "level": lv, "mode": mode,
                   "n_q": nq, "right": right / max(1, nq)}
            if model.kind == "laws":
                row["passes"] = used_sum / n
            if rev:
                row["reversed"] = {str(k): v for k, v in rev.items()}
            if bel:
                row["belief"] = {str(k): v for k, v in bel.items()}
            rows.append(row)
    return rows


def summarise(rows, mode=None):
    """Mean right per family on seen and on secret exams (one mode: settle for Laws, plain for the rival)."""
    out, parts = {}, {}
    for r in rows:
        if r["mode"] not in ("settle", "plain") if mode is None else r["mode"] != mode:
            continue
        part = "seen" if r["split"] == "seen" else "secret"
        out.setdefault(f"{r['family']}/{part}", []).append(r["right"])
        parts.setdefault(part, []).append(r["right"])
    s = {k: sum(v) / len(v) for k, v in out.items()}
    for part, vals in parts.items():           # every exam set (family x level) weighs the same
        s[part] = sum(vals) / len(vals)
    return s
