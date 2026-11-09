"""Training and examining one model of stage 1 (the Laws model or its rival) on the world.

Both learn from the world only: random scenes and random actions, and the world's true answer after each. Each has
its own curriculum (world.Curriculum), which always serves the edge of what *that* learner can do, by the same rule.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from . import world as W
from .data import laws_batch, rival_batch

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


def train(model, vocab, out: Path, steps=20000, batch=128, lr=6e-4, warmup=300, device="cpu", curriculum=True,
          minutes=None, log_every=50, eval_every=1000, eval_n=24, seed=0, say=print):
    out.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": rest, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=str(device).startswith("cuda"))
    sched = lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))  # noqa: E731
    cur = W.Curriculum(on=curriculum, seed=seed)
    rng = random.Random(seed)
    tag_rng = random.Random(seed + 1)
    metrics = open(out / "metrics.jsonl", "a")
    t0 = time.time()
    seen_eps = 0
    run_loss, run_acc, run_n = 0.0, 0.0, 0
    step = 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * sched(step)
        picks = [cur.pick() for _ in range(batch)]
        eps = [W.make(f, lv, "train", TRAIN_SEED0 + seen_eps + i) for i, (f, lv) in enumerate(picks)]
        seen_eps += batch
        b = batch_for(model, eps, vocab, device, tag_rng)
        kw = {"passes": rng.randint(1, model.cfg.train_max)} if model.kind == "laws" else {}
        with autocast(device):
            loss, hits, _ = model.loss_and_hits(b, **kw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        qh = question_hits(hits, b["meta"])
        by_ep = defaultdict(list)
        for m in b["meta"]:
            by_ep[m["q"]] = (m["family"], m["level"])
        for qid, ok in qh.items():
            f, lv = by_ep[qid]
            cur.update(f, lv, float(ok))
        run_loss += loss.item()
        run_acc += sum(qh.values()) / max(1, len(qh))
        run_n += 1
        el = time.time() - t0
        if (step + 1) % log_every == 0 or step == 0:
            rec = {"kind": "train", "step": step + 1, "loss": run_loss / run_n, "acc": run_acc / run_n,
                   "episodes": seen_eps, "minutes": el / 60, "lr": lr * sched(step), "frontier": cur.state()}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            say(f"{model.kind} step {step + 1}/{steps}  loss {rec['loss']:.4f}  right {100 * rec['acc']:.1f}%  "
                f"{el / 60:.1f} min  frontier " + " ".join(f"{k}:{v}" for k, v in cur.state().items()))
            run_loss, run_acc, run_n = 0.0, 0.0, 0
        if eval_every and (step + 1) % eval_every == 0 and step + 1 < steps:
            res = evaluate(model, vocab, n=eval_n, device=device, sweep=False)
            metrics.write(json.dumps({"kind": "exam", "step": step + 1, "minutes": el / 60, **summarise(res)}) + "\n")
            metrics.flush()
            model.train()
        if minutes and el > minutes * 60:
            say(f"{model.kind}: time is up after {step + 1} steps")
            break
    info = {"steps": step + 1, "episodes": seen_eps, "minutes": (time.time() - t0) / 60, "frontier": cur.state()}
    return info


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
            right, nq, used_sum, rev = 0, 0, 0.0, defaultdict(lambda: [0, 0])
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
                for qid, ok in qh.items():
                    if tags[qid] is not None:
                        rev[tags[qid]][0] += ok
                        rev[tags[qid]][1] += 1
            row = {"family": fam, "split": "seen" if sp == "train" else sp, "level": lv, "mode": mode,
                   "n_q": nq, "right": right / max(1, nq)}
            if model.kind == "laws":
                row["passes"] = used_sum / n
            if rev:
                row["reversed"] = {str(k): v for k, v in rev.items()}
            rows.append(row)
    return rows


def summarise(rows, mode=None):
    """Mean right per family on seen and on secret exams (one mode: settle for Laws, plain for the rival)."""
    out = {}
    for r in rows:
        if r["mode"] not in ("settle", "plain") if mode is None else r["mode"] != mode:
            continue
        key = f"{r['family']}/{'seen' if r['split'] == 'seen' else 'secret'}"
        out.setdefault(key, []).append(r["right"])
    s = {k: sum(v) / len(v) for k, v in out.items()}
    for part in ("seen", "secret"):
        vals = [v for k, v in s.items() if k.endswith("/" + part)]
        if vals:
            s[part] = sum(vals) / len(vals)
    return s
