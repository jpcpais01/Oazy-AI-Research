"""Gnosis2's finish: learning to think its way to an answer, graded by the simulator (João, Nov 11).

Pre-training teaches Gnosis2 what reasoning looks like (Opal2's worked Field1 and its thinking text). This lesson
teaches it to do it: it is given a fresh Field1 scene and a question, it writes its own work and its answer, and the
simulator, which knows the truth, says right or wrong. Nothing to copy; only getting it right counts.

    <a scene it has never seen> <the question> Let me work it out step by step.  ->  <its work> So the answer is <X>.

How (GRPO, the recipe behind DeepSeek-R1 and every reasoning model since, at our size):
  - each step draws `prompts` fresh questions (19 families, the ones it gets right about half the time more often:
    those are the ones it can learn from) and writes `group` answers to each, sampled
  - each answer's reward is 1 if it ends on the right answer, else 0; its advantage is its reward minus the group's
    average (right where the others were wrong counts most; a question everyone gets right or wrong teaches nothing)
  - the model is pushed toward the words of answers that did better than their group, and kept near where it started
    by a small KL penalty against a frozen copy (beta), so its English does not drift
  - every `every` steps (and at the start and end), 190 held-out questions are answered greedily and scored, by family

Why it may break the ceiling: it needs no more text, only a grader, and Field1 is a perfect one with no end of fresh
questions; and it is the only lesson that rewards trying to think rather than sounding right. RLP (NVIDIA, ICLR 2026)
put a thinking reward into pre-training and beat a run that read 35x more data. The warning at our size: reinforcement
sharpens what pre-training put there, which is why Opal2 carries the worked reasoning first.

The result is a run of its own, <run>-reason, with the same checkpoint format, so the dashboard, the world exams
(secret, practised, Real World, Pro: never trained on, so they measure transfer), the benchmarks and chat all read it:

    python3 -m gnosis2.rl --run gnosis2-10b                       # ~2-3 h on an H200 (400 steps)
    OAZY_RUNS=/workspace/runs python3 -m oazy.worldmix exam --run gnosis2-10b-reason --run gnosis2-10b
    python3 -m oazy.evals --run gnosis2-10b-reason --run gnosis2-10b

It works on any model of the Bee family with a reader (read/generate): Gnosis, Bee6 too.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def say(*a, **k):
    from oazy.train import say as s
    s(*a, **k)


# ---------------------------------------------------------------- questions
class Questions:
    """Fresh Field1 questions; each family drawn by how much it can still teach (success near half)."""

    def __init__(self, seed=0, families=None):
        from oazy import field1 as F1
        self.fams = list(families or F1.FAMILIES)
        self.p = {f: 0.5 for f in self.fams}       # running success per family
        self.seed, self.n = seed, 0

    def weight(self, f):
        p = self.p[f]
        return 0.05 + p * (1 - p)

    def draw(self, k):
        from oazy.field1 import work
        r = random.Random(f"rl|{self.seed}|{self.n}")
        fams = r.choices(self.fams, weights=[self.weight(f) for f in self.fams], k=k)
        out = []
        for f in fams:
            out.append(work.question(f, ("rl", self.seed, self.n)))
            self.n += 1
        return out

    def update(self, fam, acc, rate=0.1):
        self.p[fam] += rate * (acc - self.p[fam])


def held_out(n_per=10, seed=0):
    from oazy import field1 as F1
    from oazy.field1 import work
    return [work.question(f, ("rl-val", seed, i)) for f in F1.FAMILIES for i in range(n_per)]


# ---------------------------------------------------------------- writing answers
def left_pad(seqs, pad):
    L = max(len(s) for s in seqs)
    return [[pad] * (L - len(s)) + list(s) for s in seqs], L


def stops(enc):
    """Pieces that end an answer: <|endoftext|> and any piece holding a new line (the work is one line)."""
    nl = set()
    for t in range(enc.n_vocab):
        try:
            b = enc.decode_single_token_bytes(t)
        except Exception:
            continue
        if b"\n" in b:
            nl.add(t)
    nl.add(enc.eot_token)
    return nl


@torch.no_grad()
def write(model, prompts_ids, enc, stop_ids, max_new, temperature, device, greedy=False, chunk=128):
    """Answer every prompt (lists of pieces) -> list of answer pieces (cut before the stop piece). Prompts are padded on
    the left with <|endoftext|> so a whole batch reads at once (the model has read many documents after one)."""
    model.eval()
    outs = []
    stop_t = torch.tensor(sorted(stop_ids), device=device)
    for c in range(0, len(prompts_ids), chunk):
        part = prompts_ids[c:c + chunk]
        rows, _ = left_pad(part, enc.eot_token)
        x = torch.tensor(rows, device=device)
        from oazy.train import amp
        with amp(device):
            logits, st = model.read(x)
        B = x.shape[0]
        gen = torch.zeros(B, 0, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        V = model.cfg.vocab_size
        for _ in range(max_new):
            lg = logits[:, -1, :V].float()
            if greedy:
                nxt = lg.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(lg / temperature, -1), 1)[:, 0]
            nxt = torch.where(done, torch.full_like(nxt, enc.eot_token), nxt)
            gen = torch.cat([gen, nxt[:, None]], 1)
            done = done | torch.isin(nxt, stop_t)
            if bool(done.all()):
                break
            with amp(device):
                logits, st = model.read(nxt[:, None], st)
        for row in gen.tolist():
            cut = next((i for i, t in enumerate(row) if t in stop_ids), len(row))
            outs.append(row[:cut])
    return outs


def score(model, qs, enc, stop_ids, max_new, device):
    from oazy.field1 import work
    ans = write(model, [enc.encode_ordinary(q["prompt"]) for q in qs], enc, stop_ids, max_new, 1.0, device, greedy=True)
    by = {}
    for q, a in zip(qs, ans):
        by.setdefault(q["family"], []).append(work.grade(enc.decode(a), q["answer"]))
    allv = [v for vs in by.values() for v in vs]
    return sum(allv) / max(len(allv), 1), {f: sum(v) / len(v) for f, v in by.items()}, enc.decode(ans[0]) if ans else ""


# ---------------------------------------------------------------- the lesson
def token_logps(model, seqs, n_prompt, device, micro=16):
    """Sum of log-probabilities of each sequence's answer pieces (after n_prompt[i] pieces), with grad."""
    from oazy.train import amp
    out = []
    for c in range(0, len(seqs), micro):
        part, npr = seqs[c:c + micro], n_prompt[c:c + micro]
        L = max(len(s) for s in part)
        x = torch.zeros(len(part), L, dtype=torch.long, device=device)
        m = torch.zeros(len(part), L - 1, dtype=torch.float32, device=device)
        for i, (s, k) in enumerate(zip(part, npr)):
            x[i, :len(s)] = torch.tensor(s, device=device)
            m[i, k - 1:len(s) - 1] = 1.0
        with amp(device):
            z, _, _ = model(x[:, :-1])
        lp = torch.log_softmax(z.float(), -1).gather(-1, x[:, 1:, None])[..., 0]
        out.append((lp * m, m))
    return out


def main(argv=None):
    from oazy.data import default_dir, run_dir
    from oazy.train import load_model, save_ckpt, NICE
    from oazy import tok
    from oazy.field1 import work
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the pre-trained run (runs/<run>/ckpt.pt)")
    ap.add_argument("--name", default=None, help="the result's run (default <run>-reason)")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--prompts", type=int, default=16, help="questions per step")
    ap.add_argument("--group", type=int, default=8, help="answers written to each question")
    ap.add_argument("--max-new", type=int, default=256, help="pieces an answer may take")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-6, help="AdamW's rate (warm-up 10 steps, then flat, cosine over the last 20%%)")
    ap.add_argument("--beta", type=float, default=0.02, help="the KL penalty against the starting model")
    ap.add_argument("--every", type=int, default=50, help="score the held-out questions every this many steps")
    ap.add_argument("--val-per", type=int, default=10, help="held-out questions per family")
    ap.add_argument("--micro", type=int, default=8, help="answers per pass when learning")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    runs = Path(a.runs or default_dir("runs"))
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(a.seed)
    src = run_dir(runs, a.run) / "ckpt.pt"
    model, st = load_model(src, device)
    if not hasattr(model, "read"):
        raise SystemExit(f"{src}: this lesson is for the Bee family (it needs a model that reads on, word by word)")
    if getattr(model.cfg, "fp8", 0):
        model.cfg.fp8 = 0                           # bf16 for the lesson: small steps, no FP8 rounding in the way
    model._compiled = None
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    enc = tok.for_model(model)
    stop_ids = stops(enc)
    out = runs / (a.name or f"{a.run}-reason")
    out.mkdir(parents=True, exist_ok=True)
    kind = st.get("kind", "bee")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, eps=1e-8)
    qs = Questions(a.seed)
    val = held_out(a.val_per)
    log = open(out / "rl.jsonl", "a")
    say(f"[reason] {NICE.get(kind, kind)} learns to work it out: {src} -> {out}\n  {a.steps} steps x {a.prompts} questions "
        f"x {a.group} answers, up to {a.max_new} pieces each; AdamW {a.lr:.1e}, KL {a.beta}; held-out {len(val)} questions")
    acc0, fam0, ex = score(model, val, enc, stop_ids, a.max_new, device)
    say(f"[reason] before: held-out {acc0:.1%}  ·  e.g. {ex[:160]!r}")
    log.write(json.dumps({"kind": "val", "step": 0, "acc": acc0, "families": fam0}) + "\n")
    t0, hist = time.time(), []
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 10) * (1.0 if step < 0.8 * a.steps else 0.5 * (1 + math.cos(math.pi * (step - 0.8 * a.steps) / (0.2 * a.steps))))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = qs.draw(a.prompts)
        p_ids = [enc.encode_ordinary(q["prompt"]) for q in batch]
        rep = [p for p in p_ids for _ in range(a.group)]
        ans = write(model, rep, enc, stop_ids, a.max_new, a.temperature, device)
        rewards = [work.grade(enc.decode(x), batch[i // a.group]["answer"]) for i, x in enumerate(ans)]
        adv = []
        for i, q in enumerate(batch):
            rs = rewards[i * a.group:(i + 1) * a.group]
            mean = sum(rs) / len(rs)
            qs.update(q["family"], mean)
            adv += [r - mean for r in rs]
        keep = [i for i, v in enumerate(adv) if v != 0.0]
        stats = {"kind": "step", "step": step, "reward": sum(rewards) / len(rewards), "lr": lr,
                 "len": sum(len(x) for x in ans) / len(ans), "useful": len(keep) / len(ans)}
        if keep:
            model.eval()                # the same reading as when it wrote (Mixture of Depths by the helper), now with grad
            seqs = [rep[i] + ans[i] + [enc.eot_token] for i in keep]
            npr = [len(rep[i]) for i in keep]
            A = torch.tensor([adv[i] for i in keep], device=device)
            n_tok = sum(len(s) - k for s, k in zip(seqs, npr))
            pg_tot = kl_tot = 0.0
            for c in range(0, len(seqs), a.micro):
                part, kp = seqs[c:c + a.micro], npr[c:c + a.micro]
                (lp, m), = token_logps(model, part, kp, device, micro=len(part))
                with torch.no_grad():
                    (lr_, _), = token_logps(ref, part, kp, device, micro=len(part))
                d = (lr_ - lp) * m
                kl = (torch.exp(d) - d - 1) * m                     # k3, always >= 0
                pg = -(A[c:c + a.micro, None] * lp * m)
                loss = (pg.sum() + a.beta * kl.sum()) / n_tok
                loss.backward()
                pg_tot += float(pg.sum().detach()) / n_tok
                kl_tot += float(kl.sum().detach()) / n_tok
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            stats.update(kl=kl_tot, gnorm=float(gn))
        hist.append(stats["reward"])
        log.write(json.dumps(stats) + "\n")
        if step % 5 == 0 or step == 1:
            el = time.time() - t0
            say(f"step {step:4d}/{a.steps} | reward {sum(hist[-10:]) / len(hist[-10:]):.1%} (last 10) | answers {stats['len']:.0f} pieces | "
                f"useful {stats['useful']:.0%} | KL {stats.get('kl', 0):.4f} | {el / step:.1f} s/step | "
                f"{(a.steps - step) * el / step / 60:.0f} min left", flush=True)
        if step % a.every == 0 or step == a.steps:
            acc, fam, ex = score(model, val, enc, stop_ids, a.max_new, device)
            hard = sorted(fam.items(), key=lambda kv: kv[1])[:4]
            say(f"[reason] step {step}: held-out {acc:.1%} (was {acc0:.1%}); hardest " + ", ".join(f"{f} {v:.0%}" for f, v in hard))
            log.write(json.dumps({"kind": "val", "step": step, "acc": acc, "families": fam}) + "\n")
            log.flush()
            meta = {k: v for k, v in st.items() if k not in ("model", "opt")}
            meta["reason"] = {"from": str(src), "steps": step, "prompts": a.prompts, "group": a.group, "lr": a.lr,
                              "beta": a.beta, "held_out_before": acc0, "held_out": acc, "families": fam,
                              "families_before": fam0, "minutes": (time.time() - t0) / 60}

            class _Opt:
                @staticmethod
                def state_dict():
                    return {}
            save_ckpt(type("R", (), {"dir": out})(), model, _Opt, meta)
            (out / "reason.json").write_text(json.dumps(meta["reason"], indent=1))
    say(f"[reason] done: held-out {acc0:.1%} -> {acc:.1%}. checkpoint {out / 'ckpt.pt'}\n  world exams: "
        f"python3 -m oazy.worldmix exam --run {out.name} --run {a.run}")
    return acc0, acc


if __name__ == "__main__":
    main()
