"""World practice mixed into a normal next-word model's text (João, Nov 9: "a normal transformer, and like 10% of the
dataset is this world"). The model stays a plain NEXT-WORD model: the world's episodes are just more text it reads and
guesses word by word, answers included, exactly as the worldfirst rival learns (worldfirst/README.md).

    python3 -m oazy.train --model bee6 ... --world 0.10       # 10% of the run's tokens are world episodes (0 = off)
    python3 -m oazy.worldmix exam --run bee6-race bee6-world10 # the world's secret exams, asked as plain text

What it reads. worldfirst/world.py's nine families (sums, lines, boxes, family, falling, machines, cups, minds, walks):
a scene, an action, questions about the scene after it, and their answers, written as ordinary sentences:

    There is a line. Vasco is right behind Cora. ... Simao and Zoe swap places. Is Vasco ahead of Simao? Answer: no.

Episodes are made on the fly by a few CPU helper processes (PRACTICE levels only, never a secret exam's, fresh random
names every time, nothing stored), cut into SmolLM2's pieces, joined with <|endoftext|> between them and packed into
the same windows as the text (512 -> 1,024 -> 2,048 words).

How much. `--world w` gives the world 3w of the passes for the first 20% of the run, then w/2 to the end, cool-down
included: w on average (0.10 -> 30% then 5%; 0.05 -> 15% then 2.5%). Whole passes go to the world, spread evenly by a
running tally (as the mentor's share is), so a step's text and world are mixed in its gradient. The run's token count
includes the world's: a world run reads (1 - w) of the plain run's text, at the same compute. The mentor never reads a
world pass (it only teaches text), and the training-loss line stays text-only (the world's loss is logged on its own).

The exam. Every family's secret exams (longer sums and lines, deeper boxes, bigger falls, family links and machine
steps chained in ways never practised, more moves of cups and minds, longer walks), plus fresh practised scenes,
asked as the same plain text: the episode, then each question and "Answer:", and the model's guess among the same
candidates the rival chooses from (yes/no, a digit, or a name or thing of the scene), with the earlier answers given.
A question counts only if every word of its answer is right; every exam set weighs the same (as worldfirst).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np

EARLY, EARLY_X, LATE_X = 0.2, 3.0, 0.5     # the first 20% of the run at 3x the average share, then half of it
TRAIN_SEED0 = 1_000_000                    # practice episodes use seeds from here up; exam episodes 0 .. n-1
PER_JOB = 256                              # episodes per helper job
EXAM_VERSION = 1
PUNCT = {".", "?", ",", ":"}


def share_at(progress: float, w: float) -> float:
    """The world's share of the passes at this point of the run (w = the average over the run)."""
    if w <= 0:
        return 0.0
    return min(1.0, w * (EARLY_X if progress < EARLY else LATE_X))


def plan_text(w: float) -> str:
    return (f"{share_at(0, w):.0%} of the passes for the first {EARLY:.0%}, then {share_at(1, w):.1%} to the end "
            f"({w:.0%} of the run's tokens)")


# ---------------------------------------------------------------- an episode as plain text
def episode_parts(e):
    """The episode with its answers as plain sentences: (text, [(start, end, kind, target, candidates, question)]).

    Each answer word's span starts at its space (" no"), so the text can be cut there and every piece tokenized alone
    (training and the exam cut it the same way). Candidates are what the worldfirst rival chooses among."""
    from worldfirst import world as W
    words = [(w, None) for w in e.tokens]
    scene = set(e.tokens)
    names = sorted({k for k in e.things if k in scene})
    for qi, q in enumerate(e.questions):
        words += [(w, None) for w in q.tokens] + [("answer", None), (":", None)]
        for (kind, tgt), w in zip(q.specs, q.answer):
            if kind == "class":
                cands = ["yes", "no"] if tgt in W.YESNO else [str(i) for i in range(10)]
            elif kind == "pointer":
                cands = names if w in names else sorted(set(names) | {w})
            else:
                cands = [str(i) for i in range(10)]
            words.append((w, (kind, cands, qi)))
        words.append((".", None))
    out, spans, cap = [], [], True
    n = 0
    for w, ans in words:
        if cap and w and w[0].isalpha():
            w = w[0].upper() + w[1:]
        piece = w if (w in PUNCT or not out) else " " + w
        if ans is not None:
            spans.append((n, n + len(piece), ans[0], w, ans[1], ans[2]))
        out.append(piece)
        n += len(piece)
        if w in (".", "?"):
            cap = True
        elif w not in PUNCT:
            cap = False
    return "".join(out), spans


def episode_text(e) -> str:
    return episode_parts(e)[0]


def cut(text, spans):
    """The text in pieces: before each answer word, the answer word, ..., the rest."""
    pieces, at = [], 0
    for s, t, *_ in spans:
        pieces += [text[at:s], text[s:t]]
        at = t
    pieces.append(text[at:])
    return pieces


def encode_episodes(enc, eps):
    """Episodes -> one list of piece ids each (no <|endoftext|>), cut at the answers and tokenized in one batch."""
    parts = [episode_parts(e) for e in eps]
    allp, counts = [], []
    for text, spans in parts:
        p = cut(text, spans)
        allp += p
        counts.append(len(p))
    ids = enc.encode_ordinary_batch(allp)
    out, i = [], 0
    for c in counts:
        out.append([t for x in ids[i:i + c] for t in x])
        i += c
    return out


# ---------------------------------------------------------------- the stream of practice episodes
_ENC = None


def _init(tk):
    global _ENC
    from . import tok
    _ENC = tok.get(tk)


def picks(seed, k):
    """The (family, level) of job k's episodes: every family alike, every practised level alike. Never a secret one."""
    from worldfirst import world as W
    rng = random.Random(f"world|{seed}|{k}")
    out = []
    for _ in range(PER_JOB):
        f = rng.choice(W.FAMILIES)
        out.append((f, rng.choice(W.TRAIN_LEVELS[f])))
    return out


def job(args):
    """One helper job: PER_JOB fresh practice episodes as piece ids, each followed by <|endoftext|>."""
    seed, k, tk = args
    from worldfirst import world as W
    if _ENC is None:
        _init(tk)
    base = TRAIN_SEED0 + seed * 10_000_000_000 + k * PER_JOB
    eps = [W.make(f, lv, "train", base + i) for i, (f, lv) in enumerate(picks(seed, k))]
    eot = _ENC.eot_token
    flat = []
    for ids in encode_episodes(_ENC, eps):
        flat += ids + [eot]
    return np.asarray(flat, dtype=np.int32)


class WorldStream:
    """Windows of practice episodes, made by `workers` helper processes (0 = in this process), packed back to back.

    get(micro, T) -> (x, y), each (micro, T), like the text reader. state() / set_state(): the next job, so a resumed
    run carries on with new episodes (the half-used job at a stop is dropped)."""

    def __init__(self, tk="smollm2", seed=0, workers=2, ahead=8, state=None):
        self.tk, self.seed, self.workers, self.ahead = tk, seed, workers, ahead
        self.k = int((state or {}).get("job", 0))      # the next job to hand out
        self.used = self.k                             # jobs consumed (what a resume restarts from)
        self.buf = np.zeros(0, dtype=np.int32)
        self.pending = deque()
        self.pool = None
        if workers > 0:
            import multiprocessing as mp
            self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(tk,))
        self.tokens = 0

    def _fill(self, need):
        while len(self.buf) < need:
            if self.pool is None:
                arr = job((self.seed, self.k, self.tk))
                self.k += 1
            else:
                while len(self.pending) < self.ahead:
                    self.pending.append(self.pool.apply_async(job, ((self.seed, self.k, self.tk),)))
                    self.k += 1
                arr = self.pending.popleft().get()
            self.used += 1
            self.buf = np.concatenate([self.buf, arr])

    def rows(self, micro, T):
        n = micro * (T + 1)
        self._fill(n)
        out, self.buf = self.buf[:n].reshape(micro, T + 1), self.buf[n:]
        self.tokens += micro * T
        return out

    def get(self, micro, T, device):
        import torch
        r = torch.from_numpy(self.rows(micro, T).astype(np.int64))
        r = r.to(device, non_blocking=True)
        return r[:, :-1], r[:, 1:]

    def state(self):
        return {"job": self.used}

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None


# ---------------------------------------------------------------- the exam, asked as plain text
def exam_sets(secret_only=False):
    from worldfirst import world as W
    return [s for s in W.exam_sets() if not (secret_only and s[2] == "train")]


def _items(enc, e):
    """One exam episode -> (ids with a leading <|endoftext|>, [(start of the answer word's pieces, candidates' pieces,
    the true candidate's index, question)])."""
    text, spans = episode_parts(e)
    pieces = cut(text, spans)
    ids_p = enc.encode_ordinary_batch(pieces)
    ids = [enc.eot_token]
    items = []
    for j, p in enumerate(ids_p):
        if j % 2 == 1:                                 # an answer word
            _, _, kind, w, cands, qi = spans[j // 2]
            cc = enc.encode_ordinary_batch([" " + c for c in cands])
            items.append((len(ids), cc, cands.index(w), qi))
        ids += p
    return ids, items


def _logprobs(model, seqs, device, budget=4096):
    """Every sequence's log-probabilities of its own next pieces: {i: array (len - 1)}. Same-length groups, no padding."""
    import torch
    import torch.nn.functional as F
    from collections import defaultdict
    by_len = defaultdict(list)
    for i, s in enumerate(seqs):
        by_len[len(s)].append(i)
    out = {}
    for T, group in sorted(by_len.items()):
        step = max(1, budget // T)
        for b0 in range(0, len(group), step):
            g = group[b0:b0 + step]
            x = torch.tensor([seqs[i] for i in g], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                logits = model._body(x)
            lp = -F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                                  reduction="none").view(len(g), T - 1)
            for row, i in zip(lp.cpu().numpy(), g):
                out[i] = row
    return out


def _at(model, seqs, wants, device, budget=4096):
    """Log-probabilities of chosen next pieces at chosen places: wants = {i: [(position, [pieces])]} ->
    {(i, position): [log p of each piece after seqs[i][:position + 1]]}. Same-length groups, no padding."""
    import torch
    from collections import defaultdict
    by_len = defaultdict(list)
    for i, s in enumerate(seqs):
        by_len[len(s)].append(i)
    out = {}
    for T, group in sorted(by_len.items()):
        step = max(1, budget // T)
        for b0 in range(0, len(group), step):
            g = group[b0:b0 + step]
            x = torch.tensor([seqs[i] for i in g], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                logits = model._body(x)
            for b, i in enumerate(g):
                for pos, toks in wants.get(i, []):
                    lp = torch.log_softmax(logits[b, pos].float(), -1)
                    out[(i, pos)] = lp[torch.tensor(toks, device=lp.device)].cpu().numpy()
    return out


def exam(model, device, n=100, secret_only=False, sets=None):
    """The world's exams as plain text. -> {"secret", "seen", "families": {fam/part: acc}, "rows": [...], "n"}.

    Each answer word is chosen among its candidates by the model's log-probability of the candidate's pieces after
    the text so far (earlier answers given, as the rival is asked). When the candidates differ only in their last
    piece (yes/no, digits, most names), one reading of the episode scores them all; otherwise each candidate is read."""
    from . import tok
    enc = tok.for_model(model)
    was = model.training
    model.eval()
    import torch
    from worldfirst import world as W
    sets = sets or exam_sets(secret_only)
    eps_by_set = [(s, [W.make(s[0], s[1], s[2], i) for i in range(n)]) for s in sets]
    jobs = []                                # (set index, episode, question, how it is scored, data)
    cand_seqs, cand_of = [], []
    truth, wants = [], {}
    for si, (s, eps) in enumerate(eps_by_set):
        for ei, e in enumerate(eps):
            ids, items = _items(enc, e)
            ti = len(truth)
            truth.append(ids)
            for start, cc, right, qi in items:
                c = 0                         # the pieces every candidate shares (e.g. a lone space before a digit)
                while all(len(x) > c + 1 for x in cc) and len({x[c] for x in cc}) == 1:
                    c += 1
                last = [x[c:] for x in cc]
                if all(len(x) == 1 for x in last) and len({x[0] for x in last}) == len(last):
                    # the truth's reading already holds the guess there (causal): no extra reading needed
                    wants.setdefault(ti, []).append((start + c - 1, [x[0] for x in last]))
                    jobs.append((si, ei, qi, "one", ((ti, start + c - 1), right)))
                else:
                    first = len(cand_seqs)
                    for x in cc:
                        cand_seqs.append(ids[:start] + x)
                        cand_of.append(start)
                    jobs.append((si, ei, qi, "each", (first, len(cc), right)))
    with torch.no_grad():
        nx = _at(model, truth, wants, device) if wants else {}
        lp = _logprobs(model, cand_seqs, device) if cand_seqs else {}
    right_q = {}
    for si, ei, qi, how, d in jobs:
        if how == "one":
            key_, r = d
            sc = list(nx[key_])
        else:
            first, k, r = d
            sc = [float(lp[first + j][cand_of[first + j] - 1:].sum()) for j in range(k)]
        ok = int(np.argmax(sc)) == r
        key = (si, ei, qi)
        right_q[key] = right_q.get(key, True) and ok
    if was:
        model.train()
    rows = []
    for si, (s, eps) in enumerate(eps_by_set):
        qs = [v for (a, _, _), v in right_q.items() if a == si]
        rows.append({"family": s[0], "level": s[1], "split": "seen" if s[2] == "train" else s[2],
                     "n_q": len(qs), "right": sum(qs) / max(1, len(qs))})
    return summarise(rows, n)


def summarise(rows, n):
    fam, part = {}, {}
    for r in rows:
        p = "seen" if r["split"] == "seen" else "secret"
        fam.setdefault(f"{r['family']}/{p}", []).append(r["right"])
        part.setdefault(p, []).append(r["right"])
    out = {"version": EXAM_VERSION, "n": n, "rows": rows,
           "families": {k: sum(v) / len(v) for k, v in fam.items()}}
    for p, v in part.items():
        out[p] = sum(v) / len(v)
    return out


def chance(sets=None, n=40):
    """What guessing would score on the same exams (every candidate alike), for the table."""
    from worldfirst import world as W
    rows = []
    for s in sets or exam_sets():
        tot = []
        for i in range(n):
            e = W.make(s[0], s[1], s[2], i)
            _, spans = episode_parts(e)
            per_q = {}
            for *_, cands, qi in spans:
                per_q[qi] = per_q.get(qi, 1.0) / len(cands)
            tot += list(per_q.values())
        rows.append({"family": s[0], "level": s[1], "split": "seen" if s[2] == "train" else s[2],
                     "n_q": len(tot), "right": sum(tot) / max(1, len(tot))})
    return summarise(rows, n)


# ---------------------------------------------------------------- python3 -m oazy.worldmix exam --run ...
def exam_run(d: Path, device, n=100, say=print):
    """Examine a run folder's checkpoint (any SmolLM2-piece model of the lab) and write <run>/world_exam.json."""
    out = d / "world_exam.json"
    if out.exists():
        try:
            r = json.loads(out.read_text())
            ck = d / "ckpt.pt"
            if r.get("version") == EXAM_VERSION and r.get("n") == n and r.get("ckpt_mtime") == (ck.stat().st_mtime if ck.exists() else None):
                say(f"{d.name}: world exam already done (secret {r['secret'] * 100:.1f}%): kept")
                return r
        except Exception:
            pass
    from .train import load_model
    import time
    t0 = time.time()
    model, st = load_model(d / "ckpt.pt", device)
    if getattr(model.cfg, "tokenizer", "gpt2") != "smollm2":
        raise SystemExit(f"{d.name} reads GPT-2 pieces: the world exam is for SmolLM2-piece models")
    r = exam(model, device, n=n)
    r.update(ckpt_mtime=(d / "ckpt.pt").stat().st_mtime, tokens=st.get("tokens"), seconds=time.time() - t0)
    out.write_text(json.dumps(r, indent=1))
    say(f"{d.name}: world exam secret {r['secret'] * 100:.1f}%, practised {r['seen'] * 100:.1f}% "
        f"({n} episodes per exam set, {time.time() - t0:.0f} s)")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["exam", "show"])
    ap.add_argument("--run", nargs="+", default=["bee6-race"])
    ap.add_argument("--runs", default=None)
    ap.add_argument("--n", type=int, default=100, help="episodes per exam set")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.what == "show":                     # a few practice episodes exactly as the model reads them
        from worldfirst import world as W
        for f, lv in picks(0, 0)[:9]:
            print(f"[{f} {lv}]", episode_text(W.make(f, lv, "train", 123)), "\n")
        return
    import torch
    from .data import default_dir, run_dir
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    runs = Path(a.runs or default_dir("runs"))
    out = {}
    for rn in a.run:
        d = run_dir(runs, rn)
        if not (d / "ckpt.pt").exists():
            print(f"{rn}: no checkpoint yet, skipped")
            continue
        out[rn] = exam_run(d, dev, a.n)
    return out


if __name__ == "__main__":
    main()
