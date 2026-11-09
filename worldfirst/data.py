"""Words to numbers, and episodes to batches, for both models.

Both models read the very same words. What differs is only how each is asked:

- The Laws model reads the scene, the action and the questions (never the answers), keeps one vector per thing
  and one per question, and answers from those: yes/no or a digit (a "class"), a thing ("pointer"), or, for a sum,
  one digit per column ("coldigit").
- The rival reads the whole text, answers included, and learns to guess every next word. It is asked the same
  questions by looking at what it guesses right after "answer :", among the same candidates (yes/no, digits, or
  the names of the things in the scene).

Both get the same help with names: the world marks which words name the same thing. The Laws model pools those
words into the thing's vector; the rival gets a random tag per thing added to those words (a different random tag
every episode, so the tags only say "same thing", nothing more).
"""
from __future__ import annotations

import random

import torch

from . import world as W

SPECIALS = ["<pad>", "<bos>", "<unk>"]
PAD, BOS, UNK = 0, 1, 2
N_TAGS = 64


def build_vocab(seeds: int = 60):
    words = set(W.NAMES) | set(W.COLOURS) | set(W.CONTAINERS) | set(W.ITEMS) | set(W.CLASSES) | {"answer", ":", ".", "?"}
    words |= W.extra_words()                                               # World 2's rooms, events, machines ...
    if W.STYLE["natural"]:                                                 # the natural way of speaking
        words |= set(W.PRONOUNS) | {",", "smallest", "digit", "first"}
    for fam, lv, sp in W.exam_sets():
        for s in range(seeds):
            e = W.make(fam, lv, sp, 10_000_000 + s)
            words.update(e.tokens)
            for q in e.questions:
                words.update(q.tokens)
                words.update(q.answer)
    return SPECIALS + sorted(words)


class Vocab:
    def __init__(self, words=None):
        self.words = list(words) if words is not None else build_vocab()
        self.id = {w: i for i, w in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    def enc(self, toks):
        return [self.id.get(t, UNK) for t in toks]


def _item_meta(ep, qi, uid):
    q = ep.questions[qi]
    return {"family": ep.family, "level": ep.level, "split": ep.split, "q": uid, **q.tags}


def laws_batch(eps, vocab: Vocab, device="cpu"):
    """Reader text without answers; slots = things then questions; answer items in a fixed order."""
    B = len(eps)
    seqs, pools, poss, kinds = [], [], [], []
    cls, ptr, col, meta = [], [], [], []
    uid = 0
    for b, e in enumerate(eps):
        toks = list(e.tokens)
        ment = list(e.mentions)
        spans = []
        for q in e.questions:
            spans.append((len(toks), len(toks) + len(q.tokens), q))
            toks += q.tokens
            ment += q.mentions
        seqs.append(vocab.enc(toks))
        n = len(e.things)
        slot_tok = [[] for _ in range(n)]
        for i, t in enumerate(e.mentions):          # things are pooled from the scene and the action only
            if t >= 0:
                slot_tok[t].append(i)
        for t in range(n):
            if not slot_tok[t]:                     # named only in a question: pool those words instead
                slot_tok[t] = [i for i, m in enumerate(ment) if m == t]
        for (s0, s1, _) in spans:
            slot_tok.append(list(range(s0, s1)))
        pools.append(slot_tok)
        poss.append([(ix[0] if ix else 0) for ix in slot_tok])
        kinds.append([1] * n + [2] * len(spans))
        for qi, (s0, s1, q) in enumerate(spans):
            qslot = n + qi
            for kind, tgt in q.specs:
                m = _item_meta(e, qi, uid)
                if kind == "class":
                    cls.append((b, qslot, tgt, len(meta)))
                elif kind == "pointer":
                    ptr.append((b, qslot, tgt, len(meta)))
                else:
                    thing, digit = tgt
                    col.append((b, thing, digit, len(meta)))
                meta.append(m)
            uid += 1
    T = max(len(s) for s in seqs)
    S = max(len(p) for p in pools)
    tok = torch.zeros(B, T, dtype=torch.long)
    pool = torch.zeros(B, S, T)
    pos = torch.zeros(B, S, dtype=torch.long)
    kind = torch.zeros(B, S, dtype=torch.long)
    nthing = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        tok[b, :len(seqs[b])] = torch.tensor(seqs[b])
        for j, ix in enumerate(pools[b]):
            if ix:
                pool[b, j, ix] = 1.0 / len(ix)
        pos[b, :len(poss[b])] = torch.tensor(poss[b])
        kind[b, :len(kinds[b])] = torch.tensor(kinds[b])
        nthing[b] = len(eps[b].things)
    t = lambda rows, k: torch.tensor([r[k] for r in rows], dtype=torch.long)  # noqa: E731
    out = {"tok": tok, "pool": pool, "pos": pos, "kind": kind, "nthing": nthing}
    for name, rows in (("cls", cls), ("ptr", ptr), ("col", col)):
        out[name] = {"b": t(rows, 0), "slot": t(rows, 1), "y": t(rows, 2), "item": t(rows, 3)}
    out = {k: (v.to(device) if torch.is_tensor(v) else {kk: vv.to(device) for kk, vv in v.items()}) for k, v in out.items()}
    out["meta"] = meta
    return out


def finder_batch(eps, vocab: Vocab, device="cpu"):
    """For the Laws model WITHOUT marks: the same reader text, but nothing says which words name which thing.

    Questions are still found (each ends with "?"), and answers are asked as: a class (yes/no/digit) from the
    question, a "who/what" as a word of the scene (any word that is the answer's name counts), and a sum as a short
    sequence of digits that a small speaker writes. `ment` (which thing each word names) is kept only to check
    afterwards how well the model found the things; the model never sees it."""
    B = len(eps)
    V = len(vocab)
    seqs, qspans, ments = [], [], []
    cls, ptr, sq, meta = [], [], [], []
    uid = 0
    for b, e in enumerate(eps):
        toks, ment = list(e.tokens), list(e.mentions)
        spans = []
        for q in e.questions:
            spans.append((len(toks), len(toks) + len(q.tokens), q))
            toks += q.tokens
            ment += q.mentions
        seqs.append(vocab.enc(toks))
        ments.append(ment)
        qspans.append(spans)
        names = sorted({vocab.id[k] for k in e.things if k in vocab.id})
        for qi, (s0, s1, q) in enumerate(spans):
            kinds = {k for k, _ in q.specs}
            if kinds == {"coldigit"}:
                items = []
                for _ in q.specs:
                    items.append(len(meta))
                    meta.append(_item_meta(e, qi, uid))
                sq.append((b, qi, [int(w) for w in q.answer], items, s1))
            else:
                for (kind, tgt), w in zip(q.specs, q.answer):
                    if kind == "class":
                        cls.append((b, qi, tgt, len(meta)))
                    else:
                        ptr.append((b, qi, vocab.id[w], len(meta), names))
                    meta.append(_item_meta(e, qi, uid))
            uid += 1
    T = max(len(x) for x in seqs)
    Q = max(len(x) for x in qspans)
    tok = torch.zeros(B, T, dtype=torch.long)
    ment = torch.full((B, T), -1, dtype=torch.long)
    qpool = torch.zeros(B, Q, T)
    qpos = torch.zeros(B, Q, dtype=torch.long)
    qvalid = torch.zeros(B, Q, dtype=torch.bool)
    for b in range(B):
        tok[b, :len(seqs[b])] = torch.tensor(seqs[b])
        ment[b, :len(ments[b])] = torch.tensor(ments[b])
        for j, (s0, s1, _) in enumerate(qspans[b]):
            qpool[b, j, s0:s1] = 1.0 / (s1 - s0)
            qpos[b, j] = s0
            qvalid[b, j] = True
    t = lambda rows, k: torch.tensor([r[k] for r in rows], dtype=torch.long)  # noqa: E731
    slen = torch.tensor([len(e.tokens) for e in eps], dtype=torch.long)   # where the questions start (seen in the text)
    out = {"tok": tok, "ment": ment, "qpool": qpool, "qpos": qpos, "qvalid": qvalid, "slen": slen,
           "cls": {"b": t(cls, 0), "q": t(cls, 1), "y": t(cls, 2), "item": t(cls, 3)},
           "ptr": {"b": t(ptr, 0), "q": t(ptr, 1), "y": t(ptr, 2), "item": t(ptr, 3)}}
    cand = torch.zeros(len(ptr), V, dtype=torch.bool)
    for i, r in enumerate(ptr):
        cand[i, r[4]] = True
    out["ptr"]["cand"] = cand
    L = max([len(r[2]) for r in sq], default=1)
    dig = torch.full((len(sq), L), -1, dtype=torch.long)
    item = torch.full((len(sq), L), -1, dtype=torch.long)
    for i, r in enumerate(sq):
        dig[i, :len(r[2])] = torch.tensor(r[2])
        item[i, :len(r[3])] = torch.tensor(r[3])
    out["seq"] = {"b": t(sq, 0), "q": t(sq, 1), "end": t(sq, 4), "y": dig, "item": item}
    out = {k: (v.to(device) if torch.is_tensor(v) else {kk: vv.to(device) for kk, vv in v.items()}) for k, v in out.items()}
    out["meta"] = meta
    return out


def rival_batch(eps, vocab: Vocab, device="cpu", tags=True, rng=None):
    """Whole text with answers, for next-word learning; plus where each answer word is guessed, and its candidates."""
    rng = rng or random.Random(0)
    B = len(eps)
    V = len(vocab)
    seqs, tagss, ments, items, meta = [], [], [], [], []
    yes_no = [vocab.id["yes"], vocab.id["no"]]
    digits = [vocab.id[str(i)] for i in range(10)]
    uid = 0
    for b, e in enumerate(eps):
        perm = rng.sample(range(1, N_TAGS), len(e.things)) if len(e.things) < N_TAGS else list(range(1, len(e.things) + 1))
        toks = ["<bos>"] + list(e.tokens)
        tg = [0] + [perm[m] if m >= 0 else 0 for m in e.mentions]
        mt = [-1] + list(e.mentions)               # which thing each word names: only an answer key (see teach)
        names = sorted({vocab.id[k] for k in e.things if k in vocab.id})
        for qi, q in enumerate(e.questions):
            toks += q.tokens + ["answer", ":"]
            tg += [perm[m] if m >= 0 else 0 for m in q.mentions] + [0, 0]
            mt += list(q.mentions) + [-1, -1]
            for (kind, tgt), w in zip(q.specs, q.answer):
                if kind == "class":
                    cands = yes_no if tgt in W.YESNO else digits
                elif kind == "pointer":
                    cands = names
                else:
                    cands = digits
                items.append((b, len(toks) - 1, vocab.id[w], cands))
                meta.append(_item_meta(e, qi, uid))
                toks.append(w)
                tg.append(perm[tgt] if kind == "pointer" else 0)
                mt.append(-1)
            toks.append(".")
            tg.append(0)
            mt.append(-1)
            uid += 1
        seqs.append(vocab.enc(toks))
        tagss.append(tg)
        ments.append(mt)
    T = max(len(s) for s in seqs)
    tok = torch.zeros(B, T, dtype=torch.long)
    tag = torch.zeros(B, T, dtype=torch.long)
    ment = torch.full((B, T), -1, dtype=torch.long)
    for b in range(B):
        tok[b, :len(seqs[b])] = torch.tensor(seqs[b])
        ment[b, :len(ments[b])] = torch.tensor(ments[b])
        if tags:
            tag[b, :len(tagss[b])] = torch.tensor(tagss[b])
    cand = torch.zeros(len(items), V, dtype=torch.bool)
    for i, it in enumerate(items):
        cand[i, it[3]] = True
    out = {"tok": tok.to(device), "tag": tag.to(device), "ment": ment.to(device), "cand": cand.to(device),
           "ib": torch.tensor([it[0] for it in items], device=device),
           "ipos": torch.tensor([it[1] for it in items], device=device),
           "iy": torch.tensor([it[2] for it in items], device=device), "meta": meta}
    return out
