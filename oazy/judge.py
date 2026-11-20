"""The judge's lesson: real text in which some sentences, and some sentence endings, have changed places.

A next-word model is graded on writing: one right word per piece of text. A judge is graded on a harder question about
the same text: "does what I am reading belong here?" Decoys makes the text for that question on the CPU, row by row,
just before a batch goes to the GPU:

  whole   two sentences of the row change places (mostly two of the SAME document: on topic, real, in the wrong place)
  tail    two sentences exchange their endings, cut where a word starts, and where it can be after the same word
          ("... walked into the | kitchen."  <->  "... picked up the | ball and ran.")

Nothing is written by a machine: every decoy is a real stretch of the same row, so there is no style to spot, only the
sense. A swap keeps the row's length. Each piece of the row gets a mark (the judge's answer key):

   1   real text in its place
   0   a piece of a stretch that was put here (a decoy)
  -1   not graded: the sentence right after a decoy (it is real, but what it follows is not)

The model (bee6/model.py, `judge=`) reads the marks with a yes/no head on every piece, and the next-word loss is not
asked to write a decoy. A sentence closes at a piece that ends in . ! or ? (closing quotes and brackets after it
allowed) when the next piece starts a new word or line; a document closes at <|endoftext|>.

    python3 -m oazy.judge_tests
"""
from __future__ import annotations

import numpy as np

CLOSERS = (b'"', b"'", b")", b"]", b"}", "”".encode(), "’".encode(), b"*", b"_")
STOPS = (b".", b"!", b"?")


def _core(b: bytes):
    """A piece's bytes without the white space at its end, and then without closing quotes and brackets."""
    s = b.rstrip(b" \n\t\r")
    while s.endswith(CLOSERS):
        s = s[:-next(len(c) for c in CLOSERS if s.endswith(c))]
    return s


class Decoys:
    MIN, MAX = 4, 96            # a sentence that can be swapped: this many pieces
    PRE, TAIL = 2, 3            # a tail swap keeps at least PRE pieces of the sentence and moves at least TAIL

    def __init__(self, tokenizer: str = "smollm2", rate: float = 0.15, tail: float = 0.5, near: float = 0.75):
        """rate: the share of the swappable sentences that become decoys; tail: the share of swaps that exchange only
        endings; near: how often the partner is taken from the same document when there is one."""
        from . import tok
        enc = tok.get(tokenizer)
        V = enc.n_vocab
        self.eot = tok.eot_id(tokenizer)
        self.rate, self.tail, self.near = float(rate), float(tail), float(near)
        special = set(getattr(enc, "_special", ()) or ())
        hard, soft, gap, word, bad, closer = (np.zeros(V, dtype=bool) for _ in range(6))
        kind = np.full(V, 2, dtype=np.int8)                # how a piece starts: 0 a space, 1 a new line, 2 anything else
        for t in range(V):
            b = enc.decode_single_token_bytes(t)
            if not b or t == self.eot:
                bad[t] = not b
                continue
            if b.decode("utf-8", "replace") in special or (b.startswith(b"<|") and b.endswith(b"|>")):
                bad[t] = True                              # a chat mark and the like: its sentence is left alone
                continue
            closes = _core(b).endswith(STOPS)
            closer[t] = _core(b) == b"" and b.strip() == b     # only closing quotes or brackets: '"', ')', '")'
            hard[t] = closes and b[-1:] in (b" ", b"\n", b"\t", b"\r")       # ". " or ".\n": closed whatever follows
            soft[t] = closes and not hard[t]                                   # ".": closed if a new word or line follows
            gap[t] = b[:1] in (b" ", b"\n", b"\t", b"\r")
            kind[t] = 0 if b[:1] == b" " else 1 if b[:1] in (b"\n", b"\r") else 2
            word[t] = len(b) > 1 and b[:1] == b" " and (b[1:2].isalnum() or b[1] >= 0x80)
        self.hard, self.soft, self.gap, self.word, self.bad, self.kind, self.closer = hard, soft, gap, word, bad, kind, closer
        self.rows = self.pieces = self.decoys = self.skipped = 0      # counted since the start (report())

    # ------------------------------------------------------------------ one row
    def spans(self, r):
        """(start of each sentence, its last piece, whether it can be swapped, its document, how it starts) for the
        row's closed stretches; the unfinished stretch at the row's end is none of them."""
        n = len(r)
        eot = r == self.eot
        nxt = np.zeros(n, dtype=bool)
        nxt[:-1] = self.gap[r[1:]] | eot[1:]
        stop = self.soft[r]
        for _ in range(2):                                 # '.' then '"' then ')' as pieces of their own: closed at the last
            more = np.zeros(n, dtype=bool)
            more[1:] = stop[:-1]
            stop = stop | (self.closer[r] & more)
        close = self.hard[r] | (stop & nxt)
        stops = np.flatnonzero(close | eot)
        if len(stops) == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z, np.zeros(0, dtype=bool), z, z
        starts = np.empty_like(stops)
        starts[0], starts[1:] = 0, stops[:-1] + 1
        L = stops - starts + 1
        b = (self.bad[r] | eot).astype(np.int64)
        c = np.cumsum(b)
        n_bad = c[stops] - c[starts] + b[starts]
        ok = close[stops] & (n_bad == 0) & (L >= self.MIN) & (L <= self.MAX)
        ok[0] = False                                      # the row's first stretch starts mid-sentence
        return starts, stops, ok, np.cumsum(eot)[starts], self.kind[r[starts]]

    def _pick(self, cands, near, rng):
        if len(cands) == 0:
            return None
        nc = cands[near]
        if len(nc) and rng.random() < self.near:
            return int(nc[rng.integers(len(nc))])
        return int(cands[rng.integers(len(cands))])

    def row(self, r, rng):
        """One row of pieces -> (the row with its swaps, the marks: int8, the row's length)."""
        r = np.asarray(r)
        ri = r.astype(np.int64)
        n = len(ri)
        lab = np.ones(n, dtype=np.int8)
        starts, stops, ok, doc, kind = self.spans(ri)
        ids = np.flatnonzero(ok)
        want = int(self.rate * len(ids) / 2 + rng.random())
        self.rows += 1
        self.pieces += n
        if want == 0 or len(ids) < 2:
            return r, lab
        S = len(stops)
        L = stops - starts + 1
        sent = np.full(n, S, dtype=np.int64)               # each piece's sentence (S: the unfinished stretch at the end)
        sent[:stops[-1] + 1] = np.repeat(np.arange(S), L)
        free = np.append(ok, False)
        st, sp, dc = np.append(starts, n), np.append(stops, n), np.append(doc, -1)
        pos = np.arange(n)
        cut = self.word[ri] & free[sent] & (pos - st[sent] >= self.PRE) & (sp[sent] - pos + 1 >= self.TAIL)
        prev = np.empty(n, dtype=np.int64)
        prev[0], prev[1:] = -1, ri[:-1]
        swaps = []                                         # ((a0, a1), (b0, b1), sentence a, sentence b): [a0, a1) <-> [b0, b1)
        for i in rng.permutation(ids):
            if len(swaps) >= want:
                break
            if not free[i]:
                continue
            done = None
            if rng.random() < self.tail:
                cuts = np.flatnonzero(cut[starts[i]:stops[i] + 1]) + starts[i]
                open_ = cut & free[sent] & (sent != i)
                for c in (rng.permutation(cuts)[:3] if len(cuts) else ()):
                    cands = np.flatnonzero(open_ & (prev == prev[c]))
                    q = self._pick(cands, dc[sent[cands]] == doc[i], rng)
                    if q is not None:
                        done = ((int(c), int(stops[i]) + 1), (q, int(sp[sent[q]]) + 1), int(i), int(sent[q]))
                        break
                if done is None and len(cuts):             # no ending after the same word: any other sentence's ending
                    cands = np.flatnonzero(open_)
                    q = self._pick(cands, dc[sent[cands]] == doc[i], rng)
                    if q is not None:
                        c = int(cuts[rng.integers(len(cuts))])
                        done = ((c, int(stops[i]) + 1), (q, int(sp[sent[q]]) + 1), int(i), int(sent[q]))
            if done is None:
                m = free[:S] & (kind == kind[i])
                m[i] = False
                cands = np.flatnonzero(m)
                j = self._pick(cands, doc[cands] == doc[i], rng)
                if j is None:
                    continue
                done = ((int(starts[i]), int(stops[i]) + 1), (int(starts[j]), int(stops[j]) + 1), int(i), j)
            (a0, a1), (b0, b1), i_, j_ = done
            if a1 - a0 == b1 - b0 and np.array_equal(ri[a0:a1], ri[b0:b1]):
                self.skipped += 1                          # the same words twice: a swap would mark real text as a decoy
                continue
            free[i_] = free[j_] = False
            swaps.append(done)
        if not swaps:
            return r, lab
        quiet = np.zeros(n, dtype=bool)                    # the sentence right after a decoy: real, but not graded
        reps = []
        for a, b, i_, j_ in swaps:
            reps += [(a[0], a[1], b[0], b[1]), (b[0], b[1], a[0], a[1])]
            for s in (i_, j_):
                if s + 1 < S:
                    quiet[starts[s + 1]:stops[s + 1] + 1] = True
                else:
                    quiet[stops[-1] + 1:] = True
        lab[quiet] = -1
        reps.sort()
        out, marks, at = [], [], 0
        for s0, s1, t0, t1 in reps:
            out += [r[at:s0], r[t0:t1]]
            marks += [lab[at:s0], np.zeros(t1 - t0, dtype=np.int8)]
            at = s1
        out.append(r[at:])
        marks.append(lab[at:])
        self.decoys += 2 * len(swaps)
        return np.concatenate(out), np.concatenate(marks)

    # ------------------------------------------------------------------ a batch
    def batch(self, rows, rng):
        """[row, ...] -> ([row with its swaps, ...], the marks (B, T) int8)."""
        got = [self.row(r, rng) for r in rows]
        return [g[0] for g in got], np.stack([g[1] for g in got])

    def report(self):
        """Decoys made per 1,000 pieces since the start (for the run's log)."""
        return {"judge_decoys_per_1k": 1000.0 * self.decoys / max(1, self.pieces)}
