"""Strata, first piece: memory (exact counts) + abstraction (classes as slots) + one chain
from the most specific pattern to the most abstract. No weights, no gradients.

A pattern is a context in which each of the last n positions is kept at one abstraction
level: 0 = the word itself, 1..L = its class at layer 1..L (coarser each layer), DROP = any.
A chain is a list of patterns, each a coarsening of the one before. Prediction blends
the chain bottom-up with modified Kneser-Ney discounting (counts only).
"""
import numpy as np, time, sys, os, json

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
P1 = np.uint64(11400714819323198485)
P2 = np.uint64(14029467366897019727)
P3 = np.uint64(1609587929392839161)


def load(name):
    toks = {}
    for split in ("train", "valid", "test"):
        with open(f"{DATA}/{name}.{split}.txt", encoding="utf8") as f:
            out = []
            for line in f:
                ws = line.split()
                if not ws:
                    continue
                out.extend(ws)
                out.append("<eos>")
            toks[split] = out
    vocab = {}
    for w in toks["train"]:
        if w not in vocab:
            vocab[w] = len(vocab)
    unk = vocab.get("<unk>", 0)
    ids = {s: np.array([vocab.get(w, unk) for w in t], dtype=np.int64) for s, t in toks.items()}
    words = [None] * len(vocab)
    for w, i in vocab.items():
        words[i] = w
    return ids, words


# ---------------------------------------------------------------- abstraction: classes
def _f(x):
    return np.where(x > 0, x * np.log(np.maximum(x, 1e-300)), 0.0)


def bigram_lists(x, V):
    key = x[:-1] * V + x[1:]
    u, c = np.unique(key, return_counts=True)
    a, b = u // V, u % V
    return a, b, c.astype(np.float64)


def cluster(a, b, c, V, K, sweeps=6, seed=0, log=None):
    """Exchange clustering of V items into K classes. Two items share a class when they
    have the same neighbours on both sides. Each move is kept only if it shortens the
    description of the bigram stream (two-sided predictive class model)."""
    cnt = np.bincount(a, weights=c, minlength=V) + np.bincount(b, weights=c, minlength=V)
    order = np.argsort(-cnt)
    cls = np.empty(V, dtype=np.int64)
    cls[order] = np.arange(V) % K
    # ML[v, k] = count of item v followed by class k ; MR[u, k] = count of class k followed by item u
    ML = np.zeros((V, K)); MR = np.zeros((V, K))
    np.add.at(ML, (a, cls[b]), c)
    np.add.at(MR, (b, cls[a]), c)
    cw_succ = np.bincount(b, weights=c, minlength=V)   # count of w as the second item
    cw_pred = np.bincount(a, weights=c, minlength=V)
    NcS = np.bincount(cls, weights=cw_succ, minlength=K)
    NcP = np.bincount(cls, weights=cw_pred, minlength=K)
    # predecessor lists (for w as successor) and successor lists
    o1 = np.argsort(b, kind="stable"); pb, pa, pc = b[o1], a[o1], c[o1]
    s1 = np.searchsorted(pb, np.arange(V + 1))
    o2 = np.argsort(a, kind="stable"); qa, qb, qc = a[o2], b[o2], c[o2]
    s2 = np.searchsorted(qa, np.arange(V + 1))
    rng = np.random.default_rng(seed)
    for sw in range(sweeps):
        moved = 0
        for w in order:
            P = pa[s1[w]:s1[w + 1]]; nl = pc[s1[w]:s1[w + 1]]
            S = qb[s2[w]:s2[w + 1]]; nr = qc[s2[w]:s2[w + 1]]
            if len(P) == 0 and len(S) == 0:
                continue
            k0 = cls[w]
            ML[P, k0] -= nl; MR[S, k0] -= nr
            NcS[k0] -= cw_succ[w]; NcP[k0] -= cw_pred[w]
            g = np.zeros(K)
            if len(P):
                m = ML[P]; g += (_f(m + nl[:, None]) - _f(m)).sum(0)
            if len(S):
                m = MR[S]; g += (_f(m + nr[:, None]) - _f(m)).sum(0)
            g -= _f(NcS + cw_succ[w]) - _f(NcS)
            g -= _f(NcP + cw_pred[w]) - _f(NcP)
            k1 = int(np.argmax(g))
            if g[k1] <= g[k0] + 1e-9:
                k1 = k0
            if k1 != k0:
                moved += 1
            cls[w] = k1
            ML[P, k1] += nl; MR[S, k1] += nr
            NcS[k1] += cw_succ[w]; NcP[k1] += cw_pred[w]
        if log:
            log(f"   sweep {sw}: moved {moved}")
        if moved < V * 0.002:
            break
    return cls


def build_layers(x, V, sizes, sweeps=6, log=print):
    """Layers of classes, each layer a clustering of the layer below (nested)."""
    maps = []            # word -> class id at each layer
    cur = x; curV = V; word_to_cur = np.arange(V)
    for K in sizes:
        t = time.time()
        a, b, c = bigram_lists(cur, curV)
        cl = cluster(a, b, c, curV, K, sweeps=sweeps, log=log)
        word_to_cur = cl[word_to_cur]
        maps.append(word_to_cur.copy())
        cur = cl[cur]; curV = K
        log(f"  layer of {K} classes built in {time.time()-t:.0f}s")
    return maps


# ---------------------------------------------------------------- memory + chain
DROP = -1


class Chain:
    def __init__(self, V, maps, specs):
        """specs: list of tuples, one abstraction level per context position (nearest first)."""
        self.V = V; self.maps = maps; self.specs = specs
        self.n = len(specs[0])

    def _ctx(self, x, pos, spec, sid):
        h = np.full(len(pos), np.uint64(sid * 7919 + 1), dtype=np.uint64)
        for j, a in enumerate(spec, start=1):
            if a == DROP:
                continue
            t = x[pos - j]
            if a > 0:
                t = self.maps[a - 1][t]
            h = (h ^ (t.astype(np.uint64) + np.uint64(j * 1000003 + a * 97))) * P1
            h ^= h >> np.uint64(29)
        return h * P2

    def fit(self, x):
        n = self.n
        x = np.concatenate([np.full(n, x[-1]), x])        # pad with <eos>
        self.levels = []
        items = np.arange(n, len(x))
        for sid, spec in enumerate(self.specs):
            ch = self._ctx(x, items, spec, sid)
            key = ch * P3 + x[items].astype(np.uint64)
            uk, first, cnt = np.unique(key, return_index=True, return_counts=True)
            cu = ch[first]
            uc, inv = np.unique(cu, return_inverse=True)
            tot = np.bincount(inv, weights=cnt, minlength=len(uc))
            nn = [np.bincount(inv, weights=(cnt == 1), minlength=len(uc)),
                  np.bincount(inv, weights=(cnt == 2), minlength=len(uc)),
                  np.bincount(inv, weights=(cnt >= 3), minlength=len(uc))]
            c1, c2, c3, c4 = [(cnt == i).sum() for i in (1, 2, 3, 4)]
            if c1 and c2 and c3 and c4:
                Y = c1 / (c1 + 2 * c2)
                D = [max(0.05, 1 - 2 * Y * c2 / c1), max(0.05, 2 - 3 * Y * c3 / c2), max(0.05, 3 - 4 * Y * c4 / c3)]
            else:
                D = [0.5, 0.75, 0.9]
            self.levels.append(dict(uk=uk, cnt=cnt.astype(np.float64), uc=uc, tot=tot, nn=nn, D=D))
            items = items[first]
        return self

    def prob(self, x, hist=None):
        """Probability of each token of x given the tokens before it (hist = text before x)."""
        n = self.n
        pre = np.full(n, x[-1]) if hist is None else hist[-n:]
        xx = np.concatenate([pre, x]); pos = np.arange(n, len(xx))
        p = np.full(len(x), 1.0 / self.V)
        for sid in range(len(self.specs) - 1, -1, -1):
            L = self.levels[sid]
            ch = self._ctx(xx, pos, self.specs[sid], sid)
            key = ch * P3 + xx[pos].astype(np.uint64)
            i = np.searchsorted(L["uk"], key); i[i >= len(L["uk"])] = 0
            c = np.where(L["uk"][i] == key, L["cnt"][i], 0.0)
            j = np.searchsorted(L["uc"], ch); j[j >= len(L["uc"])] = 0
            ok = L["uc"][j] == ch
            tot = np.where(ok, L["tot"][j], 0.0)
            D1, D2, D3 = L["D"]
            d = np.where(c >= 3, D3, np.where(c >= 2, D2, np.where(c >= 1, D1, 0.0)))
            gam = D1 * L["nn"][0][j] + D2 * L["nn"][1][j] + D3 * L["nn"][2][j]
            with np.errstate(divide="ignore", invalid="ignore"):
                q = (np.maximum(c - d, 0) + gam * p) / tot
            p = np.where(tot > 0, q, p)
        return p


def ppl(p):
    return float(np.exp(-np.log(p).mean()))


def memory_chain(n):
    """Pure memory: words only, drop the farthest word each step (Kneser-Ney n-gram)."""
    specs = []
    for k in range(n, -1, -1):
        specs.append(tuple([0] * k + [DROP] * (n - k)))
    return specs


def strata_chain(n, L, far=0):
    """Most specific first: all words. Then the farthest kept position climbs one layer at
    a time until it is dropped, then the next one. `far` extra positions beyond n start at
    the coarsest layer (long patterns over kinds)."""
    cur = [0] * n + [L] * far
    specs = [tuple(cur)]
    for j in range(len(cur) - 1, -1, -1):
        while cur[j] != DROP:
            cur[j] = cur[j] + 1 if cur[j] < L else DROP
            specs.append(tuple(cur))
    return specs


def stagger_chain(n, L, far=0):
    """Abstraction rises gradually with distance: no position is more abstract than a
    farther one, and the farthest climbs first, round robin."""
    cur = [0] * n + [L] * far
    lev = lambda a: L + 1 if a == DROP else a
    specs = [tuple(cur)]
    while any(a != DROP for a in cur):
        # raise the position with the lowest level; ties -> farthest
        j = max(range(len(cur)), key=lambda i: (-(lev(cur[i]) - 0.0001 * 0), i) if cur[i] != DROP else (-99, -1))
        # keep monotone: only raise j if every farther position is strictly more abstract or dropped
        cand = [i for i in range(len(cur)) if cur[i] != DROP and all(lev(cur[t]) > lev(cur[i]) for t in range(i + 1, len(cur)))]
        j = max(cand)
        cur[j] = cur[j] + 1 if cur[j] < L else DROP
        specs.append(tuple(cur))
    return specs
