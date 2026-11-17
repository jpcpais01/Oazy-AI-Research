# Recursion: words merge into phrase units; units get kinds of their own; patterns over units.
# The context of each word is read as the units that END just before it (uses the past only).
import sys, time, pickle, numpy as np
from strata import *
name = sys.argv[1]; cmin = int(sys.argv[2]); coh = float(sys.argv[3])
ids, words = load(name); V = len(words)
tr, va, te = ids["train"], ids["valid"], ids["test"]
t0 = time.time()
def nhash(x, n):               # hash of the n words ending at each position (valid for pos >= n-1)
    h = np.zeros(len(x), dtype=np.uint64)
    for j in range(n):
        h[n - 1:] = (h[n - 1:] ^ (x[n - 1 - j:len(x) - j].astype(np.uint64) + np.uint64(j * 131 + 7))) * P1
        h ^= h >> np.uint64(31)
    return h
N = len(tr); uni = np.bincount(tr, minlength=V).astype(float)
phr = {}; prevcount = {1: None}
keep_hash = []; reps = []
cnt_prev = None
for n in (2, 3, 4):
    h = nhash(tr, n)[n - 1:]
    u, first, c = np.unique(h, return_index=True, return_counts=True)
    endpos = first + n - 1
    if n == 2: cpre = uni[tr[endpos - 1]]
    else:
        hp = nhash(tr, n - 1); up, cp = np.unique(hp[n - 2:], return_counts=True)
        i = np.searchsorted(up, hp[endpos - 1]); cpre = cp[i]
    score = c * N / (cpre * uni[tr[endpos]])
    ok = (c >= cmin) & (score >= coh)
    keep_hash.append(u[ok]); reps.append((endpos[ok], n))
    print(f"  {n}-word units kept: {ok.sum()}", flush=True)
base = [V, V + len(keep_hash[0]), V + len(keep_hash[0]) + len(keep_hash[1])]
NU = base[2] + len(keep_hash[2])
names = list(words) + [None] * (NU - V)
for k, (ep, n) in enumerate(reps):
    for idx, e in enumerate(ep[:]):
        names[base[k] + idx] = "_".join(words[i] for i in tr[e - n + 1:e + 1])
def units(x):                  # for every position: id and length of the longest unit ending there
    uid = x.copy(); L = np.ones(len(x), dtype=np.int64)
    for k, n in enumerate((2, 3, 4)):
        h = nhash(x, n); i = np.searchsorted(keep_hash[k], h); i[i >= len(keep_hash[k])] = 0
        hit = (keep_hash[k][i] == h) if len(keep_hash[k]) else np.zeros(len(x), bool)
        hit[:n - 1] = False
        uid = np.where(hit, base[k] + i, uid); L = np.where(hit, n, L)
    return uid, L
def feats(x, hist, k=6):
    pad = 40; xx = np.concatenate([np.full(pad, hist[-1]), hist[-200:], x]); uid, L = units(xx)
    t = np.arange(len(xx) - len(x), len(xx)); e = t - 1; F = np.zeros((len(x), k), dtype=np.int64)
    for j in range(k):
        e = np.maximum(e, 0); F[:, j] = uid[e]; e = e - L[e]
    return F
# unit stream for clustering: walk the training text backwards unit by unit
uid_tr, L_tr = units(tr); seq = []; e = N - 1
while e >= 0: seq.append(uid_tr[e]); e -= L_tr[e]
seq = np.array(seq[::-1]); print(f"  unit stream: {len(seq)} units for {N} words; distinct phrase units {NU - V}", flush=True)
a, b, c = bigram_lists(seq, NU)
maps = []; cur = seq; curV = NU; u2c = np.arange(NU)
for K in (1000, 150, 20):
    aa, bb, cc = bigram_lists(cur, curV); cl = cluster(aa, bb, cc, curV, K, sweeps=5)
    u2c = cl[u2c]; maps.append(u2c.copy()); cur = cl[cur]; curV = K
print(f"  unit kinds built ({time.time()-t0:.0f}s)")
big = np.argsort(-np.bincount(maps[0][seq], minlength=1000))
for cidx in big[5:60:6]:
    mem = [names[i] for i in np.where(maps[0] == cidx)[0] if i >= V][:8]
    if mem: print("   kind:", "  ".join(mem))
class FChain:
    def __init__(self, V, k): self.V = V; self.k = k
    def _ctx(self, F, m, sid):
        h = np.full(len(F), np.uint64(sid * 7919 + 1), dtype=np.uint64)
        for j in range(m):
            h = (h ^ (F[:, j].astype(np.uint64) + np.uint64(j * 1000003 + 5))) * P1; h ^= h >> np.uint64(29)
        return h * P2
    def fit(self, F, y):
        self.levels = []; items = np.arange(len(y))
        for sid, m in enumerate(range(self.k, -1, -1)):
            ch = self._ctx(F[items], m, sid); key = ch * P3 + y[items].astype(np.uint64)
            uk, first, cnt = np.unique(key, return_index=True, return_counts=True)
            uc, inv = np.unique(ch[first], return_inverse=True)
            tot = np.bincount(inv, weights=cnt, minlength=len(uc))
            nn = [np.bincount(inv, weights=(cnt == 1), minlength=len(uc)), np.bincount(inv, weights=(cnt == 2), minlength=len(uc)), np.bincount(inv, weights=(cnt >= 3), minlength=len(uc))]
            c1, c2, c3, c4 = [(cnt == i).sum() for i in (1, 2, 3, 4)]
            if c1 and c2 and c3 and c4:
                Y = c1 / (c1 + 2 * c2); D = [max(.05, 1 - 2 * Y * c2 / c1), max(.05, 2 - 3 * Y * c3 / c2), max(.05, 3 - 4 * Y * c4 / c3)]
            else: D = [.5, .75, .9]
            self.levels.append(dict(uk=uk, cnt=cnt.astype(float), uc=uc, tot=tot, nn=nn, D=D)); items = items[first]
        return self
    def prob(self, F, y):
        p = np.full(len(y), 1.0 / self.V)
        for sid in range(self.k, -1, -1):
            L = self.levels[sid]; ch = self._ctx(F, self.k - sid, sid); key = ch * P3 + y.astype(np.uint64)
            i = np.searchsorted(L["uk"], key); i[i >= len(L["uk"])] = 0; c = np.where(L["uk"][i] == key, L["cnt"][i], 0.0)
            j = np.searchsorted(L["uc"], ch); j[j >= len(L["uc"])] = 0; tot = np.where(L["uc"][j] == ch, L["tot"][j], 0.0)
            D1, D2, D3 = L["D"]; d = np.where(c >= 3, D3, np.where(c >= 2, D2, np.where(c >= 1, D1, 0.0)))
            gam = D1 * L["nn"][0][j] + D2 * L["nn"][1][j] + D3 * L["nn"][2][j]
            with np.errstate(divide="ignore", invalid="ignore"): q = (np.maximum(c - d, 0) + gam * p) / tot
            p = np.where(tot > 0, q, p)
        return p
Ftr = feats(tr[6:], tr[:6]); ytr = tr[6:]; Fva = feats(va, tr); Fte = feats(te, va)
m1, m2, m3 = maps
views = {
 "U": lambda F: F[:, :4],
 "UL1": lambda F: m1[F[:, :5]],
 "UL2": lambda F: m2[F[:, :6]],
 "UL3": lambda F: m3[F[:, :6]],
 "U+UL1": lambda F: np.stack([F[:, 0], m1[F[:, 1]], m1[F[:, 2]], m1[F[:, 3]], m1[F[:, 4]]], 1),
 "UU+UL2": lambda F: np.stack([F[:, 0], F[:, 1], m2[F[:, 2]], m2[F[:, 3]], m2[F[:, 4]], m2[F[:, 5]]], 1),
}
PV, PT, bv, bt = pickle.load(open(f"data/{name}.probs4.pkl", "rb")); old = list(PV)
for k, fn in views.items():
    A = fn(Ftr); mm = FChain(V, A.shape[1]).fit(A, ytr); PV["u:" + k] = mm.prob(fn(Fva), va); PT["u:" + k] = mm.prob(fn(Fte), te)
    print(f"  {k:8s} alone: valid {ppl(PV['u:'+k]):.1f} test {ppl(PT['u:'+k]):.1f}", flush=True)
def em(keys, iters=80):
    A = np.stack([PV[k] for k in keys], 1); B = np.stack([PT[k] for k in keys], 1); w = np.full(len(keys), 1 / len(keys))
    for _ in range(iters):
        r = A * w; r /= r.sum(1, keepdims=True); w = r.mean(0)
    return ppl(A @ w), ppl(B @ w), w
v, t, _ = em(old); print(f"before (words only):      valid {v:.1f} test {t:.1f}")
v, t, w = em(list(PV)); print(f"with patterns over units: valid {v:.1f} test {t:.1f}   weight on unit views {w[len(old):].sum():.2f}  ({time.time()-t0:.0f}s)")
