# Soft memory: every past moment votes for what came next, weighted by how alike its context is.
# Meaning vectors are made from counts (PPMI + SVD), no gradients.
import sys, time, pickle, numpy as np
from strata import *
name = sys.argv[1]; d = 64
ids, words = load(name); V = len(words)
tr, va, te = ids["train"], ids["valid"], ids["test"]
t0 = time.time()
def ppmi(a, b):
    M = np.zeros((V, V), dtype=np.float32); np.add.at(M, (a, b), 1.0)
    r = M.sum(1, keepdims=True); c = M.sum(0, keepdims=True); tot = M.sum()
    with np.errstate(divide="ignore"):
        P = np.log(np.maximum(M * tot, 1e-30) / np.maximum(r * c, 1e-30))
    return np.where(M > 0, np.maximum(P, 0), 0).astype(np.float32)
M = np.concatenate([ppmi(tr[1:], tr[:-1]), ppmi(tr[:-1], tr[1:]), ppmi(tr[2:], tr[:-2]), ppmi(tr[:-2], tr[2:])], 1)
rng = np.random.default_rng(0)
Y = M @ rng.standard_normal((M.shape[1], d + 20), dtype=np.float32)
for _ in range(2): Y = M @ (M.T @ Y); Y, _ = np.linalg.qr(Y)
Q, _ = np.linalg.qr(Y); U, S, _ = np.linalg.svd(Q.T @ M, full_matrices=False)
E = (Q @ U[:, :d]) * np.sqrt(S[:d]); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
print(f"meaning vectors built in {time.time()-t0:.0f}s", flush=True)
for w in ("president", "monday", "rose", "million"):
    i = words.index(w); nn = np.argsort(-(E @ E[i]))[1:8]; print("  ", w, "~", " ".join(words[j] for j in nn))
wts = np.array([1.0, 0.8, 0.6, 0.5], dtype=np.float32); n = len(wts)
def ctxvec(x, hist):
    xx = np.concatenate([hist[-n:], x]); pos = np.arange(n, len(xx))
    C = np.concatenate([E[xx[pos - j]] * wts[j - 1] for j in range(1, n + 1)], 1)
    return C / np.linalg.norm(C, axis=1, keepdims=True)
Ctr = ctxvec(tr[n:], tr[:n]); ytr = tr[n:]
def soft(x, hist, betas, chunk=200):
    Cq = ctxvec(x, hist); out = {b: np.zeros(len(x)) for b in betas}
    for s in range(0, len(x), chunk):
        S_ = Cq[s:s + chunk] @ Ctr.T
        hit = ytr[None, :] == x[s:s + chunk, None]
        for b in betas:
            K = np.exp(b * (S_ - 1.0)); out[b][s:s + chunk] = (K * hit).sum(1) / K.sum(1)
    return out
sub = slice(0, 12000)
t = time.time(); sv = soft(va[sub], tr, [15, 30, 60]); print(f"valid sample {time.time()-t:.0f}s", {b: round(ppl(np.maximum(p, 1e-9)), 1) for b, p in sv.items()}, flush=True)
PV, PT, bv, bt = pickle.load(open(f"data/{name}.probs4.pkl", "rb"))
keys = list(PV)
def em(extra_v=None, iters=80):
    A = np.stack([PV[k][sub] for k in keys] + ([extra_v] if extra_v is not None else []), 1); w = np.full(A.shape[1], 1 / A.shape[1])
    for _ in range(iters):
        r = A * w; r /= r.sum(1, keepdims=True); w = r.mean(0)
    return w, ppl(A @ w)
w0, v0 = em(); print(f"blend without soft memory: valid sample {v0:.1f}")
best = None
for b, p in sv.items():
    w, v = em(p); print(f"blend with soft memory beta={b}: valid sample {v:.1f}  weight {w[-1]:.2f}", flush=True)
    if best is None or v < best[1]: best = (b, v, w)
b, _, w = best
t = time.time(); st = soft(te, va, [b])[b]; print(f"test pass {time.time()-t:.0f}s  soft memory alone test {ppl(np.maximum(st, 1e-9)):.1f}")
B = np.stack([PT[k] for k in keys], 1)
print(f"TEST without soft memory {ppl(B @ w0):.1f}   with soft memory {ppl(np.concatenate([B, st[:, None]], 1) @ w):.1f}")
