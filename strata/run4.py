import sys, time, pickle, os, numpy as np
from strata import *
name = sys.argv[1]
ids, words = load(name); V = len(words)
tr, va, te = ids["train"], ids["valid"], ids["test"]
def layers(sizes, seed):
    cf = f"data/{name}.layers.{'-'.join(map(str,sizes))}.s{seed}.pkl"
    if os.path.exists(cf): return pickle.load(open(cf, "rb"))
    maps = []; cur = tr; curV = V; w2c = np.arange(V)
    for K in sizes:
        a, b, c = bigram_lists(cur, curV)
        if seed:  # a different view: start from a random assignment
            rng = np.random.default_rng(seed)
        cl = cluster(a, b, c, curV, K, sweeps=6, seed=seed)
        w2c = cl[w2c]; maps.append(w2c.copy()); cur = cl[cur]; curV = K
    pickle.dump(maps, open(cf, "wb")); return maps
def drop_chain(start):
    cur = list(start); specs = [tuple(cur)]
    for j in range(len(cur) - 1, -1, -1):
        if cur[j] != DROP:
            cur[j] = DROP; specs.append(tuple(cur))
    return specs
def cache(x, hist, win, maps=None):
    """Memory of the current document. With maps: remembers kinds, then picks the word inside the kind."""
    from collections import Counter, deque
    xx = np.concatenate([hist[-win:], x]); off = len(xx) - len(x)
    key = xx if maps is None else maps[xx]
    out = np.zeros(len(x)); c = Counter(key[:off].tolist()); q = deque(key[:off].tolist())
    for i, k in enumerate(key[off:].tolist()):
        out[i] = c[k] / max(len(q), 1); q.append(k); c[k] += 1
        if len(q) > win: c[q.popleft()] -= 1
    if maps is not None:
        cnt = np.bincount(tr, minlength=V).astype(float); ctot = np.bincount(maps, weights=cnt)
        out = out * (cnt[x] + 0.01) / (ctot[maps[x]] + 0.01 * V)
    return out
PV, PT = {}, {}
t0 = time.time()
m = Chain(V, [], drop_chain([0]*4)).fit(tr)
PV["memory"] = m.prob(va, tr); bv = m.bucket.copy(); PT["memory"] = m.prob(te, va); bt = m.bucket.copy()
cfgs = [([1000,150,20], 0)] + [(list(map(int, c.split(","))), i + 1) for i, c in enumerate(sys.argv[2:])]
for sizes, seed in cfgs:
    t = time.time(); maps = layers(sizes, seed); tag = f"v{seed}"
    views = {"L1": [1]*5, "L2": [2]*6, "L3": [3]*8, "w+L1": [0,1,1,1,1], "ww+L2": [0,0,2,2,2,2], "L1+w": [1,0,0,0]}
    for k, st in views.items():
        mm = Chain(V, maps, drop_chain(st)).fit(tr)
        PV[tag+k] = mm.prob(va, tr); PT[tag+k] = mm.prob(te, va)
    if seed == 0:
        for lay in (1, 2):
            PV[f"kindcache{lay+1}"] = cache(va, tr, 2000, maps[lay]); PT[f"kindcache{lay+1}"] = cache(te, va, 2000, maps[lay])
    print(f"view {tag} {sizes} done ({time.time()-t:.0f}s)", flush=True)
for win in (200, 2000):
    PV[f"cache{win}"] = cache(va, tr, win); PT[f"cache{win}"] = cache(te, va, win)
print(f"all built in {time.time()-t0:.0f}s")
def em(keys, bucketed, iters=60):
    A = np.stack([PV[k] for k in keys], 1); B = np.stack([PT[k] for k in keys], 1)
    if not bucketed:
        w = np.full(len(keys), 1/len(keys))
        for _ in range(iters):
            r = A * w; r /= r.sum(1, keepdims=True); w = r.mean(0)
        return ppl(A @ w), ppl(B @ w)
    nb = max(bv.max(), bt.max()) + 1; W = np.full((nb, len(keys)), 1/len(keys))
    for _ in range(iters):
        r = A * W[bv]; r /= r.sum(1, keepdims=True)
        for j in range(len(keys)):
            W[:, j] = (np.bincount(bv, weights=r[:, j], minlength=nb) + 1.0 / len(keys)) / (np.bincount(bv, minlength=nb) + 1.0)
    return ppl((A * W[bv]).sum(1)), ppl((B * W[bt]).sum(1))
def show(label, keys):
    a = em(keys, False); b = em(keys, True)
    print(f"{label:44s} flat: valid {a[0]:6.1f} test {a[1]:6.1f} | by context: valid {b[0]:6.1f} test {b[1]:6.1f}", flush=True)
pat0 = [k for k in PV if k.startswith("v0")]; allpat = [k for k in PV if k[0] == "v"]
show("memory + patterns (1 view)", ["memory"] + pat0)
show("memory + patterns (all views)", ["memory"] + allpat)
show("  + word cache", ["memory"] + allpat + ["cache200", "cache2000"])
show("  + word cache + kind cache", ["memory"] + allpat + ["cache200", "cache2000", "kindcache2", "kindcache3"])
pickle.dump((PV, PT, bv, bt), open(f"data/{name}.probs4.pkl", "wb"))
