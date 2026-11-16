import sys, time, pickle, os, numpy as np
from strata import *
name = sys.argv[1]; sizes = [int(s) for s in sys.argv[2].split(",")]
ids, words = load(name); V = len(words)
tr, va, te = ids["train"], ids["valid"], ids["test"]
maps = pickle.load(open(f"data/{name}.layers.{'-'.join(map(str,sizes))}.pkl", "rb"))
L = len(sizes)
def drop_chain(start):
    cur = list(start); specs = [tuple(cur)]
    for j in range(len(cur) - 1, -1, -1):
        if cur[j] != DROP:
            cur[j] = DROP; specs.append(tuple(cur))
    return specs
views = {
 "memory": drop_chain([0]*4),
 "L1": drop_chain([1]*5),
 "L2": drop_chain([2]*6),
 "L3": drop_chain([3]*8),
 "w+L1": drop_chain([0,1,1,1,1]),
 "ww+L2": drop_chain([0,0,2,2,2,2]),
 "skip1": drop_chain([DROP,0,0,0]),
 "skip1L1": drop_chain([DROP,1,1,1,1]),
 "L1+w": drop_chain([1,0,0,0]),
}
def cache(x, hist, win=500):
    xx = np.concatenate([hist[-win:], x]); out = np.zeros(len(x)); off = len(xx) - len(x)
    from collections import Counter, deque
    c = Counter(xx[:off].tolist()); q = deque(xx[:off].tolist())
    for i, w in enumerate(x.tolist()):
        out[i] = c[w] / max(len(q), 1)
        q.append(w); c[w] += 1
        if len(q) > win: c[q.popleft()] -= 1
    return out
PV, PT = {}, {}
for k, specs in views.items():
    t = time.time(); m = Chain(V, maps, specs).fit(tr)
    PV[k] = m.prob(va, tr); PT[k] = m.prob(te, va)
    print(f"{k:10s} valid {ppl(PV[k]):7.1f} test {ppl(PT[k]):7.1f} ({time.time()-t:.0f}s)", flush=True)
PV["cache"] = cache(va, tr); PT["cache"] = cache(te, va)
def em(keys, iters=50):
    A = np.stack([PV[k] for k in keys], 1); w = np.full(len(keys), 1/len(keys))
    for _ in range(iters):
        r = A * w; r /= r.sum(1, keepdims=True); w = r.mean(0)
    B = np.stack([PT[k] for k in keys], 1)
    return w, ppl(A @ w), ppl(B @ w)
def show(label, keys):
    w, v, t = em(keys)
    print(f"{label:28s} valid {v:6.1f} test {t:6.1f}   " + " ".join(f"{k}={x:.2f}" for k, x in zip(keys, w)), flush=True)
show("memory", ["memory"])
show("memory+cache", ["memory", "cache"])
show("memory+L1", ["memory", "L1"])
show("memory+L1+L2+L3", ["memory", "L1", "L2", "L3"])
show("all patterns", [k for k in views])
show("all patterns + cache", [k for k in views] + ["cache"])
pickle.dump((PV, PT), open(f"data/{name}.probs.pkl", "wb"))
