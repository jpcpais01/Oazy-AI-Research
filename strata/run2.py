import sys, time, pickle, os, numpy as np
from strata import *
name = sys.argv[1]; sizes = [int(s) for s in sys.argv[2].split(",")]
ids, words = load(name); V = len(words)
tr, va, te = ids["train"], ids["valid"], ids["test"]
cf = f"data/{name}.layers.{'-'.join(map(str,sizes))}.pkl"
if os.path.exists(cf): maps = pickle.load(open(cf, "rb"))
else:
    maps = build_layers(tr, V, sizes); pickle.dump(maps, open(cf, "wb"))
L = len(sizes)
for k in range(L):
    for c in (3, 17, 40):
        mem = np.where(maps[k] == c % sizes[k])[0][:12]
        print(f"layer {k+1} class {c % sizes[k]}:", " ".join(words[i] for i in mem))
def run(label, specs):
    t = time.time(); m = Chain(V, maps, specs).fit(tr)
    print(f"{label:34s} levels {len(specs):2d}  valid {ppl(m.prob(va, tr)):.1f}  test {ppl(m.prob(te, va)):.1f}  ({time.time()-t:.0f}s)", flush=True)
    return m
run("memory 5-gram", memory_chain(4))
for n in (4,):
    for far in (0, 2, 4):
        run(f"strata n={n} far={far}", strata_chain(n, L, far))
        run(f"stagger n={n} far={far}", stagger_chain(n, L, far))
