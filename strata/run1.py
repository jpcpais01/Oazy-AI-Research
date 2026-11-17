import sys, time, numpy as np
from strata import *
name = sys.argv[1]
ids, words = load(name); V = len(words)
tr, va, te = ids["train"], ids["valid"], ids["test"]
print(name, "V", V, "train", len(tr), flush=True)
for n in (3, 5):
    t = time.time(); m = Chain(V, [], memory_chain(n)).fit(tr)
    print(f"memory {n}-gram  valid {ppl(m.prob(va, tr)):.1f}  test {ppl(m.prob(te, va)):.1f}  ({time.time()-t:.0f}s)", flush=True)
