"""Loop-135: Bee-135 with a loop in the middle that thinks over the words' vectors. NEXT-TOKEN.

João (Nov 22 2026): think over meanings, not words; the loop built in from the start (no seeds, no hoping one sprouts).

    words ─► reader   layers 0-15 of Bee-135, each once, the whole text in sight
               │      every word is now a vector (what it means here)
               ▼
             loop     layers 16-19, the SAME four layers again and again: K rounds (2 to 6 in training, 4 when reading)
               │      every round, each word's thought looks at the thoughts of all the earlier words (causal attention)
               │      and re-reads the reader's vector of its own word (input injection), so it cannot drift off the text
               ▼
             mouth    layers 20-23, once ─► the next word

One thought per word: the mouth uses a word's thought for that one next word, and nothing is thrown away, because the
next word's thought looks back at every earlier thought, round by round.

Exactly Bee-135's 24 layers and weights (132M): with K = 1 it IS Bee-135, layer for layer (the U-net skips, the
first-input mixing, the back-out all as Bee's). Rounds 2.. add only the loop's four layers again, plus one vector of
576 numbers (how much of the reader's vector each round takes back in; it starts at 0).

Why a random number of rounds in training: a loop trained at exactly K rounds learns "stop at K"; more rounds when
reading then make it worse, not better. Trained at 2..6 it learns a thought that keeps settling, so reading with 8 or
12 rounds can help (Geiping et al. 2025, "Scaling up test-time compute with latent reasoning"). The held-out loss by
rounds (1, 2, 4, 6, 8, 12) is logged at every check (the `[loop]` line).

Memory: only the last `loop_grad` rounds carry gradient (the stream is cut before them; the reader still learns
through the re-read vector in every round). So 5 or 6 rounds cost the memory of 4.

    python3 -m oazy.train --model gnosis2 --set n_layer=24 ... loop_layers=4 loop_mouth=4 loop_steps=4 loop_min=2 loop_max=6
    (loopab/ab.py has the exact words; runpod/loopab.sh runs the A/B)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from bee5.model import norm, rope_table, ROPE_TABLE


def check(cfg, n_skip: int, backout_at: int):
    n, L, M = cfg.n_layer, cfg.loop_layers, cfg.loop_mouth
    a = n - M - L
    if L < 1 or M < 0 or a < 1:
        raise ValueError(f"loop: {n} layers cannot hold a reader, {L} loop layers and a mouth of {M}")
    if n_skip > a or backout_at >= a:
        raise ValueError("loop: the U-net skips and the back-out must start in the reader")
    if cfg.mod_set():
        raise ValueError("loop: no routed (MoD) layers")
    if cfg.loop_min and not 1 <= cfg.loop_min <= cfg.loop_max:
        raise ValueError("loop: need 1 <= loop_min <= loop_max")
    return a


def steps(model) -> int:
    """How many rounds this pass thinks: forced (loop_report), random in training, else the reading number."""
    cfg = model.cfg
    if getattr(model, "_k_force", None):
        return model._k_force
    if model.training and cfg.loop_min:
        return int(torch.randint(cfg.loop_min, cfg.loop_max + 1, (1,)).item())
    return cfg.loop_eval or cfg.loop_steps


def eager():
    """Outside training the layers run uncompiled, as Bee's do: the tests' many lengths would otherwise fill each
    layer's compile cache (64 entries), and from then on torch runs new training shapes uncompiled, ~3x slower."""
    import contextlib
    sc = getattr(torch.compiler, "set_stance", None)
    return sc("force_eager") if sc is not None else contextlib.nullcontext()


def run(model, x0, ves, layer_fn):
    """Bee5._run with the loop: reader once, loop K rounds, mouth once (uncompiled when not training)."""
    if model.training or getattr(model, "_eager", False):
        return _run(model, x0, ves, layer_fn)
    model._eager = True
    try:
        with eager():
            return _run(model, x0, ves, layer_fn)
    finally:
        model._eager = False


def _run(model, x0, ves, layer_fn):
    """layer_fn(i, x, ve) -> x; model._pass is set to a number unique to each pass before each call (the reading-on
    cache keys its layers by it)."""
    cfg = model.cfg
    n, L = cfg.n_layer, cfg.loop_layers
    n_skip = len(model.skip_gate)
    a = check(cfg, n_skip, model.backout_at)
    K = steps(model)
    vi = cfg.ve_index()
    x, skips, box = x0, [], {}
    p = 0

    def bee(i, x):                                   # Bee's own treatment of layer i
        nonlocal p
        if n_skip and i >= n - n_skip:
            x = x + model._skip(n - 1 - i, x0) * skips[n - 1 - i]
        if cfg.x0:
            x = model.lam[i] * x + model.mu[i] * x0
        model._pass = p
        p += 1
        x = layer_fn(i, x, ves[vi[i]] if vi[i] is not None else None)
        if i < n_skip:
            skips.append(x)
        if i == model.backout_at:
            box["xb"] = x
        return x

    for i in range(a):                               # the reader
        x = bee(i, x)
    r = norm(x)                                      # each word's vector after the reader: re-read every round
    cut = K - cfg.loop_grad if model.training and cfg.loop_grad else 0
    for s in range(K):                               # the loop
        if s:
            if s == cut:
                x = x.detach()                       # earlier rounds: forward only (the memory of loop_grad rounds)
            x = x + model.loop_in.to(x.dtype) * r.to(x.dtype)
        for j in range(L):
            i = a + j
            if s == 0:
                x = bee(i, x)                        # round 1 is Bee-135's layers 16-19 exactly
            else:
                model._pass = p
                p += 1
                x = layer_fn(i, x, ves[vi[i]] if vi[i] is not None else None)
    for i in range(a + L, n):                        # the mouth
        x = bee(i, x)
    if cfg.backout:
        x = x - model.backout_lam.to(x.dtype) * box["xb"]
    model._k_last = K
    return model._norm(x)


def read(model, idx, states=None, looks=False):
    """Bee5._read (reading as a stream: chat, samples) with the cache keyed by pass, not by layer."""
    st = states or {"pos": 0, "prev": None, "layers": {}}
    P, T = st["pos"], idx.shape[1]
    cfg = model.cfg
    if P + T > ROPE_TABLE:
        cos, sin = rope_table(P + T, cfg.attn_head_dim, cfg.rope_wave, cfg.rope_frac, idx.device)
        cos, sin = cos[P:], sin[P:]
    else:
        cos, sin = model.rope_cos[P:P + T], model.rope_sin[P:P + T]
    e, raw_last = model._embed(idx, st["prev"])
    x0 = norm(e)
    ves = [t(idx) for t in model.ve]
    new, infos = dict(st["layers"]), []

    def layer(i, x, ve):
        k = model._pass
        x, new[k], info = model.blocks[i].step(x, ve, cos, sin, st["layers"].get(k), looks)
        infos.append({"kind": "A", "looks": info[0] if info is not None else None})
        return x
    was = getattr(model, "_k_force", None)
    k = model._k_force = st.get("k") or steps(model)          # one number of rounds for the whole stream (the cache)
    try:
        h = run(model, x0, ves, layer)
    finally:
        model._k_force = was
    return model.scores(h), {"pos": P + T, "prev": raw_last, "layers": new, "k": k}, infos


def cost(cfg) -> float:
    """Training compute per word against Bee-135 (24 layers, each forward + backward = 3 units). A round with gradient
    costs 3 units a layer, a round before the cut 1 (forward only). Embedding and head left out (the same in both)."""
    n, L = cfg.n_layer, cfg.loop_layers
    ks = list(range(cfg.loop_min, cfg.loop_max + 1)) if cfg.loop_min else [cfg.loop_steps]
    g = cfg.loop_grad or 10 ** 9
    loop = sum(3 * min(k, g) + max(0, k - g) for k in ks) / len(ks)        # the loop's units per layer, on average
    return (3 * (n - L) + L * loop) / (3 * n)


def compile_blocks(model):
    """Each layer compiled on its own (the number of rounds changes from step to step: one whole-body graph would be
    compiled again for every number). The loss step's kernels tuned with the layers'."""
    import torch._dynamo
    from bee6 import lean as LEAN
    from bee5.loss import set_compile
    dc = torch._dynamo.config
    dc.cache_size_limit = max(getattr(dc, "cache_size_limit", 8), 64)
    if hasattr(dc, "accumulated_cache_size_limit"):
        dc.accumulated_cache_size_limit = max(dc.accumulated_cache_size_limit, 512)
    mode = model.cfg.compile_mode
    kw = LEAN.compile_kw(mode)
    set_compile(kw if mode in LEAN.TUNE else {})
    for b in model.blocks:
        b.compile(dynamic=False, **kw)
    model._compiled = None


@torch.no_grad()
def held_out(model, windows, device, batch, k):
    x_all = torch.as_tensor(windows).long().to(device)
    tot, n = 0.0, 0
    model._k_force = k
    try:
        for i in range(0, len(x_all), batch):
            xb = x_all[i:i + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = model.scores(model._trunk_aux(xb[:, :-1])[0]).float()
            tot += float(F.cross_entropy(lg.transpose(1, 2), xb[:, 1:], reduction="none").mean(1).sum())
            n += len(xb)
    finally:
        model._k_force = None
    return tot / max(1, n)


ROUNDS = (1, 2, 4, 6, 8, 12)


@torch.no_grad()
def report(model, windows, device, batch=16):
    """Does thinking longer help? Held-out loss with 1, 2, 4, 6, 8 and 12 rounds (1 = Bee-135's own depth)."""
    was = model.training
    model.eval()
    try:
        out = {k: held_out(model, windows, device, batch, k) for k in ROUNDS}
    finally:
        model.train(was)
    K = model.cfg.loop_eval or model.cfg.loop_steps
    return {"pass_loss": out, "adaptive_loss": out.get(K, min(out.values())), "adaptive_passes": float(K), "move_last": 0.0}
