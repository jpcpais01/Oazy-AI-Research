"""Self-tests for lean (bee6/lean.py): the same numbers as the plain code, value by value and gradient by gradient.

The hand-written pieces against autograd on the plain code (norm, norm + Canon, the queries' and keys' norm + rotation,
the loss step); the whole model lean and plain on the same weights (Bee-135's layers small, and a Bee6 with routed
layers); the bf16 stream on a real bf16 pass; reading and writing with lean on; the startup check and its fall-backs;
and the compiler's plan: lean moves fewer bytes (the two habits bee6/lean.py leans on still hold).

    python3 -m bee6.lean_tests                 (CPU, ~2 min)
    python3 -m bee6.lean_tests --device cuda   (on the pod: also the compiled lean step on the GPU)
"""
from __future__ import annotations

import argparse
import copy

import torch
import torch.nn.functional as F

from oazy import term

TINY = dict(d_model=64, n_layer=6, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=128,
            ctx="128:0", vocab_size=512, key_offset="1,5", fp8=0)


def check(name, ok, detail=""):
    term.check(name, ok, detail)


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))


def wake(m, s=0.05):
    """Leave the zero start, so every path (outputs, Canon weights, value tables, gates, skips) matters."""
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * s)
    return m


def grads_of(m, x, y, amp=None):
    m.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp is not None):
        loss, _ = m.losses(x, y)
    loss.backward()
    return float(loss.detach()), {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.grad is not None}


def both(m, x, y, amp=None, s16=0):
    """(plain, lean): loss and every gradient, same weights."""
    out = []
    for lean in (0, 1):
        m.cfg.lean, m.cfg.stream16 = lean, s16 if lean else 0
        out.append(grads_of(m, x, y, amp))
    return out


def worst(g0, g1):
    return max(rel(g0[k], g1[k]) for k in g0)


def cos_mean(g0, g1):
    from bee6.fp8 import cosine
    big = [n for n in g0 if g0[n].numel() >= 1024 and g0[n].norm() > 0]
    return sum(cosine(g0[n], g1[n]) for n in big) / max(1, len(big))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("lean", dev)
    torch.manual_seed(0)
    from bee5 import loss as LS
    from bee5.model import rope_table
    from bee6 import lean as L
    from bee6 import traffic as TF
    from bee6.model import Bee6, Bee6Config, rotary
    from gnosis2.model import Canon, Gnosis2, Gnosis2Config

    # ---- the hand-written pieces against autograd on the plain code
    D, k = 16, 4
    x = torch.randn(3, 9, D, device=dev, requires_grad=True)
    g = torch.randn(3, 9, D, device=dev)
    ref, out = F.rms_norm(x, (D,)), L.norm(x)
    rp = L.norm(x, None, 4)
    gp, = torch.autograd.grad(rp, x, g)
    ga, = torch.autograd.grad(ref, x, g)
    gb, = torch.autograd.grad(out, x, g)
    check("norm: F.rms_norm's values and its gradient (also with the mean taken in pieces)",
          rel(ref, out) < 1e-6 and rel(ga, gb) < 1e-5 and rel(ref, rp) < 1e-6 and rel(ga, gp) < 1e-5, f"{rel(ref, out):.1e} / {rel(ga, gb):.1e}")

    canon = Canon(D, k).to(dev)
    with torch.no_grad():
        canon.w.normal_(0, 0.3)
    w = canon.w.detach().clone().requires_grad_()
    for T, what in ((9, "norm + Canon"), (2, "norm + Canon on a text shorter than Canon's reach")):
        xs = x[:, :T]
        ref, out = canon(F.rms_norm(xs, (D,))), L.norm_canon(xs, w)
        ga = torch.autograd.grad(ref, (x, canon.w), g[:, :T])
        gb = torch.autograd.grad(out, (x, w), g[:, :T])
        check(f"{what}: the plain layers' values, the stream's gradient and the Canon weights' gradient",
              rel(ref, out) < 1e-6 and rel(ga[0], gb[0]) < 1e-5 and rel(ga[1], gb[1]) < 1e-5,
              f"{rel(ref, out):.1e} / {rel(ga[0], gb[0]):.1e} / {rel(ga[1], gb[1]):.1e}")
    ref = canon(F.rms_norm(x, (D,))).to(torch.bfloat16)
    out = L.norm_canon(x, w, torch.bfloat16)
    check("written in bf16: the 32-bit result rounded once, as the product's own cast would",
          out.dtype == torch.bfloat16 and float((ref.float() - out.float()).abs().max() / ref.float().abs().max()) < 2 ** -7)

    B, T, H, hd = 2, 9, 3, 16
    cos, sin = (t.to(dev) for t in rope_table(T, hd, 1024.0, 0.5))
    c2, s2 = torch.cat([cos, cos], -1), torch.cat([sin, -sin], -1)
    z = (torch.randn(B, T, H, hd, device=dev) * 2).requires_grad_()
    gain = (1 + 0.3 * torch.randn(hd, device=dev)).requires_grad_()
    gq = torch.randn(B, T, H, hd, device=dev)
    ref = rotary(F.rms_norm(z, (hd,), gain), cos, sin)
    out = L.qk_rot(z, gain, c2[None, :, None, :], s2[None, :, None, :])
    ga = torch.autograd.grad(ref, (z, gain), gq)
    gb = torch.autograd.grad(out, (z, gain), gq)
    check("queries and keys: norm, gain and rotation in one go = the plain three, and both gradients",
          rel(ref, out) < 1e-5 and rel(ga[0], gb[0]) < 1e-5 and rel(ga[1], gb[1]) < 1e-5,
          f"{rel(ref, out):.1e} / {rel(ga[0], gb[0]):.1e} / {rel(ga[1], gb[1]):.1e}")
    check("the rotation with full-width tables = the rotation of two halves", rel(rotary(z, cos, sin), rotary(z, c2, s2)) < 1e-6)

    # ---- the loss step
    V, Dh, n = 1000, 48, 37
    bad = []
    for kind in ("sig", "tanh"):
        for K in (1, 3):
            h = torch.randn(n, Dh, device=dev) * 3
            W = torch.randn(V, Dh, device=dev) * 0.3
            tc = torch.randint(0, V, (n, K), device=dev)
            tc[3, 0] = -1                                        # a word with no next word
            tc[5, K - 1] = -1
            if K > 1:                                            # the same next word asked twice; and a row of zeros
                tc[7, 1] = tc[7, 0]
                tc[8, 2] = tc[8, 0]
                tc[9] = 0
                tc[9, 1] = -1
            wk = torch.tensor([1.0, 0.5, 0.25][:K], device=dev)
            nv = (tc[:, 0] >= 0).sum().float()
            args = (h, W, tc, wk, None, None, kind, 15.0, torch.tensor(0.0, device=dev), nv, float(n), V)
            a, b = LS._slice(*args), LS._slice_lean(*args)
            bad.append(max(rel(a[0], b[0]), rel(a[1], b[1]), rel(a[3], b[3]), rel(a[4], b[4])))
    check("the loss step: loss, the stream's gradient and the word table's gradient = the plain slice's "
          "(both caps, 1 and 3 words ahead, missing and repeated next words)", max(bad) < 1e-5, f"worst {max(bad):.1e}")

    class C:
        cap, softcap, lean = "sig", 15.0, 0
    h = torch.randn(4, 50, Dh, device=dev).requires_grad_()
    W = (torch.randn(V, Dh, device=dev) * 0.3).requires_grad_()
    tg = torch.randint(0, V, (4, 50, 3), device=dev)
    tg[0, :5] = -1
    res = []
    for lean in (0, 1):
        C.lean = lean
        loss, ce, kd = LS.head_loss(h, W, tg, (1.0, 0.5, 0.25), C, chunk=64)
        res.append((loss, ce) + torch.autograd.grad(loss, (h, W)))
    check("the whole loss in slices, lean = plain", max(rel(a, b) for a, b in zip(*res)) < 1e-5,
          f"{float(res[0][0]):.5f} / {float(res[1][0]):.5f}")
    th, tW = torch.randn(4 * 50, 32, device=dev), torch.randn(V, 32, device=dev)
    C.lean = 1
    lk = LS.head_loss(h, W, tg, (1.0, 0.5, 0.25), C, teacher_h=th, teacher_W=tW, kd_weight=0.3, chunk=64)
    C.lean = 0
    lp = LS.head_loss(h, W, tg, (1.0, 0.5, 0.25), C, teacher_h=th, teacher_W=tW, kd_weight=0.3, chunk=64)
    check("with a mentor the loss step is the plain one (lean has none)", float(lk[0]) == float(lp[0]) and float(lk[2]) > 0)

    # ---- the whole model, lean and plain on the same weights
    torch.manual_seed(1)
    xi = torch.randint(1, 512, (2, 64), device=dev)
    xi[:, 30] = 0
    yi = torch.roll(xi, -1, 1)
    bee = wake(Gnosis2(Gnosis2Config(**{**TINY, "mod_layers": ""})).to(dev)).train()      # Bee-135's layers, small
    bee.set_progress(1.0)
    n_canon = sum(b.canon_a is not None for b in bee.blocks) + sum(b.canon_c is not None for b in bee.blocks)
    (l0, g0), (l1, g1) = both(bee, xi, yi)
    check(f"Bee-135's layers (small: {n_canon} Canons, value tables, shifted keys, skips, back-out): lean = plain, "
          f"the loss and every one of the {len(g0)} gradients", abs(l0 - l1) < 1e-5 and set(g0) == set(g1) and worst(g0, g1) < 1e-3,
          f"loss {l0:.6f} / {l1:.6f}; worst gradient {worst(g0, g1):.1e} apart")
    keep = (bee.cfg.lean_rows, bee.cfg.lean_rows_back)
    for rows, back, what in ((16, 0, "a word's mean taken in pieces (lean_rows)"), (16, 1, "and on Canon's way back too (lean_rows_back)"),
                             (0, 0, "and as one sum (lean_rows=0)")):
        bee.cfg.lean_rows, bee.cfg.lean_rows_back = rows, back           # 64 wide: 4 sums of 16
        (l0, g0), (l1, g1) = both(bee, xi, yi)
        check(f"{what}: the same loss and gradients", abs(l0 - l1) < 1e-5 and worst(g0, g1) < 1e-3,
              f"loss {l0:.6f} / {l1:.6f}; worst gradient {worst(g0, g1):.1e} apart")
    bee.cfg.lean_rows, bee.cfg.lean_rows_back = keep
    check("pieces of 64 are lean's default, and a width they do not divide keeps the one sum",
          keep == (64, 1) and torch.equal(L.norm(torch.ones(2, 3, 100), None, 64), L.norm(torch.ones(2, 3, 100), None, 0)))
    routed = wake(Bee6(Bee6Config(**{**TINY, "mod_layers": "2,4"})).to(dev)).train()
    routed.set_progress(1.0)
    (l0, g0), (l1, g1) = both(routed, xi, yi)
    check("a Bee6 with routed layers: lean = plain", abs(l0 - l1) < 1e-5 and set(g0) == set(g1) and worst(g0, g1) < 1e-3,
          f"loss {l0:.6f} / {l1:.6f}; worst gradient {worst(g0, g1):.1e} apart")

    # ---- a real bf16 pass (as on the GPU): the stream's format, and how far the bf16 stream moves the numbers
    seen = []
    hooks = [b.register_forward_hook(lambda m_, a_, o_: seen.append((o_[0] if isinstance(o_, tuple) else o_).dtype)) for b in bee.blocks]
    amp = True
    (l0, g0), (l1, g1) = both(bee, xi, yi, amp)
    kinds = set(seen)
    seen.clear()
    check("in bf16 products, lean keeps the stream 32-bit and matches the plain layers",
          kinds == {torch.float32} and abs(l0 - l1) < 0.01 * abs(l0) and cos_mean(g0, g1) > 0.995,
          f"loss {abs(l0 - l1) / abs(l0):.3%} apart, gradients {cos_mean(g0, g1):.4f} alike")
    (l0, g0), (l1, g1) = both(bee, xi, yi, amp, s16=1)
    kinds = set(seen[len(bee.blocks):])                      # (the second pass is the lean one)
    for hk in hooks:
        hk.remove()
    check("stream16: every layer hands on a bf16 stream, and loss and gradients stay with the plain layers'",
          kinds == {torch.bfloat16} and abs(l0 - l1) < 0.01 * abs(l0) and cos_mean(g0, g1) > 0.99 and set(g0) == set(g1),
          f"loss {abs(l0 - l1) / abs(l0):.3%} apart, gradients {cos_mean(g0, g1):.4f} alike")
    bee.cfg.lean, bee.cfg.stream16 = 1, 1
    l32, _ = grads_of(bee, xi, yi)
    bee.cfg.lean = 0
    lp, _ = grads_of(bee, xi, yi)
    check("stream16 without bf16 products (CPU, samples, tests) changes nothing", abs(l32 - lp) < 1e-5)

    # ---- reading and writing with lean on
    bee.eval()
    outs = []
    for lean in (0, 1):
        bee.cfg.lean, bee.cfg.stream16 = lean, lean
        with torch.no_grad():
            lg, _ = bee.read(xi[:1, :32])
        torch.manual_seed(7)
        wr, _ = bee.generate([5, 6, 7], max_new_tokens=6, eos=None)
        outs.append((lg, wr))
    bee.train()
    check("reading and writing (the trainer's samples, chat): the same with lean on", rel(outs[0][0], outs[1][0]) < 1e-5 and outs[0][1] == outs[1][1])

    # ---- the startup check and its fall-backs
    said = []
    bee.cfg.lean, bee.cfg.stream16 = 1, 1
    bee.startup_check(dev, said.append)
    check("the startup check passes lean + stream16 and says how far apart", bee.cfg.lean == 1 and bee.cfg.stream16 == 1
          and any("[lean]" in s and "alike" in s for s in said), said[-1] if said else "")
    bee.cfg.lean, bee.cfg.stream16 = 0, 1
    bee.startup_check(dev, said.append)
    check("stream16 without lean is refused, not half-applied", bee.cfg.stream16 == 0 and bee.cfg.lean == 0)
    Nm = L._FNS[bee.cfg.lean_rows, False][0]                 # the norm this body runs
    keep = Nm.forward

    def off(ctx, x_, dt):                                    # a lean norm that is wrong: the check must catch it
        out_ = keep(ctx, x_, dt)
        return out_ + 0.5 * torch.randn_like(out_)
    Nm.forward = staticmethod(off)
    try:
        bee.cfg.lean, bee.cfg.stream16 = 1, 0
        bee.startup_check(dev, said.append)
    finally:
        Nm.forward = staticmethod(keep)
    check("a lean pass that disagrees is caught at the start: the layers run the plain way", bee.cfg.lean == 0, said[-1])
    bee.cfg.lean = 1
    sd = copy.deepcopy(bee.state_dict())
    plain = Gnosis2(Gnosis2Config(**{**TINY, "mod_layers": ""})).to(dev)
    check("the same checkpoint either way: lean adds no weights", set(plain.state_dict()) == set(sd)
          and plain.load_state_dict(sd, strict=True) is not None)

    # ---- the compiler's plan: fewer bytes
    try:
        sets = [f"{k_}={v}" for k_, v in {**TINY, "mod_layers": "", "n_layer": 2}.items()]
        d0, cfg0 = TF.layers_traffic(sets, 0, torch.device("cpu"), B=2, T=64)
        d1, _ = TF.layers_traffic(sets + ["lean=1"], 0, torch.device("cpu"), B=2, T=64)
        s0, _ = TF.summary(d0, cfg0.d_model * 2)
        s1, _ = TF.summary(d1, cfg0.d_model * 2)
        a = s0[("in", "small operations")] + s0[("back", "small operations")]
        b = s1[("in", "small operations")] + s1[("back", "small operations")]
        planned = True
    except Exception as e:                                   # (no C++ compiler on this machine, or a compiler that plans otherwise)
        term.info(f"the compiler's plan could not be read here ({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''})")
        planned = False
    if planned:
        check("the compiler's plan: lean's small operations move fewer bytes per word (python3 -m bee6.traffic)", b < 0.9 * a,
              f"{a / cfg0.n_layer:.0f} -> {b / cfg0.n_layer:.0f} U a layer on this small body")

    if dev.type == "cuda":
        from gnosis2.race135 import B135
        torch.manual_seed(0)
        mg = Gnosis2(Gnosis2Config(**{**B135, "n_layer": 4, "key_offset": "3", "lean": 1, "stream16": 1})).to(dev)
        mg.startup_check(dev, lambda s_: term.info(s_))
        check("on the GPU the startup check keeps lean + stream16", mg.cfg.lean == 1 and mg.cfg.stream16 == 1)
        mg.train()
        mg.set_progress(1.0)
        xg = torch.randint(1, 49152, (4, 256), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = mg.losses(xg, xg)
        loss.backward()
        mg.zero_grad(set_to_none=True)
        mg.enable_compile()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lc, _ = mg.losses(xg, xg)
        lc.backward()
        check("the compiled lean step runs on the GPU and agrees with the uncompiled one",
              torch.isfinite(lc).item() and abs(lc.item() - loss.item()) < 0.02, f"{lc.item():.4f} vs {loss.item():.4f}")
    term.tests_passed("lean")


if __name__ == "__main__":
    main()
