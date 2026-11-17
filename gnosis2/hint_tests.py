"""Tests of the phrase hint (gnosis2/model.py) and of the 135M race's two bodies. CPU, about a minute.

    python3 -m gnosis2.hint_tests
"""
from __future__ import annotations

import sys

import torch

from oazy import term

SMALL = dict(d_model=64, n_layer=6, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=128,
             ctx="128:0", vocab_size=512, key_offset="1,5", mod_layers="", fp8=0)
HINT = dict(hint_rows=257, hint_dim=8, hint_heads=3, hint_orders="2,3")
FAILS = []


def check(name, ok, detail=""):
    term.check(name, ok, detail)
    if not ok:
        FAILS.append(name)


def main():
    from gnosis2.model import Gnosis2, Gnosis2Config
    from gnosis2.race135 import B135, C135, set_words
    torch.manual_seed(0)
    base = Gnosis2(Gnosis2Config(**SMALL)).eval()
    with torch.no_grad():
        for n, p in base.named_parameters():
            p.add_(torch.randn_like(p) * 0.05)
    m = Gnosis2(Gnosis2Config(**SMALL, **HINT)).eval()
    missing = m.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(1, 512, (2, 96))
    with torch.no_grad():
        a, b = base(x)[0], m(x)[0]
    check("the hint starts as nothing: with the plain model's weights it gives the plain model's scores exactly",
          torch.equal(a, b) and sorted(missing.missing_keys) == ["hint.weight", "hint_out.weight"],
          f"new weights {missing.missing_keys}")
    with torch.no_grad():
        m.hint_out.weight.normal_(std=0.3)
        z = m(x)[0]
        x2 = x.clone()
        x2[:, 60:] = torch.randint(1, 512, (2, 36))
        z2 = m(x2)[0]
    check("it matters once learned", not torch.allclose(z, b, atol=1e-3))
    check("still causal: changing words from 60 on leaves every score before 60 unchanged",
          torch.allclose(z[:, :60], z2[:, :60], atol=1e-5) and not torch.allclose(z[:, 60:], z2[:, 60:]),
          f"gap before {(z[:, :60] - z2[:, :60]).abs().max().item():.1e}")
    with torch.no_grad():
        h = m._hint(x)
        xs = x.clone()
        xs[:, 10] = (xs[:, 10] + 1) % 512
        hs = m._hint(xs)
    same = (h - hs).abs().amax(-1) < 1e-7
    check("a word's hint depends on that word and the two before it, nothing else",
          bool(same[:, :10].all() and same[:, 13:].all() and not same[:, 10:13].any()))
    check("the first word has no phrase behind it: no hint", bool((h[:, 0].abs().amax(-1) < 1e-7).all()) and bool((h[:, 2].abs().amax(-1) > 0).all()))
    for size in (7, 1):
        with torch.no_grad():
            full, _ = m.read(x[:1, :40])
            st, outs = None, []
            for i in range(0, 40, size):
                lg, st = m.read(x[:1, i:min(i + size, 40)], st)
                outs.append(lg)
        gap = (full - torch.cat(outs, 1)).abs().max().item()
        check(f"reading on in pieces of {size} (chat, samples) = reading all at once", gap < 1e-4, f"largest gap {gap:.1e}")
    from bee5.optim import groups
    mu, ad = groups(m)
    ids = {id(p) for p in ad["ve"]}
    check("the hint rows learn in Adam's value-table group, the squeeze in the gates group, neither in Muon's",
          id(m.hint.weight) in ids and any(p is m.hint_out.weight for p in ad["gates"])
          and all(not n.startswith("hint") for n, _ in mu))
    m.train()
    m.set_progress(0.0)
    loss, _ = m.losses(x, x)
    loss.backward()
    used = (m.hint.weight.grad.abs().sum(-1) > 0).float().mean().item()
    check("a training step: finite loss, the squeeze and the rows that were read get a gradient",
          torch.isfinite(loss).item() and m.hint_out.weight.grad.abs().sum() > 0 and used > 0, f"loss {loss.item():.3f}, {used * 100:.0f}% of rows read")
    # a phrase rule: after (a, b) comes (a * 7 + b) % 512. Only a check that the hinted body trains as well as the plain one:
    # measured over 600 steps and two seeds the two curves are the same within seed noise, so this toy shows no gain.
    def toy(n=64, T=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        s = torch.randint(1, 60, (n, T), generator=g)
        for t in range(2, T, 3):
            s[:, t] = (s[:, t - 2] * 7 + s[:, t - 1]) % 512
        return s
    res = {}
    for name, extra in (("plain", {}), ("hint", HINT)):
        torch.manual_seed(1)
        mm = Gnosis2(Gnosis2Config(**SMALL, **extra)).train()
        mm.set_progress(1.0)
        opt = mm.make_optimizer(2e-3, torch.device("cpu"))
        for step in range(300):
            xb = toy(seed=step)
            loss, _ = mm.losses(xb[:, :-1], xb[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        mm.eval()
        with torch.no_grad():
            xb = toy(seed=9999)
            res[name] = mm.losses(xb[:, :-1], xb[:, 1:])[0].item()
    check("300 steps on a phrase rule: the hinted body trains as well as the plain one (within seed noise, 0.08)", res["hint"] <= res["plain"] + 0.08,
          f"plain {res['plain']:.3f}, hint {res['hint']:.3f}")
    from oazy.train import build
    B, _ = build("gnosis2", "big", dict(B135))
    C, _ = build("gnosis2", "big", dict(C135))
    nb, nc = B.num_params(), C.num_params()
    check("full size: Bee-135 is 130M to 136M counting everything, and the hinted one is within 0.2M of it",
          130e6 < nb < 136e6 and abs(nb - nc) < 2e5 and len(C.ve) == 0,
          f"B {nb / 1e6:.2f}M, C {nc / 1e6:.2f}M (hint {C.hint_params() / 1e6:.2f}M); {C.layout()}")
    xb = torch.randint(1, 49000, (1, 64))
    with torch.no_grad():
        ok = torch.isfinite(C.eval()(xb)[0]).all().item()
    check("the full-size hinted body runs", ok, " ".join(set_words("c")))
    print()
    print("all passed" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
