"""Self-tests: the three versions of the memory rule agree, no peeking at future words,
and writing word by word gives exactly what training sees.

    python -m memtrix.tests            (CPU)
    python -m memtrix.tests --device cuda
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from . import memory
from .model import Memtrix, MemtrixConfig
from oazy.gpt import GPT, GPTConfig


def check(name, ok, detail=""):
    print(f"  {'ok ' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        raise SystemExit(f"self-test failed: {name}")


def rand_inputs(device, B=2, T=150, H=3, K=32, V=24, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    v = torch.randn(B, T, H, V, generator=g)
    lf = -torch.rand(B, T, H, generator=g) * 0.3
    tr = torch.rand(B, T, H, generator=g)
    S0 = torch.randn(B, H, K, V, generator=g) * 0.3
    return [x.to(device) for x in (q, k, v, lf, tr, S0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    print(f"Memtrix self-tests on {dev}")

    q, k, v, lf, tr, S0 = rand_inputs(dev)
    o1, S1, s1 = memory.recurrent(q, k, v, lf, tr, S0, return_surprise=True)
    o2, S2, s2 = memory.chunked(q, k, v, lf, tr, S0, chunk=32, return_surprise=True)
    err = max((o1 - o2).abs().max().item(), (S1 - S2).abs().max().item(), (s1 - s2).abs().max().item())
    check("chunked rule == word-by-word rule", err < 1e-4, f"(max diff {err:.1e})")

    # the rule really is "fix the memory by the surprise": after learning a pair with full trust
    # and no fade, asking with the same key returns exactly the value
    K = 16
    kk = F.normalize(torch.randn(1, 1, 1, K, device=dev), dim=-1)
    vv = torch.randn(1, 1, 1, 8, device=dev)
    _, S = memory.recurrent(kk, kk, vv, torch.zeros(1, 1, 1, device=dev), torch.ones(1, 1, 1, device=dev),
                            torch.randn(1, 1, K, 8, device=dev))
    got = torch.einsum("k,kv->v", kk[0, 0, 0], S[0, 0])
    check("full trust: memory answers the new value exactly", (got - vv[0, 0, 0]).abs().max().item() < 1e-5)

    if dev.type == "cuda":
        memory._FLA_OK = None
        if memory.fla_check(dev):
            check("flash-linear-attention kernel == reference", True)
        else:
            print("  --   flash-linear-attention kernel not used (the pure-PyTorch version will run)")

    torch.manual_seed(0)
    cfg = MemtrixConfig(vocab_size=300, d_model=64, n_layer=3, n_head=2, head_k=32, head_v=32, mlp_hidden=96, max_len=96)
    m = Memtrix(cfg).to(dev).eval()
    x = torch.randint(0, 300, (2, 96), device=dev)
    with torch.no_grad():
        l1 = m(x)[0].float()
        x2 = x.clone()
        x2[:, 60] = (x2[:, 60] + 1) % 300
        l2 = m(x2)[0].float()
    check("no peeking: changing word 60 leaves words 0-59 untouched", (l1[:, :60] - l2[:, :60]).abs().max().item() < 1e-5)
    check("...and does change what comes after", (l1[:, 60:] - l2[:, 60:]).abs().max().item() > 1e-4)

    with torch.no_grad():
        st, outs = None, []
        for t in range(96):
            lo, st = m.read(x[:, t:t + 1], st)
            outs.append(lo)
        l3 = torch.cat(outs, 1).float()
        la, s = m.read(x[:, :40])
        lb, _ = m.read(x[:, 40:], s)
        l4 = torch.cat([la, lb], 1).float()
    check("writing word by word == training", (l1 - l3).abs().max().item() < 1e-3, f"(max diff {(l1 - l3).abs().max().item():.1e})")
    check("reading in pieces == reading at once", (l1 - l4).abs().max().item() < 1e-3)

    m.train()
    _, loss, _ = m(x[:, :-1], x[:, 1:])
    loss.backward()
    names = ["A_log", "dt_bias", "trust.weight", "fade.weight", "conv_w"]
    for n in names:
        g = sum(float(p.grad.abs().sum()) for pn, p in m.named_parameters() if pn.endswith(n) and p.grad is not None)
        check(f"training reaches the memory's '{n}'", g > 0)

    g = GPT(GPTConfig(vocab_size=300, n_layer=2, n_head=2, d_model=64, max_len=96)).to(dev).eval()
    with torch.no_grad():
        d = (g(x)[0][:, :60] - g(x2)[0][:, :60]).abs().max().item()
    check("GPT baseline: no peeking", d < 1e-5)
    print("all self-tests passed")


if __name__ == "__main__":
    main()
