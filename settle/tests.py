"""Self-tests for Settle. Run:  python -m settle.tests [--device cuda]

1. No cheating: changing a word never changes the predictions for earlier positions.
2. Generation == training: adding words one at a time (chat) gives the same predictions
   as settling the whole text at once (training), whether or not the sheet fully settles.
3. Gradients reach every part of the model.
"""
from __future__ import annotations

import argparse

import torch

from .model import Settle, SettleConfig


def small(**kw):
    base = dict(vocab_size=300, d_embed=32, rows=4, channels=16, hidden=64, gist_dim=16, lens_width=3,
                lens_hidden=48, settle_cap=30, lens_cap=12, max_len=40, link_heads=2, link_dim=8, settle_tol=1e-4)
    base.update(kw)
    return SettleConfig(**base)


def stir(m):
    # random init barely moves the sheet; make the rule strong enough that the tests mean something
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.mul_(5)


def test_no_future_leak(device):
    torch.manual_seed(0)
    m = Settle(small()).to(device).eval()
    stir(m)
    x = torch.randint(0, 300, (2, 40), device=device)
    a, _, _ = m(x)
    for pos in (0, 7, 20, 39):
        x2 = x.clone()
        x2[:, pos] = (x2[:, pos] + 11) % 300
        b, _, _ = m(x2)
        before = (a[:, :pos] - b[:, :pos]).abs().max().item() if pos else 0.0
        after = (a[:, pos:] - b[:, pos:]).abs().max().item()
        assert before == 0.0, f"future leaked into the past at position {pos}: {before}"
        assert after > 0, "changing a word had no effect at all"
    print("ok  no future leak")


def test_generation_matches_training(device):
    for cap, tol in ((30, 1e-4), (4, 1e-6)):          # settles fully / stops at the cap
        torch.manual_seed(1)
        m = Settle(small(settle_cap=cap, lens_cap=max(2, cap // 2), settle_tol=tol)).to(device).eval()
        stir(m)
        x = torch.randint(0, 300, (1, 40), device=device)
        with torch.no_grad():
            par, _, st = m(x)
        state = m.new_state(device)
        seq = torch.stack([m.feed(state, int(t))[0] for t in x[0]])
        diff = (seq - par[0].float()).abs().max().item()
        assert diff < 1e-3, f"generation differs from training by {diff}"
        print(f"ok  generation == training (cap {cap}): max difference {diff:.1e}, steps per word {state['steps'][:12]}...")


def test_gradients(device):
    torch.manual_seed(2)
    m = Settle(small()).to(device)
    x = torch.randint(0, 300, (2, 40), device=device)
    _, loss, _ = m(x, x)
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, f"no gradient for {dead}"
    print("ok  gradients reach every parameter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    test_no_future_leak(dev)
    test_generation_matches_training(dev)
    test_gradients(dev)
    print("all Settle self-tests passed")


if __name__ == "__main__":
    main()
