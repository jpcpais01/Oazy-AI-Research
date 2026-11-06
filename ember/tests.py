"""Self-tests for Ember: growing, checkpoints at any depth, Muon, the mentor's lesson, no peeking.

    python -m ember.tests            (CPU, ~1 min)
    python -m ember.tests --device cuda
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .model import Ember, EmberConfig
from .mentor import mentor_loss
from memtrix.model import Memtrix, MemtrixConfig
from oazy.muon import straighten
from oazy.train import make_optimizer, carry_state
from oazy import term


def check(name, ok, detail=""):
    term.check(name, ok, detail)          # ✓ / ✗ in the lab's style (oazy/term.py)


SMALL = dict(vocab_size=320, d_model=64, n_layer=6, n_head=2, head_k=16, head_v=16, mlp_hidden=96, max_len=64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Ember", dev)
    torch.manual_seed(0)

    # ---- growing
    cfg = EmberConfig(**SMALL, grow="2:0,4:0.1,6:0.3")
    m = Ember(cfg).to(dev)
    check("starts shallow", m.depth == 2, f"depth {m.depth}")
    full = Memtrix(MemtrixConfig(**SMALL)).num_params()
    check("fully grown size = Memtrix's size", m.full_params() == full, f"{m.full_params()} vs {full}")
    check("no growth before its time", not m.grow_to(0.05) and m.depth == 2)
    check("grows on time", m.grow_to(0.1) and m.depth == 4, f"depth {m.depth}")
    same = all(torch.equal(a, b) for a, b in zip(m.blocks[2].state_dict().values(), m.blocks[0].state_dict().values()))
    check("a copy starts as its original", same)
    distinct = m.blocks[2].mem.qkv.weight.data_ptr() != m.blocks[0].mem.qkv.weight.data_ptr()
    check("copies are separate weights, not shared", distinct)
    m.grow_to(0.9)
    check("ends at full depth", m.depth == 6 and m.num_params() == full, f"depth {m.depth}")
    check("the last copy is of the top blocks",
          torch.equal(m.blocks[4].mem.qkv.weight, m.blocks[2].mem.qkv.weight)
          and torch.equal(m.blocks[5].mem.qkv.weight, m.blocks[3].mem.qkv.weight))

    # quiet growth keeps what the model says exactly
    q = Ember(EmberConfig(**SMALL, grow="3:0,6:0.5", grow_mode="quiet")).to(dev).eval()
    x = torch.randint(0, 320, (2, 40), device=dev)
    with torch.no_grad():
        a = q(x)[0]
        q.grow_to(0.5)
        b = q(x)[0]
    check("quiet growth: same guesses right after growing", torch.allclose(a, b, atol=1e-5),
          f"max diff {float((a - b).abs().max()):.2e}")

    # ---- checkpoints at any depth
    m2 = Ember(cfg).to(dev)
    m2.grow_to(0.15)
    sd = {k: v.clone() for k, v in m2.state_dict().items()}
    m3 = Ember(cfg).to(dev)
    m3.load_state_dict(sd)
    with torch.no_grad():
        ok = torch.allclose(m2(x)[0], m3(x)[0], atol=1e-6)
    check("a half-grown checkpoint loads back exactly", ok and m3.depth == 4, f"depth {m3.depth}")

    # ---- no peeking at future words, and reading word by word = training
    m.eval()
    with torch.no_grad():
        y = x.clone()
        y[:, 25:] = torch.randint(0, 320, y[:, 25:].shape, device=dev)
        la, lb = m(x)[0], m(y)[0]
        check("no peeking at future words", torch.allclose(la[:, :25], lb[:, :25], atol=1e-5))
        full_logits = m(x[:1])[0][0]
        st, step_logits = None, []
        for t in range(x.shape[1]):
            lg, st = m.read(x[:1, t:t + 1], st)
            step_logits.append(lg[0, -1])
        d = float((torch.stack(step_logits) - full_logits).abs().max())
        check("reading word by word = reading all at once", d < 1e-3, f"max diff {d:.2e}")
    m.train()

    # ---- Muon
    G = torch.randn(48, 96, device=dev)
    s = torch.linalg.svdvals(straighten(G).float())
    check("straighten evens out the directions", float(s.min()) > 0.5 and float(s.max()) < 1.3,
          f"strengths {float(s.min()):.2f}..{float(s.max()):.2f} (raw {float(torch.linalg.svdvals(G).min()):.1f}.."
          f"{float(torch.linalg.svdvals(G).max()):.1f})")
    g2 = Ember(cfg).to(dev)
    opt = make_optimizer(g2, 1e-3, dev, "muon", 0.02)
    ids = [id(p) for grp in opt.param_groups for p in grp["params"]]
    check("every parameter in exactly one optimizer group", sorted(ids) == sorted(id(p) for p in g2.parameters()),
          f"{len(ids)} vs {len(list(g2.parameters()))}")
    data = torch.randint(0, 320, (8, 33), device=dev)
    data[:, 1::2] = (data[:, 0::2][:, :16] * 7 + 3) % 320          # learnable: every 2nd word follows from the one before
    losses = []
    for i in range(60):
        if i == 30 and g2.grow_to(0.1):                              # grow mid-way, keeping the optimizer memory
            old, opt = opt, make_optimizer(g2, 1e-3, dev, "muon", 0.02)
            carry_state(old, opt)
        _, loss, _ = g2(data[:, :-1], data[:, 1:])
        loss.backward()
        for grp in opt.param_groups:
            grp["lr"] = 3e-3 * grp["lr_scale"]
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    check("Muon + AdamW learn, and keep learning after growing", losses[-1] < losses[0] - 1.0 and losses[-1] < losses[29] + 0.3,
          f"loss {losses[0]:.2f} -> {losses[29]:.2f} (grow) -> {losses[-1]:.2f}")
    sd = opt.state_dict()
    opt2 = make_optimizer(g2, 1e-3, dev, "muon", 0.02)
    opt2.load_state_dict(sd)
    check("optimizer checkpoint round-trip", len(opt2.opts[0].state) == len(opt.opts[0].state))

    # ---- the mentor's lesson
    t = torch.randn(2, 10, 50257, device=dev)
    sl = torch.cat([t, torch.full((2, 10, 47), -30.0, device=dev)], dim=-1).requires_grad_(True)
    check("same guess as the mentor: nothing to learn", float(mentor_loss(sl, t)) < 1e-3, f"{float(mentor_loss(sl, t)):.2e}")
    s2 = torch.randn(2, 10, 50304, device=dev, requires_grad=True)
    kd = mentor_loss(s2, t)
    kd.backward()
    check("a different guess: positive lesson, and it teaches (gradient)", float(kd) > 0.1 and s2.grad.abs().sum() > 0,
          f"KL {float(kd):.2f}")
    term.tests_passed("Ember")


if __name__ == "__main__":
    main()
