"""Self-tests for Tide: no peeking at real future words, word-by-word == training, the drafts are trained.

    python -m tide.tests            (CPU)
    python -m tide.tests --device cuda
"""
from __future__ import annotations

import argparse

import torch

from oazy.gpt import GPT, GPTConfig
from memtrix.tests import check
from .model import Tide, TideConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    print(f"Tide self-tests on {dev}")

    big = Tide(TideConfig())
    gpt = GPT(GPTConfig())
    nt, ng = big.num_params(), gpt.num_params()
    check("full size is smaller than GPT-2 small", nt < ng, f"(Tide {nt / 1e6:.1f}M vs GPT-2 {ng / 1e6:.1f}M)")
    del big, gpt

    torch.manual_seed(0)
    cfg = TideConfig(vocab_size=300, d_model=64, n_layer=2, n_head=2, head_k=32, head_v=32, mlp_hidden=96,
                     max_len=96, lookahead=5, refine_hidden=96)
    m = Tide(cfg).to(dev)
    # make the belt matter (it starts small): scramble its weights a bit
    with torch.no_grad():
        for p in m.refiner.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        m.blank.add_(torch.randn_like(m.blank))
    m.eval()
    x = torch.randint(0, 300, (2, 96), device=dev)
    with torch.no_grad():
        l1 = m(x)[0].float()
        x2 = x.clone()
        x2[:, 60] = (x2[:, 60] + 1) % 300
        l2 = m(x2)[0].float()
    check("no peeking: changing word 60 leaves the guesses for words 1-60 untouched",
          (l1[:, :60] - l2[:, :60]).abs().max().item() < 1e-5)
    check("...and does change what comes after", (l1[:, 60:] - l2[:, 60:]).abs().max().item() > 1e-4)

    with torch.no_grad():
        S = m._slots(x)
        S2 = m._slots(x2)
    check("drafts made before word 60 was read do not know it",
          (S[:, :, :60] - S2[:, :, :60]).abs().max().item() < 1e-5)

    with torch.no_grad():
        st, outs = None, []
        for t in range(96):
            lo, st = m.read(x[:, t:t + 1], st)
            outs.append(lo)
        l3 = torch.cat(outs, 1).float()
        la, s = m.read(x[:, :37])
        lb, _ = m.read(x[:, 37:], s)
        l4 = torch.cat([la, lb], 1).float()
    d3 = (l1 - l3).abs().max().item()
    check("writing word by word == training", d3 < 1e-3, f"(max diff {d3:.1e})")
    check("reading in pieces == reading at once", (l1 - l4).abs().max().item() < 1e-3)

    # the belt after reading word t, as guesses for 1..K words ahead == the training slots at t
    with torch.no_grad():
        _, st = m.read(x[:, :50])
        a = m.draft_logits(st)                                   # (B, K, V)
        b = torch.stack([torch.nn.functional.linear(S[d][:, 49], m.embed.weight) for d in range(cfg.lookahead)], 1)
    check("the live belt == what training scores", (a - b).abs().max().item() < 1e-3)

    m.train()
    _, loss, stats = m(x[:, :-1], x[:, 1:])
    loss.backward()
    check("far drafts are scored in training", any(k.startswith("draft") for k in stats), str(sorted(stats)))
    for n in ["refiner.where", "refiner.ahead.weight", "refiner.ctx.weight", "blank", "mix", "blocks.0.mem.A_log"]:
        p = dict(m.named_parameters())[n]
        check(f"training reaches '{n}'", p.grad is not None and float(p.grad.abs().sum()) > 0)
    print("all Tide self-tests passed")


if __name__ == "__main__":
    main()
