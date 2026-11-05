"""Self-tests for Fern2: its size, growing both halves, the reader's lesson, the mentor, Muon.

    python -m fern2.tests            (CPU, ~1 min)
    python -m fern2.tests --device cuda

Fern2's unfolding is Fern's own code, so Fern's checks (no peeking, writing == scoring, batches) are
run here on a fully grown Fern2 too, after checking that a fully grown Fern2 IS a Fern.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from memtrix.tests import check
from ember.mentor import mentor_loss
from fern.model import Fern, FernConfig
from oazy.gpt import GPT, GPTConfig
from oazy.train import make_optimizer, carry_state
from .model import Fern2, Fern2Config

SMALL = dict(vocab_size=300, d_model=64, n_layer=6, n_head=2, head_k=32, head_v=32, mlp_hidden=96, max_len=64,
             passage=8, unfold_layers=3, unfold_heads=2, unfold_hidden=96)
PLAN = dict(grow="2:0,4:0.1,6:0.3", unfold_grow="1:0,2:0.1,3:0.3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    print(f"Fern2 self-tests on {dev}")
    torch.manual_seed(0)

    big = Fern2(Fern2Config())
    n, ng = big.full_params(), GPT(GPTConfig()).num_params()
    check("fully grown it is smaller than GPT-2 small", n < ng, f"(Fern2 {n / 1e6:.1f}M, GPT-2 small {ng / 1e6:.1f}M)")
    check("it starts a third deep", big.depth_text() == "3 reader + 1 unfolder blocks", big.depth_text())
    del big

    # ---- growing both halves
    cfg = Fern2Config(**SMALL, **PLAN)
    m = Fern2(cfg).to(dev)
    full = Fern(FernConfig(**SMALL)).num_params() + cfg.d_model + cfg.d_model ** 2   # + the reader's norm and layer
    check("starts shallow", (len(m.blocks), len(m.unfold)) == (2, 1), m.depth_text())
    check("fully grown size = Fern's size (+ the next-word layer)", m.full_params() == full, f"{m.full_params()} vs {full}")
    check("no growth before its time", not m.grow_to(0.05) and len(m.blocks) == 2)
    check("grows on time", m.grow_to(0.1) and (len(m.blocks), len(m.unfold)) == (4, 2), m.depth_text())
    check("a copy starts as its original",
          all(torch.equal(a, b) for a, b in zip(m.blocks[2].state_dict().values(), m.blocks[0].state_dict().values()))
          and all(torch.equal(a, b) for a, b in zip(m.unfold[1].state_dict().values(), m.unfold[0].state_dict().values())))
    check("copies are separate weights", m.unfold[1].up.weight.data_ptr() != m.unfold[0].up.weight.data_ptr())
    m.grow_to(1.0)
    check("ends fully grown", (len(m.blocks), len(m.unfold)) == (6, 3) and m.num_params() == full, m.depth_text())

    q = Fern2(Fern2Config(**SMALL, grow="3:0,6:0.5", unfold_grow="2:0,3:0.5", grow_mode="quiet")).to(dev).eval()
    seq = torch.randint(0, 300, (2, 41), device=dev)
    with torch.no_grad():
        a = q.seq_logits(seq)
        q.grow_to(0.5)
        b = q.seq_logits(seq)
    check("quiet growth: same guesses right after growing", torch.allclose(a, b, atol=1e-5),
          f"max diff {float((a - b).abs().max()):.2e}")

    # ---- a half-grown checkpoint loads back exactly
    m2 = Fern2(cfg).to(dev).eval()
    m2.grow_to(0.15)
    m3 = Fern2(cfg).to(dev).eval()
    m3.load_state_dict({k: v.clone() for k, v in m2.state_dict().items()})
    with torch.no_grad():
        ok = torch.allclose(m2.seq_logits(seq), m3.seq_logits(seq), atol=1e-6)
    check("a half-grown checkpoint loads back exactly", ok and m3.depth_text() == m2.depth_text(), m3.depth_text())

    # ---- fully grown, Fern2 is a Fern: same weights, same chances for every word
    m.eval()
    f = Fern(FernConfig(**SMALL)).to(dev).eval()
    f.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith(("reader_norm", "reader_proj"))})
    with torch.no_grad():
        d = float((f.seq_logits(seq) - m.seq_logits(seq)).abs().max())
    check("fully grown Fern2 gives exactly Fern's chances", d < 1e-6, f"max diff {d:.1e}")
    print("  Fern's own checks on the grown Fern2 (no peeking, writing == scoring, batches):")
    import fern.tests as ft
    P = cfg.passage
    N = 40
    with torch.no_grad():
        base_lp = F.log_softmax(m.seq_logits(seq[:1])[0].float(), -1)
    leaks = felt = should = 0
    for qq in range(0, N + 1):
        s2 = seq[:1].clone()
        s2[0, qq] = (s2[0, qq] + 7) % 300
        with torch.no_grad():
            lp = F.log_softmax(m.seq_logits(s2)[0].float(), -1)
        for p in range(1, N + 1):
            if p == qq:
                continue
            dd = float((lp[p - 1] - base_lp[p - 1]).abs().max())
            jp, jq = (p - 1) // P, (qq - 1) // P
            vis = qq < 1 or jq < jp or (jq == jp and ft.level_of((qq - 1) % P + 1, P) < ft.level_of((p - 1) % P + 1, P))
            if vis:
                should += 1
                felt += dd > 1e-5
            elif dd > 1e-5:
                leaks += 1
    check("no peeking: no word's chance depends on a word it must not see", leaks == 0, f"({leaks} leaks)")
    check("...and words do use what they are allowed to see", felt > 0.9 * should, f"({felt}/{should})")
    torch.manual_seed(1)
    prompt = seq[0, :13].tolist()
    rec = []
    new = m.unfold_passage(prompt, temperature=1.0, top_k=0, record=rec)
    with torch.no_grad():
        pr = F.softmax(m.seq_logits(torch.tensor([prompt + new], device=dev),
                                    torch.tensor([len(prompt) - 1], device=dev))[0].float(), -1)
    err = max(abs(p - float(pr[len(prompt) - 1 + i - 1, t])) for r in rec for i, t, p in r["words"])
    check("writing round by round == scoring the written text", err < 1e-4, f"(max diff {err:.1e})")

    # ---- the reader's own next-word lesson
    with torch.no_grad():
        y = seq.clone()
        y[:, 25:] = torch.randint(0, 300, y[:, 25:].shape, device=dev)
        ra, rb = m.reader_logits(seq), m.reader_logits(y)
    check("the reader's guesses never peek at later words", torch.allclose(ra[:, :25], rb[:, :25], atol=1e-5))
    m.train()
    x = torch.randint(0, 300, (2, 64), device=dev)
    logits, loss, st = m(x[:, :-1], x[:, 1:])
    check("training loss = Fern's loss + half the reader's",
          abs(float(loss) - float(st["ce"] + 0.5 * st["reader_ce"])) < 1e-4,
          f"({float(st['ce']):.3f} + 0.5 x {float(st['reader_ce']):.3f})")
    loss.backward()
    check("the reader's lesson reaches its norm", m.reader_norm.weight.grad is not None
          and float(m.reader_norm.weight.grad.abs().sum()) > 0)
    m.zero_grad()
    m.eval()
    with torch.no_grad():
        _, vloss, vst = m(x[:, :-1], x[:, 1:])
    check("validation loss is Fern's per-word loss alone", "reader_ce" not in vst, f"({float(vloss):.3f})")
    m.train()

    # ---- the mentor teaches the reader, not the unfolder
    teacher = torch.randn(2, 63, 300, device=dev) * 3
    _, loss, st = m(x[:, :-1], x[:, 1:])
    kd = mentor_loss(st["kd_logits"], teacher)
    total = m.mix_mentor(loss, st, kd, 0.6)
    expect = st["ce"] + 0.5 * (0.4 * st["reader_ce"] + 0.6 * kd.detach())
    check("mixed lesson = Fern + 0.5 x (0.4 x reader + 0.6 x mentor)", abs(float(total) - float(expect)) < 1e-4,
          f"({float(total):.3f})")
    m.zero_grad()
    kd.backward(retain_graph=True)
    gu = sum(float(p.grad.abs().sum()) for n_, p in m.named_parameters() if n_.startswith("unfold") and p.grad is not None)
    gr = sum(float(p.grad.abs().sum()) for n_, p in m.named_parameters() if n_.startswith("blocks") and p.grad is not None)
    check("the mentor's lesson reaches the reader and never the unfolder", gr > 0 and gu == 0, f"(reader {gr:.2e}, unfolder {gu:.1e})")
    m.zero_grad()

    # ---- Muon on both halves, growing mid-way, and it learns
    torch.manual_seed(2)
    g2 = Fern2(cfg).to(dev)
    opt = make_optimizer(g2, 1e-3, dev, "muon", 0.02)
    ids = [id(p) for grp in opt.param_groups for p in grp["params"]]
    check("every parameter in exactly one optimizer group", sorted(ids) == sorted(id(p) for p in g2.parameters()))
    muon = {id(p) for p in opt.opts[0].param_groups[0]["params"]}
    check("Muon also straightens the unfolder's grids", all(id(p) in muon for p in g2.unfold[0].parameters() if p.dim() == 2
                                                            and p.shape[0] >= 64))
    xs = torch.randint(0, 300, (4, 65), device=dev)
    losses = []
    for i in range(80):
        if i == 40 and g2.grow_to(0.1):
            old, opt = opt, make_optimizer(g2, 1e-3, dev, "muon", 0.02)
            carry_state(old, opt)
        _, loss, st = g2(xs[:, :-1], xs[:, 1:])
        loss.backward()
        for grp in opt.param_groups:
            grp["lr"] = 3e-3 * grp["lr_scale"]
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(st["ce"]))
    check("Muon + AdamW learn, and keep learning after growing", losses[-1] < 0.5 * losses[0] and losses[-1] < losses[39] + 0.3,
          f"Fern loss {losses[0]:.2f} -> {losses[39]:.2f} (grow) -> {losses[-1]:.2f}")
    print("all Fern2 self-tests passed")


if __name__ == "__main__":
    main()
