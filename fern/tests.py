"""Self-tests for Fern: each word sees exactly what it should, writing == scoring, batching is exact.

    python -m fern.tests            (CPU)
    python -m fern.tests --device cuda
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from engram.tests import check
from oazy.gpt import GPT, GPTConfig
from .model import Fern, FernConfig, level_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    print(f"Fern self-tests on {dev}")

    big = Fern(FernConfig())
    n, ng = big.num_params(), GPT(GPTConfig()).num_params()
    check("full size is about 200M parameters", 190e6 < n < 210e6, f"(Fern {n / 1e6:.1f}M, GPT-2 small {ng / 1e6:.1f}M)")
    del big

    P = 8
    torch.manual_seed(0)
    cfg = FernConfig(vocab_size=300, d_model=64, n_layer=2, n_head=2, head_k=32, head_v=32, mlp_hidden=96,
                     max_len=64, passage=P, unfold_layers=2, unfold_heads=2, unfold_hidden=96)
    m = Fern(cfg).to(dev).eval()
    rounds = [level_of(i, P) for i in range(1, P + 1)]
    check("the frond: rounds of an 8-word passage", rounds == [3, 2, 3, 1, 3, 2, 3, 0], str(rounds))

    N = 40
    seq = torch.randint(0, 300, (1, N + 1), device=dev)
    with torch.no_grad():
        base_lp = F.log_softmax(m.seq_logits(seq)[0].float(), -1)

    def visible(p, q):
        """Should the chance of word p depend on word q? (passages start after position 0)"""
        if q < 1:
            return True
        jp, jq = (p - 1) // P, (q - 1) // P
        if jq < jp:
            return True
        if jq > jp:
            return False
        return level_of((q - 1) % P + 1, P) < level_of((p - 1) % P + 1, P)

    leaks, felt, should = 0, 0, 0
    for q in range(0, N + 1):
        s2 = seq.clone()
        s2[0, q] = (s2[0, q] + 7) % 300
        with torch.no_grad():
            lp = F.log_softmax(m.seq_logits(s2)[0].float(), -1)
        for p in range(1, N + 1):
            if p == q:
                continue
            d = float((lp[p - 1] - base_lp[p - 1]).abs().max())
            if visible(p, q):
                should += 1
                felt += d > 1e-5
            elif d > 1e-5:
                leaks += 1
    check("no peeking: no word's chance depends on a word it must not see", leaks == 0, f"({leaks} leaks)")
    check("...and words do use what they are allowed to see", felt > 0.9 * should, f"({felt}/{should})")

    # writing round by round gives exactly the chances that scoring gives
    torch.manual_seed(1)
    prompt = seq[0, :13].tolist()
    rec = []
    new = m.unfold_passage(prompt, temperature=1.0, top_k=0, record=rec)
    full = torch.tensor([prompt + new], device=dev)
    st = torch.tensor([len(prompt) - 1], device=dev)
    with torch.no_grad():
        pr = F.softmax(m.seq_logits(full, st)[0].float(), -1)
    err = 0.0
    for r in rec:
        for i, tok, p in r["words"]:
            err = max(err, abs(p - float(pr[len(prompt) - 1 + i - 1, tok])))
    check("writing round by round == scoring the written text", err < 1e-4, f"(max diff {err:.1e})")
    check("5 rounds would write 16 words; here 4 rounds wrote 8", len(rec) == 4 and len(new) == P)

    # several texts of different lengths in one batch == one at a time (HellaSwag needs this)
    a = torch.randint(0, 300, (1, 30), device=dev)
    b = torch.randint(0, 300, (1, 21), device=dev)
    xb = torch.zeros(2, 30, dtype=torch.long, device=dev)
    xb[0], xb[1, :21] = a[0], b[0]
    starts = torch.tensor([11, 5], device=dev)
    ends = torch.tensor([29, 20], device=dev)
    with torch.no_grad():
        lb = m.seq_logits(xb, starts, ends).float()
        la1 = m.seq_logits(a, starts[:1]).float()
        lb1 = m.seq_logits(b, starts[1:]).float()
    e1 = (lb[0, 11:] - la1[0, 11:]).abs().max().item()
    e2 = (lb[1, 5:20] - lb1[0, 5:]).abs().max().item()
    check("a batch of different lengths == one text at a time", max(e1, e2) < 1e-4, f"(max diff {max(e1, e2):.1e})")

    # training: the grid moves, the loss is finite and reaches every part
    m.train()
    x = torch.randint(0, 300, (2, 64), device=dev)
    _, loss, _ = m(x[:, :-1], x[:, 1:])
    loss.backward()
    check("training loss is finite", bool(torch.isfinite(loss.detach())), f"({loss.item():.3f}, ln 300 = 5.70)")
    for name in ["blank", "where", "round_emb", "past_where", "unfold.0.glance.qkv.weight", "blocks.0.mem.trust.weight"]:
        g = dict(m.named_parameters())[name].grad
        check(f"training reaches '{name}'", g is not None and float(g.abs().sum()) > 0)

    # it can learn: a short fixed text memorised in a few steps
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    xs = torch.randint(0, 300, (4, 65), device=dev)
    first = None
    for i in range(60):
        _, loss, _ = m(xs[:, :-1], xs[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(loss)
    check("it learns (memorising a few texts)", float(loss) < 0.5 * first, f"({first:.2f} -> {float(loss):.2f})")
    print("all self-tests passed")


if __name__ == "__main__":
    main()
