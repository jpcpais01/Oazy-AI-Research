"""Self-tests for Fern3: no peeking, writing == scoring, batches, recall, growing, the mentor's path.

    python -m fern3.tests            (CPU, ~2 min)
    python -m fern3.tests --device cuda
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from memtrix.tests import check
from ember.mentor import mentor_loss
from fern.model import level_of
from oazy.gpt import GPT, GPTConfig
from oazy.train import make_optimizer, carry_state
from .model import Fern3, Fern3Config
from oazy import term

P = 4
SMALL = dict(vocab_size=300, d_model=64, n_layer=2, max_len=128, passage=P, unfold_layers=2, unfold_heads=2,
             unfold_hidden=96, think_layers=1, recall=3, grow="1:0,2:0.3", unfold_grow="1:0,2:0.3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Fern3", dev)
    torch.manual_seed(0)

    big = Fern3(Fern3Config())
    n, ng = big.full_params(), GPT(GPTConfig()).num_params()
    check("fully grown it is smaller than GPT-2 small", n < ng, f"(Fern3 {n / 1e6:.1f}M, GPT-2 small {ng / 1e6:.1f}M)")
    check("no fixed-size memory: it keeps every word", big.memory_numbers() is None)
    del big

    m = Fern3(Fern3Config(**SMALL)).to(dev)
    m.grow_to(1.0)
    m.eval()

    # ---- no peeking: a word's chance depends only on what it may see
    N = 44
    seq = torch.randint(0, 300, (1, N + 1), device=dev)
    with torch.no_grad():
        base_lp = F.log_softmax(m.seq_logits(seq)[0].float(), -1)

    def visible(p, q):
        """passages start after position 0 (position 0 is context)"""
        if q < 1:
            return True
        jp, jq = (p - 1) // P, (q - 1) // P
        if jq != jp:
            return jq < jp
        return level_of((q - 1) % P + 1, P) < level_of((p - 1) % P + 1, P)

    leaks = felt = should = far = far_felt = 0
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
                if q >= 1 and (p - 1) // P - (q - 1) // P >= 3:
                    far += 1
                    far_felt += d > 1e-5
            elif d > 1e-5:
                leaks += 1
    check("no peeking: no word's chance depends on a word it must not see", leaks == 0, f"({leaks} leaks)")
    check("...and words do use what they are allowed to see", felt > 0.9 * should, f"({felt}/{should})")
    check("...including words 3+ passages back (gists and recall)", far_felt > 0.9 * far, f"({far_felt}/{far})")

    # ---- writing round by round gives exactly the chances that scoring gives
    torch.manual_seed(1)
    prompt = seq[0, :21].tolist()
    rec = []
    new = m.unfold_passage(prompt, temperature=1.0, top_k=0, record=rec)
    with torch.no_grad():
        pr = F.softmax(m.seq_logits(torch.tensor([prompt + new], device=dev),
                                    torch.tensor([len(prompt) - 1], device=dev))[0].float(), -1)
    err = max(abs(p - float(pr[len(prompt) - 1 + i - 1, t])) for r in rec for i, t, p in r["words"])
    check("writing round by round == scoring the written text", err < 1e-4, f"(max diff {err:.1e})")

    # ---- texts of different lengths and starts in one batch == one at a time (HellaSwag)
    a = torch.randint(0, 300, (1, 30), device=dev)
    b = torch.randint(0, 300, (1, 21), device=dev)
    xb = torch.zeros(2, 30, dtype=torch.long, device=dev)
    xb[0], xb[1, :21] = a[0], b[0]
    starts, ends = torch.tensor([11, 6], device=dev), torch.tensor([29, 20], device=dev)
    with torch.no_grad():
        lb = m.seq_logits(xb, starts, ends).float()
        la1 = m.seq_logits(a, starts[:1]).float()
        lb1 = m.seq_logits(b, starts[1:]).float()
    e = max((lb[0, 11:] - la1[0, 11:]).abs().max().item(), (lb[1, 6:20] - lb1[0, 6:]).abs().max().item())
    check("a batch of different lengths == one text at a time", e < 1e-4, f"(max diff {e:.1e})")

    # ---- recall picks only old passages (3+ back), the best-matching ones
    ids = torch.randint(0, 300, (60,)).tolist()
    idx, ok, pad = m.recalled(ids)
    bad = [(j, k) for j, row in enumerate(idx) for k, v in zip(row, ok[j]) if v and k > j - 3]
    check("recall only reaches passages at least 3 back", not bad, f"({sum(map(sum, ok))} recalls, {len(bad)} too recent)")
    check("every passage 3+ in recalls 3 (when there are 3 to choose from)",
          all(sum(ok[j]) == min(3, j - 2) for j in range(3, len(idx))), str([sum(r) for r in ok]))

    # ---- the folder's own next-word guesses never peek
    with torch.no_grad():
        y = seq.clone()
        y[:, 25:] = torch.randint(0, 300, y[:, 25:].shape, device=dev)
        ra, rb = m.reader_logits(seq), m.reader_logits(y)
    check("the folder's next-word guesses never peek at later words", torch.allclose(ra[:, :25], rb[:, :25], atol=1e-5))

    # ---- training: losses, every part learns, the mentor teaches only the folder
    m.train()
    x = torch.randint(0, 300, (2, 61), device=dev)
    _, loss, st = m(x[:, :-1], x[:, 1:])
    loss.backward()
    named = dict(m.named_parameters())
    for name in ["recall_q.weight", "recall_k.weight", "recall_v.weight", "merge.pair.weight", "think.0.glance.qkv.weight",
                 "blocks.1.up.weight", "unfold.1.glance.qkv.weight", "reader_norm.weight", "floor_emb"]:
        g = named[name].grad
        check(f"training reaches '{name}'", g is not None and float(g.abs().sum()) > 0)
    m.zero_grad()
    _, loss, st = m(x[:, :-1], x[:, 1:])
    kd = mentor_loss(st["kd_logits"], torch.randn(2, 60, 300, device=dev) * 3)
    kd.backward()
    other = sum(float(p.grad.abs().sum()) for k, p in m.named_parameters() if p.grad is not None
                and k.startswith(("unfold", "think", "recall", "merge")))
    mine = sum(float(p.grad.abs().sum()) for k, p in m.named_parameters() if p.grad is not None and k.startswith("blocks"))
    check("the mentor teaches the folder, never the unfolder, thinker or recall", mine > 0 and other == 0,
          f"(folder {mine:.1e}, rest {other:.0e})")
    m.zero_grad()
    m.eval()
    with torch.no_grad():
        _, vl, vst = m(x[:, :-1], x[:, 1:])
    check("validation loss is the per-word loss alone", "reader_ce" not in vst, f"({float(vl):.3f})")

    # ---- growing, checkpoints at any depth
    g = Fern3(Fern3Config(**SMALL)).to(dev)
    check("starts shallow", g.depth_text() == "1 folder + 1 unfolder blocks", g.depth_text())
    g.grow_to(0.3)
    check("grows on time", g.depth_text() == "2 folder + 2 unfolder blocks", g.depth_text())
    h = Fern3(Fern3Config(**SMALL)).to(dev)
    h.load_state_dict({k: v.clone() for k, v in m.state_dict().items()})
    with torch.no_grad():
        d = float((h.eval().seq_logits(seq) - m.seq_logits(seq)).abs().max())
    check("a checkpoint loads back exactly", d < 1e-6, f"max diff {d:.1e}")

    # ---- recall learns: a detail seen long ago (beyond the 2 local passages) comes back
    # each text: noise, with a "key" passage (marked by word 1) somewhere among the first 6 passages,
    # then a cue passage ("the key comes next") and the key again. Its words can only be known by
    # recalling the right old passage: the unfolder sees just the last 2 passages in full, and the cue
    # says WHEN to recall, not what or where.
    torch.manual_seed(3)
    cfg = Fern3Config(**{**SMALL, "grow": "2:0", "unfold_grow": "2:0", "d_model": 96, "unfold_heads": 2,
                         "shift": False, "recall_warm_until": 0.1})
    r = Fern3(cfg).to(dev)
    opt = make_optimizer(r, 3e-3, dev, "adamw")
    second = slice(1 + 10 * P, 1 + 11 * P)

    def batch(nb=32):
        t = torch.randint(10, 300, (nb, 2 + 11 * P), device=dev)
        key = torch.randint(10, 300, (nb, P), device=dev)
        key[:, 0] = 1                                                    # a marker: "remember this"
        where = torch.randint(0, 6, (nb,), device=dev)
        for b in range(nb):
            t[b, 1 + P * where[b]: 1 + P * (where[b] + 1)] = key[b]
        t[:, 1 + 9 * P: 1 + 10 * P] = 2                                  # the cue
        t[:, second] = key
        return t, where

    def key_check():
        r.eval()
        with torch.no_grad():
            t, where = batch(64)
            lg = r.seq_logits(t)
            lo = F.cross_entropy(lg[:, second.start: second.stop - 1].transpose(1, 2).float(), t[:, second.start + 1:second.stop])
            X, ex, _ = r._pad(t, torch.zeros(64, dtype=torch.long, device=dev), torch.full((64,), t.shape[1] - 1, device=dev))
            first = r._core3(X, ex)[2][:, 11, 0]                             # passage 11 = the second key (1 pad passage)
        r.train()
        return float(lo), int((first == where + 1).sum())

    before = key_check()
    steps = 900
    for i in range(steps):
        r.grow_to(i / steps)
        t, _ = batch()
        _, loss, st = r(t[:, :-1], t[:, 1:])
        loss.backward()
        for grp in opt.param_groups:
            grp["lr"] = 3e-3 * grp["lr_scale"]
        opt.step()
        opt.zero_grad(set_to_none=True)
    after = key_check()
    check("recall learns which old passage to fetch (it moves around)", after[1] > 0.9 * 64,
          f"(right passage recalled first: {before[1]}/64 -> {after[1]}/64)")
    check("...and writes its words again from memory", after[0] < 0.5,
          f"(loss on the repeated words {before[0]:.2f} -> {after[0]:.2f}; guessing = 5.67)")
    term.tests_passed("Fern3")


if __name__ == "__main__":
    main()
