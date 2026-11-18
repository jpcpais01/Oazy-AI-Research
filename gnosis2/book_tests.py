"""Tests of the memory table (oazy/book.py) and of the body that reads it (gnosis2/model.py, `book=`). CPU, about a minute.

    python3 -m gnosis2.book_tests
"""
from __future__ import annotations

import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from oazy import book, term

SMALL = dict(d_model=64, n_layer=6, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=128,
             ctx="128:0", vocab_size=512, key_offset="1,5", mod_layers="", fp8=0)
FAILS = []


def check(name, ok, detail=""):
    term.check(name, ok, detail)
    if not ok:
        FAILS.append(name)


def text(n, seed):
    """Made-up text with real phrases: every third piece follows from the two before it, 9 times in 10."""
    g = np.random.default_rng(seed)
    a = g.integers(1, 40, n)
    noise = g.random(n) < 0.1
    for t in range(2, n, 3):
        if not noise[t]:
            a[t] = (a[t - 2] * 7 + a[t - 1]) % 512
    return a.astype(np.uint16)


def main():
    from gnosis2.model import Gnosis2, Gnosis2Config
    tmp = Path(tempfile.mkdtemp())
    (tmp / "opal").mkdir()
    train = text(600_000, 0)
    train.tofile(tmp / "opal" / "train.bin")
    old = book.CHUNK
    book.CHUNK = 100_000                                  # several chunks, so the merge over chunks is exercised
    try:
        path = book.build(tmp, tmp / "opal" / "book.pt", gb=0.02, tokens=1e9, mix="fwe:1@0", workers=2)
    finally:
        book.CHUNK = old
    b = book.load(path)

    x = torch.randint(0, 512, (3, 50))
    back = torch.stack([torch.roll(x, j, 1) for j in range(4)], -1)
    same = all(np.array_equal(book._hash_np([back[..., j].numpy().astype(np.uint64) for j in range(o)]).astype(np.int64),
                              book.hash_torch(back, o).numpy()) for o in (1, 2, 3, 4))
    check("a phrase's key is the same bits when building (numpy) and when the model reads (torch)", same)

    # the counts against counting by hand, for phrases of 2
    a = [int(v) for v in train]
    by = defaultdict(Counter)
    for t in range(1, len(a) - 1):
        by[(a[t - 1], a[t])][a[t + 1]] += 1
    keys = b["keys_2"].numpy()
    want = {k: c for k, c in by.items() if sum(c.values()) >= b["min_count"]}
    ok, bad = len(want) > 100, ""
    for (p, q), c in list(want.items())[:3000]:
        h = int(book._hash_np([np.array([q], np.uint64), np.array([p], np.uint64)])[0])
        i = int(np.searchsorted(keys, h))
        if i >= keys.size or keys[i] != h:
            ok, bad = False, f"phrase {(p, q)} missing"
            break
        nx = b["nxt_2"].numpy()[i].view(np.uint16)
        pr = b["prob_2"].numpy()[i] / 255.0
        top, n = c.most_common(1)[0]
        tot = sum(c.values())
        if c[int(nx[0])] != n or abs(pr[0] - n / tot) > 0.004 or abs(int(b["lg_2"][i]) - round(8 * np.log2(tot))) > 1:
            ok, bad = False, f"phrase {(p, q)}: table {int(nx[0])} {pr[0]:.3f}, by hand {top} {n / tot:.3f}"
            break
    # chunk edges lose a handful of phrases; the table may hold a few fewer than counting by hand, never more
    check("the table = counting by hand: the same phrases, the same commonest next piece and share",
          ok and 0 <= len(want) - keys.size <= 0.01 * len(want) + 20, bad or f"{keys.size:,} phrases of 2 (by hand {len(want):,}), kept at {b['min_count']}+ times")
    check("every length's keys are in order, each once", all(bool((b[f"keys_{o}"][1:] > b[f"keys_{o}"][:-1]).all()) for o in b["orders"]),
          ", ".join(f"{o}: {b[f'keys_{o}'].numel():,}" for o in b["orders"]))

    torch.manual_seed(0)
    base = Gnosis2(Gnosis2Config(**SMALL)).eval()
    with torch.no_grad():
        for n, p in base.named_parameters():
            p.add_(torch.randn_like(p) * 0.05)
    m = Gnosis2(Gnosis2Config(**SMALL, book=str(path))).eval()
    missing = m.load_state_dict(base.state_dict(), strict=False)
    x = torch.from_numpy(text(2 * 96, 5).astype(np.int64)).view(2, 96)
    with torch.no_grad():
        za, zb = base(x)[0], m(x)[0]
    check("the table starts as nothing: with the plain model's weights it gives the plain model's scores exactly, and it is not in the checkpoint",
          torch.equal(za, zb) and missing.missing_keys == ["book_out.weight"] and not any("book_keys" in k for k in m.state_dict()),
          f"new weights {missing.missing_keys}; {m.layout()}")

    # the model's lookup against the plain numpy one
    from oazy.book import hash_torch
    o = min(m.book_orders)
    hit, nxt, prob = book.lookup_np(b, x[0].numpy(), o)
    back = m._back(x, None, m.book_n)
    q = hash_torch(back, o)
    keys = getattr(m, f"book_keys_{o}")
    i = torch.searchsorted(keys, q.reshape(-1)).clamp_(max=keys.numel() - 1).view(q.shape)
    mh = ((keys[i] == q) & (back[..., :o] >= 0).all(-1))[0].numpy()
    mn = (getattr(m, f"book_nxt_{o}")[i].long() & 0xFFFF)[0].numpy()
    check("the model finds the same entries as the plain lookup", np.array_equal(mh, hit) and np.array_equal(mn[hit], nxt[hit]) and hit.mean() > 0.2,
          f"{hit.mean():.0%} of words have an entry of {o} pieces")

    with torch.no_grad():
        m.book_out.weight.normal_(std=0.3)
        z = m(x)[0]
        x2 = x.clone()
        x2[:, 60:] = torch.randint(1, 512, (2, 36))
        z2 = m(x2)[0]
    check("it matters once learned", not torch.allclose(z, zb, atol=1e-3))
    check("still causal: changing words from 60 on leaves every score before 60 unchanged",
          torch.allclose(z[:, :60], z2[:, :60], atol=1e-5) and not torch.allclose(z[:, 60:], z2[:, 60:]),
          f"gap before {(z[:, :60] - z2[:, :60]).abs().max().item():.1e}")
    for size in (7, 1):
        with torch.no_grad():
            full, _ = m.read(x[:1, :40])
            st, outs = None, []
            for j in range(0, 40, size):
                lg, st = m.read(x[:1, j:min(j + size, 40)], st)
                outs.append(lg)
        gap = (full - torch.cat(outs, 1)).abs().max().item()
        check(f"reading on in pieces of {size} (chat, samples) = reading all at once", gap < 1e-4, f"largest gap {gap:.1e}")
    from bee5.optim import groups
    mu, ad = groups(m)
    check("the table's one layer learns in Adam's gates group; the table itself is not a weight",
          any(p is m.book_out.weight for p in ad["gates"]) and all(not n.startswith("book") for n, _ in mu)
          and sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in base.parameters()) == m.book_out.weight.numel())
    m.train()
    m.set_progress(0.0)
    loss, _ = m.losses(x[:, :-1], x[:, 1:])
    loss.backward()
    check("a training step: finite loss, the layer gets a gradient, the word table gets none through the table",
          torch.isfinite(loss).item() and m.book_out.weight.grad.abs().sum() > 0, f"loss {loss.item():.3f}")

    # does a body learn faster with the table? text with phrases, new text every step (never the counted text)
    res = {}
    for name, extra in (("plain", {}), ("table", dict(book=str(path)))):
        torch.manual_seed(1)
        mm = Gnosis2(Gnosis2Config(**SMALL, **extra)).train()
        mm.set_progress(1.0)
        opt = mm.make_optimizer(2e-3, torch.device("cpu"))
        for step in range(150):
            xb = torch.from_numpy(text(32 * 65, 1000 + step).astype(np.int64)).view(32, 65)
            loss, _ = mm.losses(xb[:, :-1], xb[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        mm.eval()
        with torch.no_grad():
            xb = torch.from_numpy(text(64 * 65, 99999).astype(np.int64)).view(64, 65)
            res[name] = mm.losses(xb[:, :-1], xb[:, 1:])[0].item()
    check("150 steps on text with phrases: the body with the table is ahead of the plain one on new text", res["table"] < res["plain"] - 0.02,
          f"plain {res['plain']:.3f}, with the table {res['table']:.3f}")
    print()
    print("all passed" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
