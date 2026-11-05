"""Is the memory really used? Checks that work the same way for Engram and the GPT.

context curve   the loss when the model may only use the last 4, 32, 256 or all 1,024 words. A model
                that ignores its memory has a flat curve; the drop from 4 to 1,024 is what memory buys.
by position     the loss of word 1-4, 4-16, ..., 256-1024 of a text: later words are easier only if
                what was read earlier is being remembered.
recall          a passage of 128 tokens, then the same passage again: how well is the repeat
                predicted? (a transformer can look back and copy; Engram must have stored it)
snapshot        (Engram only) one probe text read with the lid off: each word's surprise, trust
                and fade in every block, each memory's forgetting speed, and pictures of memories.
"""
from __future__ import annotations

import base64
import math

import numpy as np
import torch
import torch.nn.functional as F

SPANS = [4, 32, 256]
POS_BUCKETS = [(0, 4), (4, 16), (16, 64), (64, 256), (256, 1024)]
PROBE_TEXT = (
    "Marie Curie was born in Warsaw in 1867. She moved to Paris to study physics and mathematics, "
    "and there she met Pierre Curie. Together they discovered two new elements, which they named "
    "polonium and radium. In 1903 she became the first woman to win a Nobel Prize, and in 1911 she "
    "won a second one, in chemistry. Today, students still learn that Marie Curie, born in Warsaw "
    "in 1867, discovered polonium and radium with Pierre Curie."
)


def _amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def token_losses(model, x, y, span=None, batch=16):
    """Per-token loss (N, T). span: cut every text into pieces of `span` tokens, read separately."""
    N, T = x.shape
    if span and span < T:
        x, y = x.reshape(-1, span), y.reshape(-1, span)
        batch = batch * (T // span)
    out = []
    for i in range(0, x.shape[0], batch):
        with _amp(x.device):
            logits = model.score(x[i:i + batch], y[i:i + batch]) if hasattr(model, "score") else model._body(x[i:i + batch])
        out.append(F.cross_entropy(logits.float().transpose(1, 2), y[i:i + batch], reduction="none"))
        del logits
    return torch.cat(out).reshape(N, T)


@torch.no_grad()
def memory_report(model, windows: np.ndarray, device, batch=16):
    was = model.training
    model.eval()
    xy = torch.from_numpy(windows).to(device)
    x, y = xy[:, :-1], xy[:, 1:]
    T = x.shape[1]
    full = token_losses(model, x, y, batch=batch)
    rep = {"loss": float(full.mean())}
    ctx = []
    for s in SPANS:
        if s < T:
            ctx.append([s, float(token_losses(model, x, y, s, batch).mean())])
    ctx.append([T, rep["loss"]])
    rep["context"] = ctx
    per_pos = full.mean(0)
    rep["by_pos"] = [[a, min(b, T), float(per_pos[a:min(b, T)].mean())] for a, b in POS_BUCKETS if a < T]
    # recall: a passage, then the same passage again
    L = min(128, T // 2)
    rx = torch.cat([x[:, :L], x[:, :L]], 1)
    first = second = acc = 0.0
    for i in range(0, len(rx), batch):
        r = rx[i:i + batch]
        with _amp(device):
            logits = (model.score(r[:, :-1], r[:, 1:]) if hasattr(model, "score") else model._body(r[:, :-1])).float()
        tgt = r[:, 1:]
        lo = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        first += float(lo[:, :L - 1].mean(1).sum())
        second += float(lo[:, L - 1:].mean(1).sum())            # predictions of the repeated passage
        acc += float((logits[:, L - 1:].argmax(-1) == tgt[:, L - 1:]).float().mean(1).sum())
        del logits, lo
    n = len(rx)
    rep["first_loss"], rep["recall_loss"], rep["recall_acc"] = first / n, second / n, acc / n
    if was:
        model.train()
    return rep


def _img(m: torch.Tensor):
    """A matrix as base64 int8 bytes, scaled so the largest value is 127. Returns (b64, scale)."""
    m = m.float()
    s = float(m.abs().max()) or 1.0
    q = (m / s * 127).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
    return base64.b64encode(q.tobytes()).decode(), s


def token_strings(enc, ids):
    out = []
    for t in ids:
        try:
            out.append(enc.decode_single_token_bytes(t).decode("utf-8", errors="replace"))
        except Exception:
            out.append("?")
    return out


@torch.no_grad()
def snapshot(model, enc, text=PROBE_TEXT):
    """Read the probe text with the lid off (Engram only)."""
    was = model.training
    model.eval()
    dev = model.embed.weight.device
    ids = enc.encode_ordinary(text)
    x = torch.tensor([ids], device=dev)
    logits, states, infos = model.inspect(x)
    lp = F.log_softmax(logits[0].float(), -1)
    nll = [None] + [float(-lp[i, ids[i + 1]]) for i in range(len(ids) - 1)]
    L = len(infos)
    pick = sorted({0, L // 2, L - 1})
    snap = {
        "pieces": token_strings(enc, ids),
        "word_loss": nll,                                                       # how unexpected each word was
        "surprise": [inf["surprise"][0].mean(-1).tolist() for inf in infos],    # (L, T) memory surprise
        "trust": [inf["trust"][0].mean(-1).tolist() for inf in infos],
        "fade": [inf["fade"][0].mean(-1).tolist() for inf in infos],
        # each memory's typical "half-life" in words on this text: how long until half is faded
        "half_life": [[float(math.log(0.5) / min(-1e-6, math.log(max(1e-6, f))))
                       for f in inf["fade"][0].float().mean(0).tolist()] for inf in infos],
        "memories": [],
    }
    for l in pick:
        S = states[l][1][0]                                  # (H, K, V)
        b64, s = _img(S[0])
        snap["memories"].append({"layer": l, "head": 0, "k": S.shape[1], "v": S.shape[2], "data": b64, "scale": s})
    if was:
        model.train()
    return snap
