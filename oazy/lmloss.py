"""The last step of a model, done in pieces: scores against the word table, the next-word loss, and the
mentor's lesson, without ever writing out all 49,152 scores of every token at once.

Why it matters. A micro-batch of 32 x 2,048 tokens has 65,536 tokens; their scores against a 49,152-word
table are 3.2 billion numbers: 6.4 GB in 32-bit, and the usual way (scores, then log-softmax, then the
loss, then the same again backwards, plus the mentor's scores next to them) writes and reads that pile
six or eight times per step. It is the single biggest piece of memory traffic in training a small model.

Here the tokens are taken a slice at a time (1,024 tokens: 200 MB of scores). For each slice we compute
the scores, the loss, and *also the gradient of the loss with respect to the scores* right away (for
the next-word loss it is just "probabilities minus the right answer"; for the mentor's lesson
"the student's probabilities minus the mentor's"), and push that gradient back onto the token
vectors and the word table at once. Nothing per-token is kept for the backward pass: it has already
been done. Memory for the head drops from gigabytes to a few hundred megabytes, and the traffic with it.

    loss, ce, kd = head_loss(h, W, targets, softcap, teacher_h, teacher_W, kd_weight)

  h          (N, D) the token vectors after the final norm (bf16 under autocast)
  W          (V, D) the word table (tied with the input embedding; its gradient comes back through here)
  targets    (N,)   the right next words; -1 = do not score this token (used by the chat fine-tune)
  softcap    the final scores are kept within +-softcap by softcap * tanh(score / softcap) (0 = off)
  teacher_h  (N, Dt) the mentor's token vectors, teacher_W (Vt, Dt) its word table, or None
  kd_weight  the mentor's share: loss = (1 - w) * next-word loss + w * KL(mentor || student)

The arithmetic is the same as F.cross_entropy + F.kl_div on the full scores (oazy/tests check this to
1e-4); only the order of operations and the memory differ. Same idea as Liger / "cut cross-entropy".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _autocast_on(device_type):
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:                                  # older torch: cuda only
        return device_type == "cuda" and torch.is_autocast_enabled()


class _HeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, targets, softcap, th, tW, kd_w, chunk):
        N, D = h.shape
        V = W.shape[0]
        dt = h.dtype
        if dt == torch.float32 and _autocast_on(h.device.type):   # training: these matmuls in bf16, like the rest of the model
            dt = torch.get_autocast_dtype(h.device.type) if hasattr(torch, "get_autocast_dtype") else torch.bfloat16
        with torch.autocast(device_type=h.device.type, enabled=False):
            Wc = W.to(dt)
            valid = targets >= 0
            n_valid = valid.sum().clamp_min(1).to(torch.float32)
            n_all = float(N)
            kd = th is not None and kd_w > 0
            if kd:
                tWc = tW.to(dt)
                Vt = tW.shape[0]
            gh = torch.empty_like(h)
            gW = torch.zeros(V, D, dtype=torch.float32, device=h.device)
            ce_sum = torch.zeros((), dtype=torch.float32, device=h.device)
            kd_sum = torch.zeros((), dtype=torch.float32, device=h.device)
            for i in range(0, N, chunk):
                hc = h[i:i + chunk].to(dt)
                tc = targets[i:i + chunk]
                vc = valid[i:i + chunk]
                z = (hc @ Wc.t()).float()                                   # (n, V) scores
                if softcap:
                    t = torch.tanh(z / softcap)
                    z = softcap * t
                lp = torch.log_softmax(z, dim=-1)
                # --- next-word loss and its gradient: probabilities minus the right answer
                tgt = tc.clamp_min(0)
                ce_sum += (-(lp.gather(1, tgt[:, None])[:, 0]) * vc).sum()
                g = lp.exp()                                                # p
                g.scatter_add_(1, tgt[:, None], -vc.to(torch.float32)[:, None])
                g *= (vc.to(torch.float32) * ((1.0 - kd_w) / n_valid))[:, None]
                # --- the mentor's lesson: the student's probabilities minus the mentor's
                if kd:
                    tz = (th[i:i + chunk].to(dt) @ tWc.t()).float()         # (n, Vt) mentor scores
                    tlp = torch.log_softmax(tz, dim=-1)
                    pt = tlp.exp()
                    if Vt == V:
                        kd_sum += (pt * (tlp - lp)).sum()
                        g += (lp.exp() - pt) * (kd_w / n_all)
                    else:                               # a GPT-2 mentor (50,257) and a padded student table (50,304)
                        lps = torch.log_softmax(z[:, :Vt], dim=-1)
                        kd_sum += (pt * (tlp - lps)).sum()
                        g[:, :Vt] += (lps.exp() - pt) * (kd_w / n_all)
                    del tz, tlp, pt
                if softcap:
                    g *= 1.0 - t * t                                        # through the soft cap
                gb = g.to(dt)
                gh[i:i + chunk] = (gb @ Wc).to(h.dtype)                     # d loss / d h
                gW += (gb.t() @ hc).float()                                 # d loss / d W
                del z, lp, g, gb
        loss = (1.0 - kd_w) * ce_sum / n_valid + kd_w * kd_sum / n_all
        ctx.save_for_backward(gh, gW)
        ctx.w_dtype = W.dtype
        ce, kdm = ce_sum / n_valid, kd_sum / n_all
        ctx.mark_non_differentiable(ce, kdm)
        return loss, ce, kdm

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_kd):
        gh, gW = ctx.saved_tensors
        return gh * g_loss.to(gh.dtype), (gW * g_loss).to(ctx.w_dtype), None, None, None, None, None, None


def head_loss(h, W, targets, softcap=0.0, teacher_h=None, teacher_W=None, kd_weight=0.0, chunk=1024):
    """See the module docstring. h: (..., D), targets: (...). Returns (loss, next-word loss, mentor KL), the
    last two detached, all as 0-d tensors. The loss is the mean over scored tokens (targets >= 0); the mentor
    term the mean over all tokens, as ember/mentor.mentor_loss does."""
    D = h.shape[-1]
    h2 = h.reshape(-1, D)
    t2 = targets.reshape(-1)
    th2 = teacher_h.reshape(-1, teacher_h.shape[-1]) if teacher_h is not None else None
    if th2 is not None and th2.shape[0] != h2.shape[0]:
        raise ValueError(f"mentor read {th2.shape[0]} tokens, student {h2.shape[0]}")
    if th2 is None or kd_weight <= 0:
        th2, teacher_W, kd_weight = None, None, 0.0
    elif teacher_W.shape[0] > W.shape[0]:
        raise ValueError(f"the mentor knows {teacher_W.shape[0]} pieces, the student only {W.shape[0]}: they must cut text the same way")
    return _HeadLoss.apply(h2, W, t2, float(softcap or 0.0), th2, teacher_W, float(kd_weight), int(chunk))


def plain_loss(h, W, targets, softcap=0.0, teacher_h=None, teacher_W=None, kd_weight=0.0):
    """The same numbers the slow way (full scores in 32-bit), for the tests and for --loss plain."""
    logits = F.linear(h, W.to(h.dtype)).float()
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
    if teacher_h is None or kd_weight <= 0:
        return ce, ce.detach(), torch.zeros((), device=h.device)
    tl = F.linear(teacher_h, teacher_W.to(teacher_h.dtype)).float()
    Vt = tl.shape[-1]
    s = F.log_softmax(logits[..., :Vt], dim=-1)
    t = F.log_softmax(tl, dim=-1)
    kd = F.kl_div(s.reshape(-1, Vt), t.reshape(-1, Vt), log_target=True, reduction="sum") / targets.numel()
    return (1 - kd_weight) * ce + kd_weight * kd, ce.detach(), kd.detach()
