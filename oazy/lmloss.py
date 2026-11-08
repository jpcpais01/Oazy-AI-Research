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

Choosing what to learn from (select=, oazy/select.py; off unless the run asks for it). Each word's own loss
is compared with a reference (what the mentor loses on that word), and the words still worth learning count
fully while the others count a little (their weight is the floor). The weights are computed inside the same
slices from numbers that are already there, so it costs almost nothing. The per-word losses come back too,
so the trainer can score whole passages for rereading.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _autocast_on(device_type):
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:                                  # older torch: cuda only
        return device_type == "cuda" and torch.is_autocast_enabled()


def _slice(hc, Wc, tc, vc, thc, tWc, softcap: float, kd_w, n_valid, n_all: float, Vt: int,
           rc=None, thr=None, floor=None, wscale=None):
    """One slice of tokens: scores, loss terms, and the gradient pushed straight back onto the token vectors
    and the word table. Pure tensor work, so torch.compile fuses the fp32 elementwise chain into a few kernels.
    rc (n,): the reference loss per word (oazy/select.py), or None = every word counts the same. With it, a word
    whose excess (own loss - reference; the mentor's own loss on it where the mentor reads this pass) is above
    thr counts 1, the others floor, all times wscale (so the weights average about 1)."""
    dt = hc.dtype
    z = (hc @ Wc.t()).float()                                       # (n, V) scores
    if softcap:
        t = torch.tanh(z / softcap)
        z = softcap * t
    lp = torch.log_softmax(z, dim=-1)
    # --- next-word loss and its gradient: probabilities minus the right answer
    tgt = tc.clamp_min(0)
    vcf = vc.to(torch.float32)
    tok = -(lp.gather(1, tgt[:, None])[:, 0])                       # (n,) each word's own loss
    ce = (tok * vcf).sum()
    tlp = None
    if thc is not None:
        tz = (thc @ tWc.t()).float()                                # (n, Vt) mentor scores
        tlp = torch.log_softmax(tz, dim=-1)
        ttok = -(tlp.gather(1, tgt.clamp_max(Vt - 1)[:, None])[:, 0])   # (n,) the mentor's own loss on each word
    else:
        ttok = torch.zeros_like(tok)
    if rc is not None:                                              # choosing what to learn from
        ref = ttok if thc is not None else rc
        w = torch.where(tok - ref > thr, torch.ones_like(tok), floor.expand_as(tok)) * wscale * vcf
    else:
        w = vcf
    wce = (tok * w).sum()
    g = lp.exp()                                                    # p
    g.scatter_add_(1, tgt[:, None], -vcf[:, None])
    g *= (w * ((1.0 - kd_w) / n_valid))[:, None]
    # --- the mentor's lesson: the student's probabilities minus the mentor's
    if tlp is not None:
        pt = tlp.exp()
        V = z.shape[1]
        if Vt == V:
            kd = (pt * (tlp - lp)).sum()
            g += (lp.exp() - pt) * (kd_w / n_all)
        else:                                   # a GPT-2 mentor (50,257) and a padded student table (50,304)
            lps = torch.log_softmax(z[:, :Vt], dim=-1)
            kd = (pt * (tlp - lps)).sum()
            g[:, :Vt] += (lps.exp() - pt) * (kd_w / n_all)
    else:
        kd = torch.zeros((), dtype=torch.float32, device=hc.device)
    if softcap:
        g *= 1.0 - t * t                                            # through the soft cap
    gb = g.to(dt)
    # d loss / d h (slice), d loss / d W (whole table); the words' own losses and the mentor's
    return ce, wce, kd, gb @ Wc, (gb.t() @ hc).float(), tok, ttok


_compiled_slice = None


def _no_compile():
    global _compiled_slice
    _compiled_slice = _slice


def slice_fn(device):
    """The slice, compiled on the GPU (OAZY_LOSS_COMPILE=0 turns that off), plain elsewhere."""
    global _compiled_slice
    if device.type != "cuda" or os.environ.get("OAZY_LOSS_COMPILE", "1") == "0":
        return _slice
    if _compiled_slice is None:
        _compiled_slice = torch.compile(_slice, dynamic=False)
    return _compiled_slice


class _HeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, targets, softcap, th, tW, kd_w, chunk, ref=None, thr=None, floor=None, wscale=None):
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
            tWc, Vt = (tW.to(dt), int(tW.shape[0])) if kd else (None, V)
            f = slice_fn(h.device)
            kw = torch.tensor(float(kd_w), dtype=torch.float32, device=h.device)   # a tensor: the mentor's share changes every step
            gh = torch.empty_like(h)
            gW = torch.zeros(V, D, dtype=torch.float32, device=h.device)
            ce_sum = torch.zeros((), dtype=torch.float32, device=h.device)
            wce_sum = torch.zeros((), dtype=torch.float32, device=h.device)
            kd_sum = torch.zeros((), dtype=torch.float32, device=h.device)
            tok_all = torch.empty(N, dtype=torch.float32, device=h.device)
            ttok_all = torch.zeros(N, dtype=torch.float32, device=h.device)
            for i in range(0, N, chunk):
                args = (h[i:i + chunk].to(dt), Wc, targets[i:i + chunk], valid[i:i + chunk],
                        th[i:i + chunk].to(dt) if kd else None, tWc, float(softcap), kw, n_valid, n_all, Vt,
                        ref[i:i + chunk] if ref is not None else None, thr, floor, wscale)
                try:
                    ce, wce, kdv, ghc, gWc, tk, ttk = f(*args)
                except Exception as e:
                    if f is _slice:
                        raise
                    print(f"[lmloss] torch.compile failed ({type(e).__name__}: {str(e)[:160]}); the loss runs uncompiled", flush=True)
                    f = _slice
                    _no_compile()
                    ce, wce, kdv, ghc, gWc, tk, ttk = f(*args)
                ce_sum += ce
                wce_sum += wce
                kd_sum += kdv
                tok_all[i:i + chunk] = tk
                if kd:
                    ttok_all[i:i + chunk] = ttk
                gh[i:i + chunk] = ghc.to(h.dtype)
                gW += gWc
        loss = (1.0 - kd_w) * wce_sum / n_valid + kd_w * kd_sum / n_all     # (wce = ce when every word counts the same)
        ctx.save_for_backward(gh, gW)
        ctx.w_dtype = W.dtype
        ce, kdm = ce_sum / n_valid, kd_sum / n_all
        ctx.mark_non_differentiable(ce, kdm, tok_all, ttok_all)
        return loss, ce, kdm, tok_all, ttok_all

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_kd, g_tok, g_ttok):
        gh, gW = ctx.saved_tensors
        return (gh * g_loss.to(gh.dtype), (gW * g_loss).to(ctx.w_dtype)) + (None,) * 10


def head_loss(h, W, targets, softcap=0.0, teacher_h=None, teacher_W=None, kd_weight=0.0, chunk=1024, select=None):
    """See the module docstring. h: (..., D), targets: (...). Returns (loss, next-word loss, mentor KL), the
    last two detached, all as 0-d tensors. The loss is the mean over scored tokens (targets >= 0); the mentor
    term the mean over all tokens, as ember/mentor.mentor_loss does. select: an oazy.select.Select (choosing
    what to learn from) or None; the next-word loss returned is always the plain, unweighted one."""
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
    if select is None:
        loss, ce, kd, _, _ = _HeadLoss.apply(h2, W, t2, float(softcap or 0.0), th2, teacher_W, float(kd_weight), int(chunk))
        return loss, ce, kd
    ref, thr, floor, wscale = select.before(t2, exact=th2 is not None)
    loss, ce, kd, tok_, ttok = _HeadLoss.apply(h2, W, t2, float(softcap or 0.0), th2, teacher_W, float(kd_weight), int(chunk),
                                               ref, thr, floor, wscale)
    select.after(t2, tok_, ttok if th2 is not None else None, ref)
    return loss, ce, kd


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
