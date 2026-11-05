"""The one rule: a memory that is corrected by its own surprise, word after word.

Each memory is a small grid of numbers S (keys x values, e.g. 64 x 64). For every word t:

    S      <- a_t * S                      fade: old, unimportant things blur a little
    guess  =  k_t . S                      ask: "what goes with this situation?"
    surprise = v_t - guess                 what the memory got wrong
    S      <- S + b_t * k_t (x) surprise   learn: fix the memory by the size of the surprise
    answer =  q_t . S                      recall from the updated memory

a_t (fade, 0..1) and b_t (trust, 0..1) are chosen per word by small learned layers.
In the research literature this is the "gated delta rule" (Widrow-Hoff learning, used as a
fast-weight memory: DeltaNet, Gated DeltaNet). Written as a loop it is slow on a GPU, so this file
has three versions of the same thing:

  recurrent()  the loop above, word by word. Obvious, slow; used for chat and as the reference.
  chunked()    the same maths, but 64 words at a time with matrix products (pure PyTorch, runs anywhere).
  fla          the flash-linear-attention Triton kernel for the chunked form (fastest; GPU only, optional).

All three give the same answer (memtrix/tests.py checks this). Shapes, everywhere:
    q, k: (B, T, H, K)   v: (B, T, H, V)   log_fade: (B, T, H) (= log a_t <= 0)   trust: (B, T, H)
    state S: (B, H, K, V)
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_FLA = None
_FLA_OK = None          # None = not checked yet; True/False after the first check on this machine


def _load_fla():
    global _FLA
    if _FLA is None:
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            _FLA = chunk_gated_delta_rule
        except Exception:
            _FLA = False
    return _FLA


def recurrent(q, k, v, log_fade, trust, state=None, scale=None, return_surprise=False):
    """Word by word. Returns (answers, final state[, surprise]); surprise = v - guess, before the fix."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    dt = q.dtype
    q, k, v, log_fade, trust = (x.float() for x in (q, k, v, log_fade, trust))
    S = torch.zeros(B, H, K, V, device=q.device) if state is None else state.float()
    outs, surs = [], []
    for t in range(T):
        S = S * log_fade[:, t, :, None, None].exp()                       # fade
        guess = torch.einsum("bhk,bhkv->bhv", k[:, t], S)                 # ask
        sur = v[:, t] - guess                                             # surprise
        S = S + trust[:, t, :, None, None] * k[:, t, :, :, None] * sur[:, :, None, :]   # learn
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))  # recall
        if return_surprise:
            surs.append(sur)
    o = torch.stack(outs, 1).to(dt)
    if return_surprise:
        return o, S, torch.stack(surs, 1)
    return o, S


def chunked(q, k, v, log_fade, trust, state=None, scale=None, chunk=64, return_surprise=False):
    """Exactly the same rule as recurrent(), computed 64 words at a time.

    Inside a chunk, every word's correction depends on the corrections of the earlier words in the
    same chunk. That chain is a small lower-triangular system (I + A) U = rhs, solved in one go;
    then the chunk's answers and the memory at the chunk's end are plain matrix products.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk
    scale = K ** -0.5 if scale is None else scale
    dt = q.dtype
    pad = (-T) % C
    if pad:  # padded words have trust 0 and no fade: they change nothing
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        log_fade, trust = (F.pad(x, (0, 0, 0, pad)) for x in (log_fade, trust))
    N = (T + pad) // C
    # (B, H, N, C, d), float32 for the triangular solve
    q, k, v = (x.float().reshape(B, N, C, H, -1).permute(0, 3, 1, 2, 4) for x in (q, k, v))
    g, b = (x.float().reshape(B, N, C, H).permute(0, 3, 1, 2) for x in (log_fade, trust))
    q = q * scale
    G = g.cumsum(-1)                                                 # total fade since the chunk began
    tri = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    decay = (G[..., :, None] - G[..., None, :]).masked_fill(~tri, float("-inf")).exp()   # fade from word j to i (i >= j)
    kb = k * b[..., None]
    A = (kb @ k.transpose(-1, -2)) * decay
    A = A.tril(-1)                                                   # strictly lower part
    eye = torch.eye(C, device=q.device).expand_as(A)
    Tinv = torch.linalg.solve_triangular(eye + A, eye, upper=False)  # (I + A)^-1
    W = Tinv @ (kb * G[..., None].exp())                             # how the incoming memory shifts each correction
    U = Tinv @ (v * b[..., None])                                    # corrections if the incoming memory were blank
    qk = (q @ k.transpose(-1, -2)) * decay                           # recall weights inside the chunk
    S = torch.zeros(B, H, K, V, device=q.device) if state is None else state.float()
    outs, surs = [], []
    for n in range(N):
        u = U[:, :, n] - W[:, :, n] @ S                              # trust * surprise for every word in the chunk
        outs.append((q[:, :, n] * G[:, :, n, :, None].exp()) @ S + qk[:, :, n] @ u)
        if return_surprise:
            surs.append(u / b[:, :, n, :, None].clamp_min(1e-6))
        gl = G[:, :, n, -1]
        S = S * gl[..., None, None].exp() + (k[:, :, n] * (gl[..., None] - G[:, :, n]).exp()[..., None]).transpose(-1, -2) @ u
    o = torch.stack(outs, 2).reshape(B, H, N * C, V).transpose(1, 2)[:, :T].to(dt)
    if return_surprise:
        sur = torch.stack(surs, 2).reshape(B, H, N * C, V).transpose(1, 2)[:, :T]
        return o, S, sur
    return o, S


@torch.compiler.disable
def _fla_call(q, k, v, log_fade, trust, state, scale):
    o, S = _load_fla()(q=q, k=k, v=v, g=log_fade.float(), beta=trust, scale=scale,
                       initial_state=state, output_final_state=True)
    return o, S


@torch.compiler.disable
def fla_check(device) -> bool:
    """Run the fla kernel once against the reference; use it only if it agrees (checked once per process)."""
    global _FLA_OK
    if _FLA_OK is not None:
        return _FLA_OK
    _FLA_OK = False
    if (os.environ.get("OAZY_NO_FLA") or os.environ.get("ENGRAM_NO_FLA")) or device.type != "cuda" or not _load_fla():
        return False
    try:
        g = torch.Generator(device=device).manual_seed(0)
        B, T, H, K = 2, 200, 3, 64
        q = F.normalize(torch.randn(B, T, H, K, device=device, generator=g), dim=-1)
        k = F.normalize(torch.randn(B, T, H, K, device=device, generator=g), dim=-1)
        v = torch.randn(B, T, H, K, device=device, generator=g)
        lf = -torch.rand(B, T, H, device=device, generator=g) * 0.2
        tr = torch.rand(B, T, H, device=device, generator=g)
        ref, Sref = chunked(q, k, v, lf, tr)
        o, S = _fla_call(*(x.bfloat16() for x in (q, k, v)), lf, tr.bfloat16(), None, K ** -0.5)
        err = (o.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        errS = (S.float() - Sref).abs().max().item() / (Sref.abs().max().item() + 1e-6)
        _FLA_OK = err < 0.05 and errS < 0.05
        print(f"[memtrix] flash-linear-attention kernel {'OK' if _FLA_OK else 'DISAGREES, not used'} "
              f"(rel. error {err:.4f} / state {errS:.4f})", flush=True)
    except Exception as e:  # missing Triton support for this GPU, API change, ...
        print(f"[memtrix] flash-linear-attention kernel unavailable ({type(e).__name__}: {str(e)[:200]}); "
              "using the pure-PyTorch chunked version", flush=True)
        _FLA_OK = False
    return _FLA_OK


def delta_rule(q, k, v, log_fade, trust, state=None, scale=None):
    """Pick the fastest version that is correct on this machine. Returns (answers, final state)."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if q.is_cuda and fla_check(q.device):
        return _fla_call(q, k, v, log_fade, trust, state, scale)
    return chunked(q, k, v, log_fade, trust, state, scale)
