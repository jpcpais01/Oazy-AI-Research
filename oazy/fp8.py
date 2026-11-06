"""FP8 training for the big matrix products (Frog).

The B200 multiplies 8-bit floating-point numbers about twice as fast as 16-bit ones. torchao's float8 training
swaps each large nn.Linear for one that, on every pass, scales its input and weight into FP8 (one scale per
tensor), multiplies in FP8 and keeps the result and the master weights in 16/32 bits; the backward pass does the
same for its two products. The word table (the input embedding and the score head, which share weights) and the
small gate layers stay in bf16, so does everything that is not a matrix product.

    convert(model, device, say)   -> (model, used)   on a GPU that has FP8 (compute capability 8.9 or newer)

It is checked before it is kept: the same batch through the model before and after the swap must give nearly the
same loss (within 2%) and a finite gradient, else the model is handed back unchanged. Checkpoints stay ordinary
(the weights are plain tensors), so chat, evals and the fine-tune load them as usual.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


STACKS = ("blocks.", "enc.", "slow.", "dec.")     # the layer stacks (Code Black has three: fast encoder, slow, fast decoder)


def keep(mod: nn.Module, fqn: str) -> bool:
    """Which layers go to FP8: the large Linear layers of the stack (projections and MLPs). FP8's matrix
    products want sizes divisible by 16; a small layer (trust, fade: 1,024 -> 8) gains nothing."""
    return (isinstance(mod, nn.Linear) and fqn.startswith(STACKS) and mod.in_features % 16 == 0
            and mod.out_features % 16 == 0 and min(mod.in_features, mod.out_features) >= 256)


def available(device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 9)


def convert(model: nn.Module, device, say=print, vocab: int | None = None):
    """Swap the large Linear layers for FP8 ones if this GPU and torchao allow it and a check passes."""
    if not available(device):
        say("[fp8] this device has no FP8 matrix products: staying in bf16")
        return model, False
    try:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    except Exception as e:
        say(f"[fp8] torchao is not installed ({type(e).__name__}); staying in bf16. Install: python3 -m pip install torchao")
        return model, False
    before = copy.deepcopy(model)
    try:
        convert_to_float8_training(model, config=Float8LinearConfig(), module_filter_fn=keep)
        n = sum(1 for m in model.modules() if type(m).__name__ == "Float8Linear")
        rel, finite = check(before, model, device, vocab)
        if not (rel < 0.02 and finite):
            say(f"[fp8] check failed (loss differs by {rel:.2%}, gradient finite: {finite}); staying in bf16")
            return before, False
        say(f"[fp8] {n} layers multiply in FP8 (loss vs bf16 on a test batch: {rel:.2%} apart)")
        del before
        return model, True
    except Exception as e:
        say(f"[fp8] could not switch to FP8 ({type(e).__name__}: {str(e)[:160]}); staying in bf16")
        return before, False


def check(ref: nn.Module, m: nn.Module, device, vocab=None):
    """Loss of one random batch through both models (bf16 autocast), and whether m's gradient is finite."""
    V = vocab or ref.embed.weight.shape[0]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randint(1, V, (2, 512), device=device, generator=g)
    y = torch.roll(x, -1, 1)
    out = []
    for mod in (ref, m):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = mod.losses(x, y) if hasattr(mod, "losses") else (mod(x, y)[1], None)
        out.append(loss)
    out[1].backward()
    finite = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
    m.zero_grad(set_to_none=True)
    return abs(out[1].item() - out[0].item()) / max(1e-6, abs(out[0].item())), finite
