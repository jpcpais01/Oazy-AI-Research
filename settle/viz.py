"""Turning the sheet into pictures: colours for cells, and a 'probe' that records a settle in detail."""
from __future__ import annotations

import base64

import numpy as np
import torch

PROBE_TEXT = ("Once upon a time, there was a little girl named Lily. She loved to play outside "
              "in the park. One day, she saw a big red ball stuck in a tall tree. Lily was sad, "
              "so she asked her friend Tom for help.")


class Palette:
    """Colours a cell by projecting its vector on the sheet's 3 main directions (PCA) -> RGB."""

    def __init__(self, cells: torch.Tensor):
        x = cells.reshape(-1, cells.shape[-1]).float().cpu()
        self.mean = x.mean(0)
        xc = x - self.mean
        _, _, V = torch.linalg.svd(xc, full_matrices=False)
        comps = V[:3]
        proj = xc @ comps.T
        # fix the arbitrary sign of each direction so colours do not flip between snapshots
        sign = torch.sign((proj ** 3).mean(0))
        sign[sign == 0] = 1
        self.comps = comps * sign[:, None]
        proj = proj * sign
        self.scale = proj.std(0).clamp(min=1e-6) * 1.6

    def rgb(self, cells: torch.Tensor) -> np.ndarray:
        x = cells.float().cpu()
        shp = x.shape[:-1]
        p = ((x.reshape(-1, x.shape[-1]) - self.mean) @ self.comps.T) / self.scale
        c = 0.5 + 0.5 * torch.tanh(p)
        # gentle mixing so colours read as hues rather than raw channels
        M = torch.tensor([[0.85, 0.25, 0.10], [0.10, 0.80, 0.30], [0.30, 0.15, 0.90]])
        c = (c @ M.T).clamp(0, 1)
        return (c * 255).round().byte().numpy().reshape(*shp, 3)


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode()


def token_strings(enc, ids):
    out = []
    for i in ids:
        try:
            out.append(enc.decode([int(i)]))
        except Exception:
            out.append("?")
    return out


@torch.no_grad()
def probe(model, enc, text: str = PROBE_TEXT):
    """Run the probe text word by word (exactly like generation) and once in parallel (like training).

    Returns a JSON-able dict with the settled sheet as an image, steps per word,
    the parallel settle as a little movie, and how well the two agree.
    """
    was_training = model.training
    model.eval()
    compiled, model._compiled = model._compiled, None
    dev = model.embed.weight.device
    ids = enc.encode_ordinary(text)[: model.cfg.max_len]
    T, R = len(ids), model.cfg.rows

    # word by word (generation path)
    st = model.new_state(dev)
    seq_logits, lens_steps, revisions = [], [], []
    for t in ids:
        lg, info = model.feed(st, t)
        seq_logits.append(lg)
        lens_steps.append(info["lens_steps"])
        revisions.append((info["lens"] - info["anchor"]).pow(2).mean((1, 2)).sqrt().cpu())
    seq_logits = torch.stack(seq_logits)
    sheet = st["sheet"]                                        # (T,R,C)

    # all at once (training path)
    idx = torch.tensor([ids], device=dev)
    words = model._words(idx)
    z, stats = model.settle_sheet(words, record=True)
    gist = model._gist_parallel(z)
    L, _ = model.settle_lens(z, gist)
    par_logits = model._read(L[:, :, -1], idx)[0].float()

    pal = Palette(sheet)
    frames = np.stack([pal.rgb(f) for f in stats["frames"]])  # (K,T,R,3)
    curve = torch.stack(stats["curve"]).cpu().numpy()          # (K,T)
    probs = torch.softmax(seq_logits.float(), -1)
    nxt = torch.tensor(ids[1:] + [ids[-1]], device=dev)
    p_true = probs[torch.arange(T, device=dev), nxt].cpu().numpy()
    gist_rgb = Palette(gist[0][:, None, :]).rgb(gist[0][:, None, :])[:, 0]

    out = {
        "tokens": token_strings(enc, ids),
        "rows": R,
        "steps": st["steps"],
        "lens_steps": lens_steps,
        "sheet": b64(pal.rgb(sheet)),                         # (T,R,3) uint8, base64
        "frames": b64(frames),
        "n_frames": int(frames.shape[0]),
        "par_change": [[round(float(v), 5) for v in row] for row in curve],
        "revision": [[round(float(v), 4) for v in r] for r in revisions],
        "gist": b64(gist_rgb),
        "p_next": [round(float(v), 4) for v in p_true[:-1]],
        "gap": float((seq_logits - par_logits).abs().max()),
        "agree": float((seq_logits.argmax(-1) == par_logits.argmax(-1)).float().mean()),
    }
    model._compiled = compiled
    model.train(was_training)
    return out
