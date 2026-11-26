"""The creature's mind (the clean v2 recipe).

  eye        two stereo pictures -> 4096 numbers. Kept rich by VICReg: every number must vary
             across moments (std >= 1) and no two may move together. No picture is ever redrawn.
  memory     h: 512 numbers carried from moment to moment (a GRU), updated from the last
             moment and the muscles just used.
  moment     z: 32 slots, each picking 1 of 16 options. Before looking it is guessed from
             memory alone (prior); after looking (eye + body feelings + intensity + ears) it is
             corrected (posterior). Training pulls the guess toward the correction (the KL,
             "surprise"): a prediction of its whole next inner state, never of pixels.
  heads      read (h, z): its senses (body feelings, intensity, ears) and whether it is still
             alive. These keep the inner state about what matters to the body.
  actor      muscles from (h, z); learned only inside imagination (15 steps = 1.5 s ahead),
             by backpropagating imagined wellbeing through the learned dynamics.
  critic     how good an inner state looks beyond the imagined 1.5 s.
  mind's eye a small decoder that draws the left eye from a *detached* inner state, for the
             page only; nothing flows back into the creature.

The only drive is wellbeing = (energy + health) / 2, read from its own predicted senses,
and the end of life. Nothing points it at food, crates or anything else.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import ACT, SENSES

H_DIM, Z_SLOTS, Z_CLS = 512, 32, 16
Z_DIM = Z_SLOTS * Z_CLS
FEAT = H_DIM + Z_DIM
EMB = 4096


def mlp(i, o, hid=512, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, hid), nn.LayerNorm(hid), nn.SiLU()]
        d = hid
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Eye(nn.Module):
    """Stride-2 convolutions down to 4x4; channels 32, 64, 128, 256, (256...); output 4x4x256 = 4096."""

    def __init__(self, res):
        super().__init__()
        n = 0
        while res >> (n + 1) >= 4:
            n += 1
        plan = [32, 64, 128, 256] + [256] * 8
        layers, c = [], 6
        for i in range(n):
            o = plan[i] if i < n - 1 else 256
            layers.append(nn.Conv2d(c, o, 4, 2, 1))
            if i < n - 1:                      # the last layer stays linear: VICReg sets its scale
                layers += [nn.GroupNorm(1, o), nn.SiLU()]
            c = o
        self.net = nn.Sequential(*layers)

    def forward(self, img_u8):
        x = img_u8.float() / 255.0 - 0.5
        return self.net(x).flatten(1)


class MindsEye(nn.Module):
    """Draws the left eye (3 x res x res) from a detached inner state. Page only."""

    def __init__(self, res):
        super().__init__()
        self.res = res
        self.fc = nn.Linear(FEAT, 256 * 4 * 4)
        layers, c, s = [], 256, 4
        while s < res:
            o = max(32, c // 2)
            last = s * 2 >= res
            layers += [nn.ConvTranspose2d(c, 3 if last else o, 4, 2, 1)]
            if not last:
                layers += [nn.GroupNorm(1, o), nn.SiLU()]
            c, s = o, s * 2
        self.net = nn.Sequential(*layers)

    def forward(self, feat):
        return self.net(self.fc(feat).view(-1, 256, 4, 4)) + 0.5


class Mind(nn.Module):
    def __init__(self, res=64):
        super().__init__()
        self.res = res
        self.eye = Eye(res)
        self.eye_in = nn.Sequential(nn.Linear(EMB, 512), nn.LayerNorm(512), nn.SiLU())
        self.img_in = nn.Sequential(nn.Linear(Z_DIM + ACT, 512), nn.LayerNorm(512), nn.SiLU())
        self.gru = nn.GRUCell(512, H_DIM)
        self.prior = mlp(H_DIM, Z_DIM, n=1)
        self.post = mlp(H_DIM + 512 + SENSES, Z_DIM, n=1)
        self.sense_head = mlp(FEAT, SENSES)
        self.cont_head = mlp(FEAT, 1)
        self.actor = mlp(FEAT, 2 * ACT)
        self.critic = mlp(FEAT, 1)
        self.slow_critic = mlp(FEAT, 1)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.minds_eye = MindsEye(res)

    # --- the moment as 32 categorical slots
    @staticmethod
    def dist(logits):
        p = F.softmax(logits.view(*logits.shape[:-1], Z_SLOTS, Z_CLS).float(), -1)
        return 0.99 * p + 0.01 / Z_CLS                                   # 1% uniform mix

    @staticmethod
    def sample(probs):
        idx = torch.multinomial(probs.reshape(-1, Z_CLS), 1).view(probs.shape[:-1])
        hard = F.one_hot(idx, Z_CLS).to(probs.dtype)
        return (hard + probs - probs.detach()).flatten(-2)              # straight-through

    def initial(self, n, dev):
        return torch.zeros(n, H_DIM, device=dev), torch.zeros(n, Z_DIM, device=dev)

    def step_h(self, h, z, a):
        return self.gru(self.img_in(torch.cat([z, a], -1)).float(), h.float())

    def observe(self, h, emb, senses):
        logits = self.post(torch.cat([h, self.eye_in(emb), senses], -1))
        return self.dist(logits)

    def imagine_z(self, h):
        return self.dist(self.prior(h))

    def act(self, feat, deterministic=False):
        out = self.actor(feat).float()
        mean, std = out.chunk(2, -1)
        std = 2 * torch.sigmoid(std / 2) + 0.05
        if deterministic:
            return torch.tanh(mean), None
        eps = torch.randn_like(mean)
        pre = mean + std * eps
        ent = (torch.log(std) + 0.5 * (1 + 1.8378770664093453)).sum(-1)  # Gaussian entropy (pre-tanh)
        return torch.tanh(pre), ent


def kl_cat(p, q):
    """KL(p || q) for (..., 32, 16) probabilities, summed over slots."""
    return (p * (torch.log(p) - torch.log(q))).sum((-1, -2))


def wellbeing(senses):
    """The built-in drive: (energy + health) / 2 from (predicted) body feelings."""
    return 0.5 * (senses[..., 0].clamp(0, 1) + senses[..., 1].clamp(0, 1))


def vicreg(e):
    """Variance (std >= 1 per number) + covariance (numbers must not move together)."""
    e = e.float()
    e = e - e.mean(0)
    std = torch.sqrt(e.var(0) + 1e-4)
    var = F.relu(1 - std).mean()
    n, d = e.shape
    cov = (e.T @ e) / (n - 1)
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return var, off / d, std.mean()


def eff_rank(e):
    e = e.float() - e.float().mean(0)
    s = torch.linalg.svdvals(e)
    p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
