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
  intentions (optional, --intent K) a slow level. Every K steps a "will" picks an intention:
             8 slots x 8 options, decoded into a memory it wants to be in ~K steps later. The
             muscles see the intention; inside imagination they are paid wellbeing plus how close
             memory gets to the intended one, and the will is paid the wellbeing that follows its
             choice (its own critic looks past the dream). Intentions are not given: their code
             is learned by squeezing real memories through it (an autoencoder), so every
             intention is a state the creature has really been in.
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
G_SLOTS, G_CLS = 8, 8      # the first intention code (will_s1, willn_s1); new runs choose with --g-slots/--g-cls
G_DIM = G_SLOTS * G_CLS


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
        x = (img_u8.float() / 255.0 - 0.5).contiguous(memory_format=torch.channels_last)
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


def gdist(logits, cls=G_CLS):
    """Intention code: (..., slots, cls) probabilities with a 1% uniform mix."""
    p = F.softmax(logits.view(*logits.shape[:-1], logits.shape[-1] // cls, cls).float(), -1)
    return 0.99 * p + 0.01 / cls


def gsample(probs, deterministic=False):
    if deterministic:
        idx = probs.argmax(-1)
    else:
        u = torch.rand_like(probs).clamp_(1e-10, 1.0)
        idx = (torch.log(probs) - torch.log(-torch.log(u))).argmax(-1)
    hard = F.one_hot(idx, probs.shape[-1]).to(probs.dtype)
    return (hard + probs - probs.detach()).flatten(-2)


def maxcos(a, b):
    """Director's max-cosine: 1 only when a matches b in direction AND size."""
    m = torch.maximum(a.norm(dim=-1), b.norm(dim=-1)).clamp(min=1e-6)
    return (a * b).sum(-1) / (m * m)


class Mind(nn.Module):
    def __init__(self, res=64, intent=0, g_slots=G_SLOTS, g_cls=G_CLS, matter=False, jumps=""):
        super().__init__()
        self.res = res
        self.intent = int(intent or 0)
        self.g_cls, self.g_dim = int(g_cls), int(g_slots) * int(g_cls)
        gin = self.g_dim if self.intent else 0
        self.eye = Eye(res)
        self.eye_in = nn.Sequential(nn.Linear(EMB, 512), nn.LayerNorm(512), nn.SiLU())
        self.img_in = nn.Sequential(nn.Linear(Z_DIM + ACT, 512), nn.LayerNorm(512), nn.SiLU())
        self.gru = nn.GRUCell(512, H_DIM)
        self.prior = mlp(H_DIM, Z_DIM, n=1)
        self.post = mlp(H_DIM + 512 + SENSES, Z_DIM, n=1)
        self.sense_head = mlp(FEAT, SENSES)
        self.cont_head = mlp(FEAT, 1)
        self.actor = mlp(FEAT + gin, 2 * ACT)
        self.critic = mlp(FEAT + gin, 1)
        self.slow_critic = mlp(FEAT + gin, 1)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        if self.intent:
            self.goal_enc = mlp(H_DIM, self.g_dim)           # real memory -> intention code
            self.goal_dec = mlp(self.g_dim, H_DIM)           # intention code -> wanted memory
            self.will = mlp(FEAT, self.g_dim)               # picks the intention
            self.will_critic = mlp(FEAT, 1)
            self.will_slow = mlp(FEAT, 1)
            self.will_slow.load_state_dict(self.will_critic.state_dict())
            for p in self.will_slow.parameters():
                p.requires_grad_(False)
        self.minds_eye = MindsEye(res)
        # jumpy forecasts: from a memory h, where will the memory be k steps later, how much
        # wellbeing comes on the way (discounted sum / k) and will it still be alive. One head per
        # jump size; learned from what really happened (replay), used to look past the dream.
        self.jump_chain = [int(x) for x in str(jumps or "").split(",") if x.strip()]
        self.jump_sizes = sorted(set(self.jump_chain))
        self.jump = nn.ModuleDict({str(k): mlp(H_DIM, H_DIM + 2) for k in self.jump_sizes})
        if matter:      # what in this picture matters to the body: will I eat / get hurt within 1 s?
            self.matter = nn.Sequential(nn.Linear(EMB, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 2))

    # --- the moment as 32 categorical slots
    @staticmethod
    def dist(logits):
        p = F.softmax(logits.view(*logits.shape[:-1], Z_SLOTS, Z_CLS).float(), -1)
        return 0.99 * p + 0.01 / Z_CLS                                   # 1% uniform mix

    @staticmethod
    def sample(probs):
        # Gumbel-max: the same draw as multinomial, but a few fused element-wise ops (compiles well)
        u = torch.rand_like(probs).clamp_(1e-10, 1.0)
        idx = (torch.log(probs) - torch.log(-torch.log(u))).argmax(-1)
        hard = F.one_hot(idx, Z_CLS).to(probs.dtype)
        return (hard + probs - probs.detach()).flatten(-2)              # straight-through

    def jump_step(self, h, k):
        """Jump k steps ahead from memory h: (next memory, mean discounted wellbeing on the way, alive logit)."""
        out = self.jump[str(k)](h).float()
        return torch.tanh(h.float() + out[:, :H_DIM]), out[:, H_DIM], out[:, H_DIM + 1]

    def initial(self, n, dev):
        return torch.zeros(n, H_DIM, device=dev), torch.zeros(n, Z_DIM, device=dev)

    def step_h(self, h, z, a):
        return self.gru(self.img_in(torch.cat([z, a], -1)).float(), h.float())

    def observe(self, h, emb, senses):
        logits = self.post(torch.cat([h, self.eye_in(emb), senses], -1))
        return self.dist(logits)

    def imagine_z(self, h):
        return self.dist(self.prior(h))

    ent_squash = False

    def act(self, feat, deterministic=False, per=False):
        out = self.actor(feat).float()
        mean, std = out.chunk(2, -1)
        std = 2 * torch.sigmoid(std / 2) + 0.05
        if deterministic:
            return torch.tanh(mean), None
        eps = torch.randn_like(mean)
        pre = mean + std * eps
        ent = torch.log(std) + 0.5 * (1 + 1.8378770664093453)            # Gaussian entropy (pre-tanh), per muscle
        if self.ent_squash:
            # entropy of the muscles as they really are (after tanh): add log|d tanh/dx|.
            # Without this, "be random" is rewarded by pushing the muscles into saturation
            # (bang-bang, where tanh has no gradient), which also starves the actor of its
            # return gradient: a runaway that ends in thrashing.
            ent = ent + 2 * (0.6931471805599453 - pre - F.softplus(-2 * pre))
        return torch.tanh(pre), (ent if per else ent.sum(-1))


def kl_cat(p, q):
    """KL(p || q) for (..., 32, 16) probabilities, summed over slots."""
    return (p * (torch.log(p) - torch.log(q))).sum((-1, -2))


DRIVE = "linear"


def set_drive(name):
    """linear: wellbeing = (energy + health) / 2 (all runs before v3).
    homeo: wellbeing = (health + 1 - hunger) / 2, hunger being the felt -log(energy) sense: a
    half-full stomach is fine, an almost empty one hurts more and more (like real hunger)."""
    global DRIVE
    DRIVE = name or "linear"


def wellbeing(senses):
    """The built-in drive, from (predicted) body feelings."""
    if DRIVE == "homeo" and senses.shape[-1] > 30:
        return 0.5 * (senses[..., 1].clamp(0, 1) + 1 - senses[..., 30].clamp(0, 1))   # 30 = hunger
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


def make_mind(args, res=None):
    """A Mind shaped as the run's args say (args: dict)."""
    return Mind(res or args["res"], args.get("intent", 0), args.get("g_slots", G_SLOTS), args.get("g_cls", G_CLS),
                bool(args.get("matter", 0)), args.get("jumps", ""))


def load_mind(mind, sd):
    """Load a state dict, also from minds saved with fewer muscles (4 before gaze): the new
    muscle's weights start at zero (its input to memory has no effect; its actor output is a
    zero mean with the usual starting spread)."""
    own = mind.state_dict()
    for k, v in sd.items():
        if k not in own:          # e.g. jump heads of a mind loaded into one without them
            continue
        t = own[k]
        if t.shape == v.shape:
            t.copy_(v)
        elif k.startswith("actor") and t.dim() >= 1 and t.shape[0] == 2 * ACT and v.shape[0] < t.shape[0]:
            a0 = v.shape[0] // 2                              # [means, stds] halves
            t.zero_()
            t[:a0].copy_(v[:a0] if t.dim() == 1 else v[:a0, :t.shape[1]])
            t[ACT:ACT + a0].copy_(v[a0:] if t.dim() == 1 else v[a0:, :t.shape[1]])
        else:                                                 # inputs appended at the end (z, muscles)
            t.zero_()
            t[tuple(slice(0, n) for n in v.shape)].copy_(v)
