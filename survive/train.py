"""Train the creature: live in many worlds at once, learn a world model from what it lived,
learn habits inside imagination.

    python3 -m survive.train --minutes 300 --res 64 --out /workspace/oazy_runs/survive/clean

Every step all B creatures act once (1 frame each); then one learning update on a batch of
remembered clips (carried memory: each clip starts from the memory the creature really had).
Saves the mind every --save-every minutes, logs to log.jsonl, and runs quick tests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import FEAT, H_DIM, set_drive, Z_DIM, Mind, eff_rank, gdist, gsample, kl_cat, maxcos, vicreg, wellbeing
from .render import render, set_eyes
from . import world as _world
from .world import ACT, NF, SENSES, World


class Replay:
    """What the creatures lived, kept on the GPU: eyes as uint8, plus senses, muscles, memory.

    Emotion (optional): every moment carries how much it mattered to the body (the size of the
    change in wellbeing, and death) and how surprising it was (the world model's KL the last
    time it was studied; new moments start as "very surprising"). Clips are then drawn around
    moments in proportion to surprise x importance, half the time; the other half uniformly.
    Nothing about rewards or goals changes: only what the mind goes back to study.
    """

    def __init__(self, cap, B, res, dev, emotion=0.0):
        self.cap, self.B, self.n, self.ptr = cap, B, 0, 0
        self.img = torch.zeros(cap, B, 6, res, res, dtype=torch.uint8, device=dev)
        self.sen = torch.zeros(cap, B, SENSES, dtype=torch.float16, device=dev)
        self.act = torch.zeros(cap, B, ACT, dtype=torch.float16, device=dev)      # muscles that led here
        self.first = torch.zeros(cap, B, dtype=torch.bool, device=dev)
        self.cont = torch.zeros(cap, B, dtype=torch.bool, device=dev)
        self.h = torch.zeros(cap, B, H_DIM, dtype=torch.float16, device=dev)     # memory it had
        self.emotion = emotion
        self.imp = torch.zeros(cap, B, device=dev)                                # how much it mattered
        self.surp = torch.zeros(cap, B, device=dev)                               # how surprising it was
        self.surp_max = torch.ones((), device=dev)
        self.prev_wb = None
        self.dev = dev

    def add(self, img, sen, act, first, cont, h):
        i = self.ptr
        self.img[i], self.sen[i], self.act[i] = img, sen, act
        self.first[i], self.cont[i], self.h[i] = first, cont, h
        wb = wellbeing(sen.float())
        if self.prev_wb is None:
            self.prev_wb = wb
        self.imp[i] = (wb - self.prev_wb).abs() * (~first) + (~cont).float() * 0.1
        self.prev_wb = wb
        self.surp[i] = self.surp_max
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs, T):
        base = (self.ptr - self.n) % self.cap
        s = torch.randint(0, self.n - T + 1, (bs,), device=self.dev)
        e = torch.randint(0, self.B, (bs,), device=self.dev)
        if self.emotion > 0:
            k = int(bs * self.emotion)
            logical = (torch.arange(self.cap, device=self.dev) - base) % self.cap         # cap
            filled = (logical < self.n).float()[:, None]
            cnt = self.n * self.B
            imp = self.imp / ((self.imp * filled).sum() / cnt + 1e-6)
            surp = self.surp / ((self.surp * filled).sum() / cnt + 1e-6)
            prio = ((0.5 + surp) * (0.5 + imp) * filled).flatten()
            anchor = torch.multinomial(prio, k, replacement=True)
            at, ae = anchor // self.B, anchor % self.B
            start = (logical[at] - torch.randint(0, T, (k,), device=self.dev)).clamp(0, self.n - T)
            s = torch.cat([start, s[k:]])
            e = torch.cat([ae, e[k:]])
        t = (base + s[:, None] + torch.arange(T, device=self.dev)[None]) % self.cap      # bs,T
        ee = e[:, None].expand(bs, T)
        self.last_idx = (t, ee)
        return dict(img=self.img[t, ee], sen=self.sen[t, ee].float(), act=self.act[t, ee].float(),
                    first=self.first[t, ee], cont=self.cont[t, ee].float(), h0=self.h[t[:, 0], e].float())

    def feel(self, kl):
        """After studying a batch: remember how surprising each of its moments was."""
        t, ee = self.last_idx
        self.surp[t, ee] = kl.detach().float()
        self.surp_max = torch.maximum(self.surp_max * 0.999, kl.detach().float().max())


class LongMemory:
    """Clips kept from the creature's WHOLE life, not just the last minute (reservoir sampling:
    every clip ever offered has the same chance to be here). Without it the replay holds only
    the last ~600 updates of experience, so when the creature stops finding food for a while
    the world model forgets what food was worth, foraging fades further: a slow unlearning loop.
    """

    def __init__(self, C, T, res, dev):
        self.C, self.T, self.n, self.seen, self.dev = C, T, 0, 0, dev
        self.fresh = False
        self.d = dict(img=torch.zeros(C, T, 6, res, res, dtype=torch.uint8, device=dev),
                      sen=torch.zeros(C, T, SENSES, dtype=torch.float16, device=dev),
                      act=torch.zeros(C, T, ACT, dtype=torch.float16, device=dev),
                      first=torch.zeros(C, T, dtype=torch.bool, device=dev),
                      cont=torch.zeros(C, T, dtype=torch.float16, device=dev),
                      h0=torch.zeros(C, H_DIM, dtype=torch.float16, device=dev))

    def offer(self, b):
        k = b["img"].shape[0]
        if self.n < self.C:                                   # filling up
            m = min(k, self.C - self.n)
            slots = torch.arange(self.n, self.n + m, device=self.dev)
            src = torch.arange(m, device=self.dev)
            self.n += m
        else:
            r = (torch.rand(k, device=self.dev) * (self.seen + torch.arange(1, k + 1, device=self.dev))).long()
            keep = r < self.C
            slots, src = r[keep], keep.nonzero().squeeze(1)
        self.seen += k
        for key, v in self.d.items():
            v[slots] = b[key][src].to(v.dtype)

    def sample(self, bs):
        i = torch.randint(0, self.n, (bs,), device=self.dev)
        out = {k: (v[i] if k in ("img", "first") else v[i].float()) for k, v in self.d.items()}
        if self.fresh:
            # the stored memory is from a mind that may be hours old: start the clip from an empty
            # memory instead (as at birth), not from a state today's mind never makes
            out["h0"] = torch.zeros_like(out["h0"])
            out["first"] = out["first"].clone()
            out["first"][:, 0] = True
        return out


def lambda_returns(r, c, v, gamma, lam):
    """r, c: (T, N) reward/continue for steps 1..T; v: (T+1, N) values for 0..T."""
    T = r.shape[0]
    out = [None] * T
    last = v[-1]
    for t in reversed(range(T)):
        last = r[t] + gamma * c[t] * ((1 - lam) * v[t + 1] + lam * last)
        out[t] = last
    return torch.stack(out)


METRICS = ["sen", "cont", "kl", "var", "cov", "estd", "me", "actor", "critic", "wb", "ret", "ent", "rgoal", "went", "goal", "entw", "gn",
           "e_push", "e_turn", "e_jump", "e_grip", "e_gaze"]


class Learner:
    """One learning update = world model + mind's eye + actor/critic in imagination.

    Speed: the three parts are pure functions of tensors, compiled with torch.compile (the
    32-step and 15-step loops unroll into a few hundred fused kernels instead of thousands of
    tiny launches); optimizers are fused; nothing waits on the GPU (metrics stay on the GPU and
    are read every 100 updates).
    """

    def __init__(self, mind: Mind, args):
        self.m, self.a = mind, args
        dev = next(mind.parameters()).device
        fused = dict(fused=True) if dev.type == "cuda" else {}
        wm = [p for n, p in mind.named_parameters()
              if not n.startswith(("actor", "critic", "slow_critic", "minds_eye", "goal_", "will"))]
        self.wm_params = wm
        self.opt_wm = torch.optim.Adam(wm, lr=args.lr_wm, eps=1e-8, **fused)
        self.opt_actor = torch.optim.Adam(mind.actor.parameters(), lr=args.lr_ac, eps=1e-5, **fused)
        self.opt_critic = torch.optim.Adam(mind.critic.parameters(), lr=args.lr_ac, eps=1e-5, **fused)
        self.opt_me = torch.optim.Adam(mind.minds_eye.parameters(), lr=3e-4, **fused)
        if mind.intent:
            self.opt_goal = torch.optim.Adam(list(mind.goal_enc.parameters()) + list(mind.goal_dec.parameters()),
                                             lr=getattr(args, "lr_goal", 3e-4), **fused)
            self.opt_will = torch.optim.Adam(list(mind.will.parameters()) + list(mind.will_critic.parameters()),
                                             lr=getattr(args, "lr_will", 3e-4), eps=1e-5, **fused)
        et = getattr(args, "ent_target", None)
        self.ent_per = bool(getattr(args, "ent_per", 0))
        # one thermostat for the sum of all muscles (v3), or one per muscle (--ent-per 1): with a
        # single sum the cheap muscles (jump, grip, gaze) can stay random and meet the target
        # while push and turn slam to their limits
        self.ent_target = None if et is None or et != et else float(et) * (1 if self.ent_per else ACT)
        shape = (ACT,) if self.ent_per else ()
        self.log_ent = torch.full(shape, math.log(args.ent), device=dev, requires_grad=True)
        self.opt_ent = torch.optim.Adam([self.log_ent], lr=1e-2)
        # how much the world model cares to get each feeling right: the drive is read from
        # energy, health and hunger (3 of 31 numbers, changing by ~0.001 a step), so in the plain sum
        # the loud intensity/ear numbers drown them (--body-w > 1 weights those three up)
        self.sen_w = torch.ones(SENSES, device=dev)
        self.sen_w[[0, 1, 30]] = float(getattr(args, "body_w", 1.0))            # energy, health, hunger
        self.ret_lo = torch.zeros((), device=dev)
        self.ret_hi = torch.zeros((), device=dev)
        self.first_call = True
        self.amp = bool(dev.type == "cuda" and args.bf16)
        mode = getattr(args, "compile_mode", "default")
        comp = (lambda f: torch.compile(f, mode=None if mode == "default" else mode)) if getattr(args, "compile", 0) else (lambda f: f)
        self.wm_fn, self.me_fn, self.ac_fn = comp(self._wm), comp(self._me), comp(self._ac)
        self.goal_fn = comp(self._goal)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp)

    # ------------------------------------------------ world model
    def _wm(self, img, sen, act, first, cont, h0):
        m, a = self.m, self.a
        bs, T = first.shape
        with self._autocast():
            emb = m.eye(img.flatten(0, 1)).view(bs, T, -1)
            # the posterior's first layer, split: the part that sees eye+senses is done for all
            # moments at once; only the memory part runs inside the loop
            W, bias = m.post[0].weight, m.post[0].bias
            xin = F.linear(torch.cat([m.eye_in(emb), sen.to(emb.dtype)], -1), W[:, H_DIM:], bias)
        post_rest = m.post[1:]
        h = h0
        z = torch.zeros(bs, Z_DIM, device=h0.device)
        firstf = first.float()[..., None]
        hs, zs, posts = [], [], []
        for t in range(T):
            if t > 0:      # (at t = 0 the stored memory is exactly what the creature had)
                keep = 1 - firstf[:, t]
                with self._autocast():
                    h = m.step_h(h * keep, z * keep, act[:, t] * keep)
            with self._autocast():
                q = m.dist(post_rest(F.linear(h.to(xin.dtype), W[:, :H_DIM]) + xin[:, t]))
            z = m.sample(q)
            hs.append(h); zs.append(z); posts.append(q)
        hs, zs, post = torch.stack(hs, 1), torch.stack(zs, 1), torch.stack(posts, 1)
        feat = torch.cat([hs, zs], -1)
        with self._autocast():
            prior = m.imagine_z(hs)
            sen_pred = m.sense_head(feat).float()
            cont_logit = m.cont_head(feat).float().squeeze(-1)
        l_sen = (F.mse_loss(sen_pred, sen, reduction="none") * self.sen_w).sum(-1).mean()
        l_cont = F.binary_cross_entropy_with_logits(cont_logit, cont)
        dyn = kl_cat(post.detach(), prior)
        rep = kl_cat(post, prior.detach())
        # free bits: the prior is always pulled toward the posterior above 1 nat; the posterior is
        # only squeezed toward the prior above rep_free nats. With no picture loss, nothing else
        # asks the moment to keep what the eyes saw, so a 1-nat squeeze slowly pushed vision out
        # of z (KL fell to the floor and foraging faded); rep_free 3 leaves room for it.
        l_kl = 0.5 * dyn.clamp(min=1.0).mean() + 0.1 * rep.clamp(min=a.rep_free).mean()
        e_all = emb.flatten(0, 1)
        rows = getattr(a, "vic_rows", 0)
        if rows and e_all.shape[0] > rows:          # the 4096x4096 covariance on a random subset of moments
            e_all = e_all[torch.randperm(e_all.shape[0], device=e_all.device)[:rows]]
        var, cov, estd = vicreg(e_all)
        loss = l_sen + l_cont + l_kl + a.vic_var * var + a.vic_cov * cov
        rw = getattr(a, "recon", 0.0)
        if rw > 0:
            # the inner state must be able to redraw what the left eye saw (Dreamer-style): small things
            # like a food cube barely move the eye's own summary, so without this they were dropped
            # (linear probe on v4: food direction/distance R^2 <= 0 while energy .97, predator .90).
            # The decoder itself is the mind's eye (trained on its own below); here only the gradient
            # into memory, moment and eye is used. Summed over pixels, every 2nd moment.
            k = max(1, int(getattr(a, "recon_stride", 2)))
            with self._autocast():
                pic = m.minds_eye(feat[:, ::k].flatten(0, 1)).float()
            tgt = img[:, ::k, :3].flatten(0, 1).float() / 255.0
            l_rec = (pic - tgt).pow(2).sum((-1, -2, -3)).mean()
            loss = loss + rw * l_rec
        if getattr(m, "matter", None) is not None:
            # the eye alone guesses what its body will feel within the next second: food eaten
            # (energy rises only by eating) and harm (health falls). Its own feelings are the
            # teacher; nothing is named. Pulls food and danger into the eye's features.
            K = 10
            e0, e1 = sen[:, :-K, 0], sen[:, K:, 0]
            h0_, h1 = sen[:, :-K, 1], sen[:, K:, 1]
            cf = torch.cumsum(first.float(), 1)
            ok = (cf[:, K:] == cf[:, :-K]).float()                       # same life all the way
            tgt = torch.stack([(e1 - e0).clamp(min=0) * 3, (h0_ - h1).clamp(min=0) * 3], -1)
            with self._autocast():
                pm = m.matter(emb[:, :-K]).float()
            l_mat = ((pm - tgt) ** 2).sum(-1).mul(ok).sum() / ok.sum().clamp(min=1)
            loss = loss + getattr(a, "matter", 0.1) * l_mat
        met = torch.stack([l_sen, l_cont, dyn.mean(), var, cov, estd]).detach()
        return loss, hs.detach(), zs.detach(), met, emb.detach(), dyn.detach()

    # ------------------------------------------------ mind's eye (detached, page only)
    def _me(self, feat, img):
        with self._autocast():
            pic = self.m.minds_eye(feat).float()
        return F.mse_loss(pic, img[:, :3].float() / 255.0)

    # ------------------------------------------------ intentions: the code of real memories
    def _goal(self, h):
        """Squeeze real memories through the 8x8 intention code and back (an autoencoder), so
        every intention the will can pick decodes to a memory the creature has really had."""
        m = self.m
        with self._autocast():
            q = gdist(m.goal_enc(h), m.g_cls)
            rec = m.goal_dec(gsample(q).to(h.dtype)).float()
        hf = h.float()
        l_rec = (rec - hf).pow(2).sum(-1).mean()
        l_kl = (q * torch.log(q * m.g_cls)).sum((-1, -2)).mean()            # toward using every option
        unexpl = l_rec.detach() / hf.var(0).sum().clamp(min=1e-6)          # logged: share of memory the code misses
        return l_rec + l_kl, unexpl

    # ------------------------------------------------ imagination: actor and critic
    def _ac(self, hs, zs, cont0, ret_lo, ret_hi, init, ent_w):
        m, a = self.m, self.a
        st = max(1, int(getattr(a, "start_stride", 1)))
        h, z = hs.flatten(0, 1)[::st], zs.flatten(0, 1)[::st]
        K = m.intent
        fts, ents, xs = [torch.cat([h, z], -1)], [], []
        goals, rgoal, logps, wents, dec_at = [], [], [], [], []
        code = None
        for k in range(a.horizon):
            if K and k % K == 0:                       # the will picks an intention (not differentiated)
                with self._autocast():
                    q = gdist(m.will(fts[-1].detach()), m.g_cls)
                hard = gsample(q.detach()).detach()
                logps.append((hard.view_as(q) * torch.log(q)).sum((-1, -2)))
                wents.append(-(q * torch.log(q)).sum((-1, -2)))
                dec_at.append(k)
                code = hard
                with torch.no_grad(), self._autocast():
                    goal = m.goal_dec(code).float()
            x = torch.cat([fts[-1], code.to(fts[-1].dtype)], -1) if K else fts[-1]
            xs.append(x)
            with self._autocast():
                act, ent = m.act(x, per=True)                             # N, ACT
                h = m.step_h(h, z, act)
                z = m.sample(m.imagine_z(h))
            fts.append(torch.cat([h, z], -1))
            ents.append(ent)
            if K:
                rgoal.append(maxcos(h.float(), goal))
        if K:
            xs.append(torch.cat([fts[-1], code.to(fts[-1].dtype)], -1))
        ft = torch.stack(fts)                                              # H+1, N, FEAT
        x = torch.stack(xs) if K else ft
        with self._autocast():
            sen = m.sense_head(ft).float()
            cont = torch.sigmoid(m.cont_head(ft).float().squeeze(-1))
            v_slow = m.slow_critic(x).float().squeeze(-1)
        r_ext = wellbeing(sen[1:])                                         # reward on arrival
        r = r_ext + getattr(a, "intent_w", 0.3) * torch.stack(rgoal) if K else r_ext
        c = cont[1:]
        scale = 1.0 / (1 - a.gamma)
        ret = lambda_returns(r, c, v_slow * scale, a.gamma, a.lam)         # H, N
        start_alive = cont0.flatten()[::st][None]                          # don't imagine from a dead moment
        w = (torch.cumprod(torch.cat([torch.ones_like(c[:1]), c[:-1]], 0), 0) * start_alive).detach()
        # return normalisation (Dreamer v3): divide by the 5-95% range, at least 1
        flat = ret.detach().flatten()
        flat = flat[:: max(1, flat.numel() // 65536)]                    # every imagined step, evenly
        qs = torch.quantile(flat[:65536], torch.tensor([0.05, 0.95], device=ret.device))
        lo = torch.where(init, qs[0], 0.99 * ret_lo + 0.01 * qs[0])
        hi = torch.where(init, qs[1], 0.99 * ret_hi + 0.01 * qs[1])
        S = torch.clamp(hi - lo, min=1.0)
        ent_m = torch.stack(ents)                                          # H, N, ACT
        ent = ent_m.sum(-1)
        l_actor = -(w * ret / S).mean() - (w[..., None] * ent_m * ent_w).sum(-1).mean()
        with self._autocast():
            v = m.critic(x[:-1].detach()).float().squeeze(-1)
        target = ret.detach() / scale
        l_critic = (w * ((v - target) ** 2 + (v - v_slow[:-1].detach()) ** 2)).mean()
        zero = torch.zeros((), device=ret.device)
        rg, went = zero, zero
        if K:
            # the will: paid the wellbeing that follows its choice; its critic looks past the dream
            ftd = ft.detach()
            with self._autocast():
                vw_slow = m.will_slow(ftd).float().squeeze(-1)
                vw = m.will_critic(ftd[:-1]).float().squeeze(-1)
            ret_w = lambda_returns(r_ext.detach(), c.detach(), vw_slow * scale, a.gamma, a.lam)
            l_wcrit = (w * ((vw - ret_w / scale) ** 2 + (vw - vw_slow[:-1]) ** 2)).mean()
            adv = torch.stack([(ret_w[d] - vw[d].detach() * scale) for d in dec_at])
            wd = torch.stack([w[d] for d in dec_at])
            if getattr(a, "will_norm", 1):
                # standardise the advantage: which intention was better is a small difference in a
                # long return, and divided by the return range it was too weak to teach the will
                # (it stayed fully random in will_s1)
                m_ = (wd * adv).sum() / wd.sum().clamp(min=1)
                sd = torch.sqrt((wd * (adv - m_) ** 2).sum() / wd.sum().clamp(min=1) + 1e-8)
                adv = (adv - m_) / sd
            else:
                adv = adv / S
            lp, we = torch.stack(logps), torch.stack(wents)
            l_will = -(wd * lp * adv.detach()).mean() - getattr(a, "will_ent", 1e-3) * (wd * we).mean()
            l_actor = l_actor + l_will
            l_critic = l_critic + l_wcrit
            rg, went = torch.stack(rgoal).mean(), we.mean()
        met = torch.cat([torch.stack([l_actor, l_critic, r_ext.mean(), ret.mean() / scale, ent.mean(), rg, went]),
                         ent_m.mean((0, 1))]).detach()
        return l_actor, l_critic, lo.detach(), hi.detach(), met

    def update(self, batch):
        m, a = self.m, self.a
        T = batch["first"].shape[1]
        loss, hs, zs, met_wm, emb, kl = self.wm_fn(batch["img"], batch["sen"], batch["act"], batch["first"],
                                               batch["cont"], batch["h0"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.wm_params, 1000.0, foreach=True)
        self.opt_wm.step()

        k = max(1, T // 4)
        l_me = self.me_fn(torch.cat([hs, zs], -1)[:, ::k].flatten(0, 1), batch["img"][:, ::k].flatten(0, 1))
        self.opt_me.zero_grad(set_to_none=True)
        l_me.backward()
        self.opt_me.step()

        for p_ in self.wm_params:
            p_.requires_grad_(False)
        l_goal = g_un = torch.zeros((), device=hs.device)
        if m.intent:
            hh = hs.flatten(0, 1)
            l_goal, g_un = self.goal_fn(hh[:: max(1, hh.shape[0] // 4096)])
            self.opt_goal.zero_grad(set_to_none=True)
            l_goal.backward()
            self.opt_goal.step()
        init = torch.tensor(self.first_call, device=hs.device)
        ent_w = self.log_ent.exp().detach() if self.ent_target is not None else torch.tensor(a.ent, device=hs.device)
        l_actor, l_critic, self.ret_lo, self.ret_hi, met_ac = self.ac_fn(hs, zs, batch["cont"], self.ret_lo, self.ret_hi,
                                                                         init, ent_w)
        if self.ent_target is not None:
            # a thermostat on randomness: the entropy weight rises when the muscles get too sure
            # (bang-bang habits, circling) and falls when they are too random (thrashing)
            e_now = met_ac[7:7 + ACT] if self.ent_per else met_ac[4]
            l_e = (self.log_ent * (e_now - self.ent_target)).sum()      # below target -> weight rises
            self.opt_ent.zero_grad(set_to_none=True)
            l_e.backward()
            self.opt_ent.step()
            with torch.no_grad():
                self.log_ent.clamp_(math.log(1e-5), math.log(3e-2))
        self.first_call = False
        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        if m.intent:
            self.opt_will.zero_grad(set_to_none=True)
        (l_actor + l_critic).backward()          # critic sees detached states; actor never touches the critic
        for p_ in self.wm_params:
            p_.requires_grad_(True)
        torch.nn.utils.clip_grad_norm_(m.actor.parameters(), 100.0, foreach=True)
        torch.nn.utils.clip_grad_norm_(m.critic.parameters(), 100.0, foreach=True)
        self.opt_actor.step()
        self.opt_critic.step()
        if m.intent:
            torch.nn.utils.clip_grad_norm_(list(m.will.parameters()) + list(m.will_critic.parameters()), 100.0, foreach=True)
            self.opt_will.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(m.slow_critic.parameters()), list(m.critic.parameters()), 0.02)
            if m.intent:
                torch._foreach_lerp_(list(m.will_slow.parameters()), list(m.will_critic.parameters()), 0.02)
        met = torch.cat([met_wm, l_me.detach()[None], met_ac[:7], g_un[None], ent_w.detach().float().mean().reshape(1),
                         gn.detach().float()[None], met_ac[7:]])
        return met, emb, kl


def _policy_step(m, h, z, prev_a, first, img, sen, code, ctr, det: bool, amp: bool):
    """One moment of a creature: update memory, look, settle the moment, (every K steps pick an
    intention), choose muscles. ctr = steps since the intention was picked."""
    keep = 1 - first.float()[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        h = m.step_h(h * keep, z * keep, prev_a * keep)
        emb = m.eye(img)
        q = m.observe(h, emb, sen)
    z = m.sample(q)
    ft = torch.cat([h, z], -1)
    if m.intent:
        renew = first | (ctr >= m.intent)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            new = gsample(gdist(m.will(ft), m.g_cls), det).float()
        code = torch.where(renew[:, None], new, code)
        ctr = torch.where(renew, torch.ones_like(ctr), ctr + 1)
        ft = torch.cat([ft, code.to(ft.dtype)], -1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        a, _ = m.act(ft, deterministic=det)
    return a.float(), h.float(), z.float(), emb, code, ctr


_policy_compiled = None


class Actor:
    """Runs the B creatures in their worlds with the current mind."""

    def __init__(self, mind, world, res, dev, deterministic=False, compile=False):
        self.m, self.w, self.res, self.dev, self.det = mind, world, res, dev, deterministic
        B = world.B
        self.h, self.z = mind.initial(B, dev)
        self.prev_a = torch.zeros(B, ACT, device=dev)
        self.first = torch.ones(B, dtype=torch.bool, device=dev)
        self.cont = torch.ones(B, dtype=torch.bool, device=dev)
        self.ended = torch.zeros(B, dtype=torch.bool, device=dev)     # the life ended at the last step
        self.life_t = torch.zeros(B, dtype=torch.long, device=dev)
        self.life_food = torch.zeros(B, 3, device=dev)
        self.code = torch.zeros(B, max(1, mind.g_dim), device=dev)
        self.ctr = torch.zeros(B, dtype=torch.long, device=dev)
        global _policy_compiled
        if compile and _policy_compiled is None:
            _policy_compiled = torch.compile(_policy_step, dynamic=False)
        self.policy = _policy_compiled if compile else _policy_step
        self.compile = compile

    @torch.no_grad()
    def observe_act(self, bf16=True):
        img = render(self.w, self.res, compile=self.compile)
        sen = self.w.senses.clone()
        amp = bool(self.dev.type == "cuda" and bf16)
        a, h, z, emb, self.code, self.ctr = self.policy(self.m, self.h, self.z, self.prev_a, self.first, img, sen,
                                                        self.code, self.ctr, self.det, amp)
        rec = (img, sen, self.prev_a.clone(), self.first.clone(), self.cont.clone(), h.half())
        self.h, self.z = h, z
        return a, rec, emb

    def advance(self, a):
        """For tests that step the world themselves: remember the muscles, the life goes on."""
        self.prev_a = a
        self.first = torch.zeros_like(self.first)

    @torch.no_grad()
    def env_step(self, a):
        """Step the worlds. Creatures whose last observation was terminal get a fresh life."""
        w = self.w
        was_terminal = self.ended
        done, trunc, cause, ate = w.step(a)
        self.life_t += 1
        okf = ate >= 0
        self.life_food[okf, ate[okf]] += 1
        ended = []
        # finished lives (death -> next obs is terminal; truncation -> fresh life right away)
        newly_dead = done & ~was_terminal
        t_life = torch.where(newly_dead, self.life_t - 1 - _world.LIGHT, self.life_t)   # up to the moment of death
        stats = dict(dead=newly_dead, trunc=trunc & ~was_terminal, cause=cause, t=t_life, food=self.life_food.clone())
        reset = was_terminal | (trunc & ~was_terminal)
        idx = reset.nonzero().squeeze(-1)
        if len(idx):
            w.reset(idx)
            self.life_t[idx] = 0
            self.life_food[idx] = 0
        self.first = reset
        self.ended = done & ~was_terminal
        self.ended[idx] = False
        # death counts from the first moment of the light (the light is after the end: absorbing),
        # not from its last frame: otherwise a death lands 21 steps later, beyond every 1.5 s dream,
        # and starving costs nothing in the moment (audit, Nov 27)
        self.cont = ~(self.ended | (w.dying > 0))
        self.cont[idx] = True
        self.prev_a = torch.where(reset[:, None], torch.zeros_like(a), a)
        return stats


def quick_test(mind, res, dev, mode="test", n=64, seed=12345, bf16=True, compile=False):
    """n lives in unseen worlds, acting on habit. Returns mean steps alive and food."""
    w = World(n, dev, mode=mode, seed=seed, compile=compile)
    act = Actor(mind, w, res, dev, deterministic=True, compile=compile)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    steps = torch.zeros(n, device=dev)
    food = torch.zeros(n, device=dev)
    causes = torch.zeros(n, dtype=torch.long, device=dev)
    for _ in range(_world.MAX_STEPS + _world.LIGHT):
        a, _, _ = act.observe_act(bf16)
        act.advance(a)
        done, trunc, cause, ate = w.step(a)
        steps += (alive & (w.dying == 0)).float()
        food += (alive & (ate >= 0)).float()
        causes = torch.where(alive & done, cause, causes)
        alive &= ~(done | trunc)
        if not alive.any():
            break
    return steps.mean().item(), food.mean().item(), causes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=300)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--rooms", type=int, default=65536, help="fixed list of training rooms, same order in every run (0 = always new)")
    ap.add_argument("--eye-sep", type=float, default=0.08, help="distance between the two eyes (m)")
    ap.add_argument("--gaze", type=float, default=0, help="degrees the eyes can turn left/right with the 5th muscle (0 = off)")
    ap.add_argument("--light", type=int, default=0, help="steps of white light the creature sees after dying, before the life ends (0 = off; 20 = 2 s)")
    ap.add_argument("--stamina", type=int, default=0, help="1 = a fast tiredness pool (sprints/jumps drain it, rest refills it)")
    ap.add_argument("--metab", type=float, default=1.0, help="food-energy drain (0.2 = a full tank lasts 150 s standing still)")
    ap.add_argument("--life-cap", type=float, default=100, help="longest life in seconds (a time limit, not a death)")
    ap.add_argument("--heat", type=int, default=1, help="1: heat felt from far (old); 2: fades within ~1.5 m, feet burn only on lava")
    ap.add_argument("--back-speed", type=float, default=1.0, help="top speed walking backward relative to forward (v3: 0.5)")
    ap.add_argument("--drive", default="linear", help="linear (energy+health)/2, or homeo (health + 1 - hunger)/2")
    ap.add_argument("--matter", type=float, default=0, help="weight of the eye's 'will I eat / get hurt within 1 s' guess (0 = off)")
    ap.add_argument("--ent-target", type=float, default=float("nan"), help="per-muscle entropy the thermostat holds (nan = fixed --ent)")
    ap.add_argument("--ent-per", type=int, default=0, help="1 = one entropy thermostat per muscle (target per muscle)")
    ap.add_argument("--body-w", type=float, default=1.0, help="weight of energy/health/hunger in the senses loss (1 = plain sum)")
    ap.add_argument("--fovea", type=float, default=0, help="0 = plain eye; k = centre k times sharper (same pixels)")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--replay-gb", type=float, default=8.0)
    ap.add_argument("--long", type=float, default=0.5, help="share of replay memory kept as clips from the whole life (0 = off)")
    ap.add_argument("--recon", type=float, default=0.0, help="weight of redrawing the left eye from the inner state (0 = off; 1 = Dreamer scale)")
    ap.add_argument("--recon-stride", type=int, default=2)
    ap.add_argument("--smell", type=int, default=0, help="1 = two nostrils smell food")
    ap.add_argument("--long-fresh", type=int, default=0, help="1 = long-memory clips start from an empty memory, not the hours-old stored one")
    ap.add_argument("--world-v", type=int, default=1, help="2 = audit fixes: crates spawn free, z-aware crate overlap, failed carry drops, "
                    "back pressure sensor, blows/stamina from the real drive, dying body stops; 3 = also the predator lunges when close, rooms half as cluttered")
    ap.add_argument("--long-offer", type=int, default=64, help="clips offered to the long memory every 10 updates")
    ap.add_argument("--every", type=int, default=1, help="env steps per learning update")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--intent", type=int, default=0, help="a will picks an intention every K steps (0 = off; 10 = every 1 s)")
    ap.add_argument("--intent-w", type=float, default=0.3, help="pay to the muscles for reaching the intended memory (wellbeing pays 1)")
    ap.add_argument("--lr-will", type=float, default=3e-4)
    ap.add_argument("--g-slots", type=int, default=1, help="intention code: slots x options (1 x 16 = 16 intentions; willn_s1 used 8 x 8)")
    ap.add_argument("--g-cls", type=int, default=16)
    ap.add_argument("--lr-goal", type=float, default=3e-5, help="how fast intentions change meaning (willn_s1: 3e-4, they drifted)")
    ap.add_argument("--will-norm", type=int, default=1, help="standardise the will's advantage (0 = divide by the return range, as in will_s1)")
    ap.add_argument("--will-ent", type=float, default=1e-3, help="how much the will is paid to keep its choices varied")
    ap.add_argument("--start-stride", type=int, default=1, help="dream from every k-th remembered moment (2 halves the dream cost)")
    ap.add_argument("--gamma", type=float, default=0.997, help="how far ahead it cares: 1/(1-gamma) steps (0.997 = ~33 s)")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr-wm", type=float, default=3e-4)
    ap.add_argument("--lr-ac", type=float, default=8e-5)
    ap.add_argument("--ent", type=float, default=3e-4)
    ap.add_argument("--rep-free", type=float, default=3.0, help="free nats before the posterior is squeezed toward the prior (was 1)")
    ap.add_argument("--ent-squash", type=int, default=1, help="entropy of the real (tanh) muscles, not pre-tanh")
    ap.add_argument("--vic-var", type=float, default=1.0)
    ap.add_argument("--vic-cov", type=float, default=1.0)
    ap.add_argument("--vic-rows", type=int, default=2048, help="moments used for VICReg (0 = all)")
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--emotion", type=float, default=0.0, help="share of clips drawn around moments that mattered (surprise x importance)")
    ap.add_argument("--compile", type=int, default=1, help="torch.compile the learner and the creatures' step")
    ap.add_argument("--compile-mode", default="default", help="default | reduce-overhead | max-autotune-no-cudagraphs")
    ap.add_argument("--prefill", type=int, default=64, help="random steps before learning")
    ap.add_argument("--save-every", type=float, default=15)
    ap.add_argument("--test-every", type=float, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/workspace/oazy_runs/survive/clean")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    set_eyes(vars(a))
    mind = Mind(a.res, a.intent, a.g_slots, a.g_cls, matter=a.matter > 0).to(dev)
    mind.ent_squash = bool(a.ent_squash)
    print(f"mind: {sum(p.numel() for p in mind.parameters()) / 1e6:.1f}M parameters, eyes {a.res} px", flush=True)
    world = World(a.envs, dev, mode="train", seed=a.seed, compile=bool(a.compile), rooms=a.rooms)
    per = 6 * a.res * a.res + SENSES * 2 + ACT * 2 + 2 + H_DIM * 2
    cap = max(a.seq * 4, int(a.replay_gb * (1 - a.long) * 1e9 / (per * a.envs)))
    print(f"replay: {cap} steps per world ({cap / 10:.0f} s), {cap * a.envs / 1e6:.2f}M moments", flush=True)
    rb = Replay(cap, a.envs, a.res, dev, emotion=a.emotion)
    lm = None
    if a.long > 0:
        lm = LongMemory(int(a.replay_gb * a.long * 1e9 / (per * a.seq)), a.seq, a.res, dev)
        lm.fresh = bool(a.long_fresh)
        print(f"long memory: {lm.C} clips of {a.seq} steps from the whole life", flush=True)
    learner = Learner(mind, a)
    actor = Actor(mind, world, a.res, dev, compile=a.compile)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    last_save = last_test = t0
    steps = updates = 0
    def fresh_lives():
        return dict(n=torch.zeros((), device=dev), t=torch.zeros((), device=dev),
                    food=torch.zeros(3, device=dev), cause=torch.zeros(6, device=dev))
    lives = fresh_lives()
    stats_acc = torch.zeros(len(METRICS), device=dev)
    t_rate = time.time()
    while True:
        el = (time.time() - t0) / 60
        a_, rec, _ = actor.observe_act(a.bf16)
        if steps < a.prefill:
            a_ = torch.rand_like(a_) * 2 - 1
        rb.add(*rec)
        st = actor.env_step(a_)
        steps += 1
        fin = (st["dead"] | st["trunc"]).float()            # all on the GPU: nothing waits here
        lives["n"] += fin.sum()
        lives["t"] += (st["t"].float() * fin).sum()
        lives["food"] += (st["food"] * fin[:, None]).sum(0)
        cc = torch.where(st["dead"], st["cause"], torch.zeros_like(st["cause"]))
        lives["cause"] += torch.zeros(6, device=dev).index_add_(0, cc.clamp(0, 5), fin)
        if rb.n >= a.seq + 1 and steps >= a.prefill and steps % a.every == 0:
            if lm is not None and lm.n >= a.batch:
                h = a.batch // 2
                b1, b2 = rb.sample(a.batch - h, a.seq), lm.sample(h)
                met, emb, kl = learner.update({k: torch.cat([b1[k], b2[k].to(b1[k].dtype)]) for k in b1})
                if a.emotion > 0:
                    rb.feel(kl[:a.batch - h])
            else:
                met, emb, kl = learner.update(rb.sample(a.batch, a.seq))
                if a.emotion > 0:
                    rb.feel(kl)
            if lm is not None and updates % 10 == 0:
                lm.offer(rb.sample(a.long_offer, a.seq))
            updates += 1
            stats_acc += met
            if updates % 100 == 0:
                row = {k: round(v / 100, 5) for k, v in zip(METRICS, stats_acc.tolist())}
                stats_acc.zero_()
                row.update(min=round(el, 2), updates=updates, steps=steps, frames=updates * a.batch * a.seq,
                           rank=round(eff_rank(emb.flatten(0, 1)[:2048]), 1),
                           fps=round(100 * a.batch * a.seq / max(1e-3, time.time() - t_rate)))
                t_rate = time.time()
                n = lives["n"].item()
                if n:
                    row.update(lives=int(n), life_steps=round(lives["t"].item() / n, 1),
                               food=[round(x / n, 2) for x in lives["food"].tolist()],
                               deaths={k: int(v) for k, v in zip(["cap", "hunger", "lava", "predator", "strange", "blows"], lives["cause"].tolist())})
                    lives = fresh_lives()
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
        now = time.time()
        done_time = el >= a.minutes
        if now - last_test >= a.test_every * 60 or done_time:
            mind.eval()
            s, f, _ = quick_test(mind, a.res, dev, bf16=a.bf16, compile=a.compile)
            mind.train()
            row = dict(min=round(el, 2), updates=updates, test_steps=round(s, 1), test_food=round(f, 2))
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
            last_test = now
        if now - last_save >= a.save_every * 60 or done_time:
            tag = f"mind_{int(round(el))}min.pt"
            torch.save(dict(mind=mind.state_dict(), args=vars(a), minutes=el, updates=updates,
                            frames=updates * a.batch * a.seq), out / tag)
            torch.save(dict(mind=mind.state_dict(), args=vars(a), minutes=el, updates=updates,
                            frames=updates * a.batch * a.seq), out / "mind_last.pt")
            print(f"saved {tag}", flush=True)
            last_save = now
        if done_time:
            break


if __name__ == "__main__":
    main()
