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

from .model import FEAT, H_DIM, Z_DIM, Mind, eff_rank, kl_cat, vicreg, wellbeing
from .render import render
from .world import ACT, NF, SENSES, World


class Replay:
    """What the creatures lived, kept on the GPU: eyes as uint8, plus senses, muscles, memory."""

    def __init__(self, cap, B, res, dev):
        self.cap, self.B, self.n, self.ptr = cap, B, 0, 0
        self.img = torch.zeros(cap, B, 6, res, res, dtype=torch.uint8, device=dev)
        self.sen = torch.zeros(cap, B, SENSES, dtype=torch.float16, device=dev)
        self.act = torch.zeros(cap, B, ACT, dtype=torch.float16, device=dev)      # muscles that led here
        self.first = torch.zeros(cap, B, dtype=torch.bool, device=dev)
        self.cont = torch.zeros(cap, B, dtype=torch.bool, device=dev)
        self.h = torch.zeros(cap, B, H_DIM, dtype=torch.float16, device=dev)     # memory it had
        self.dev = dev

    def add(self, img, sen, act, first, cont, h):
        i = self.ptr
        self.img[i], self.sen[i], self.act[i] = img, sen, act
        self.first[i], self.cont[i], self.h[i] = first, cont, h
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs, T):
        base = (self.ptr - self.n) % self.cap
        s = torch.randint(0, self.n - T + 1, (bs,), device=self.dev)
        e = torch.randint(0, self.B, (bs,), device=self.dev)
        t = (base + s[:, None] + torch.arange(T, device=self.dev)[None]) % self.cap      # bs,T
        ee = e[:, None].expand(bs, T)
        return dict(img=self.img[t, ee], sen=self.sen[t, ee].float(), act=self.act[t, ee].float(),
                    first=self.first[t, ee], cont=self.cont[t, ee].float(), h0=self.h[t[:, 0], e].float())


def lambda_returns(r, c, v, gamma, lam):
    """r, c: (T, N) reward/continue for steps 1..T; v: (T+1, N) values for 0..T."""
    T = r.shape[0]
    out = [None] * T
    last = v[-1]
    for t in reversed(range(T)):
        last = r[t] + gamma * c[t] * ((1 - lam) * v[t + 1] + lam * last)
        out[t] = last
    return torch.stack(out)


class Learner:
    def __init__(self, mind: Mind, args):
        self.m, self.a = mind, args
        wm = [p for n, p in mind.named_parameters() if not n.startswith(("actor", "critic", "slow_critic", "minds_eye"))]
        self.opt_wm = torch.optim.Adam(wm, lr=args.lr_wm, eps=1e-8)
        self.wm_params = wm
        self.opt_actor = torch.optim.Adam(mind.actor.parameters(), lr=args.lr_ac, eps=1e-5)
        self.opt_critic = torch.optim.Adam(mind.critic.parameters(), lr=args.lr_ac, eps=1e-5)
        self.opt_me = torch.optim.Adam(mind.minds_eye.parameters(), lr=3e-4)
        self.ret_lo = self.ret_hi = None

    def update(self, batch):
        m, a = self.m, self.a
        bs, T = batch["first"].shape
        dev = batch["first"].device
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(dev.type == "cuda" and a.bf16))
        # ------------------------------------------------ world model
        with amp:
            emb = m.eye(batch["img"].flatten(0, 1)).view(bs, T, -1)
        h = batch["h0"]
        z = torch.zeros(bs, Z_DIM, device=dev)
        hs, zs, posts, priors = [], [], [], []
        for t in range(T):
            f = batch["first"][:, t].float()[:, None]
            if t > 0:      # (at t = 0 the stored memory is exactly what the creature had)
                with amp:
                    h = m.step_h(h * (1 - f), z * (1 - f), batch["act"][:, t] * (1 - f))
            with amp:
                q = m.observe(h, emb[:, t], batch["sen"][:, t])
                p = m.imagine_z(h)
            z = m.sample(q)
            hs.append(h); zs.append(z); posts.append(q); priors.append(p)
        hs, zs = torch.stack(hs, 1), torch.stack(zs, 1)
        post, prior = torch.stack(posts, 1), torch.stack(priors, 1)
        feat = torch.cat([hs, zs], -1)
        with amp:
            sen_pred = m.sense_head(feat).float()
            cont_logit = m.cont_head(feat).float().squeeze(-1)
        l_sen = F.mse_loss(sen_pred, batch["sen"], reduction="none").sum(-1).mean()
        l_cont = F.binary_cross_entropy_with_logits(cont_logit, batch["cont"])
        dyn = kl_cat(post.detach(), prior)
        rep = kl_cat(post, prior.detach())
        l_kl = 0.5 * dyn.clamp(min=1.0).mean() + 0.1 * rep.clamp(min=1.0).mean()
        var, cov, estd = vicreg(emb.flatten(0, 1))
        l_eye = a.vic_var * var + a.vic_cov * cov
        loss = l_sen + l_cont + l_kl + l_eye
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.wm_params, 1000.0)
        self.opt_wm.step()

        # ------------------------------------------------ mind's eye (detached, page only)
        with amp:
            pic = m.minds_eye(feat.detach().flatten(0, 1)[:: max(1, T // 4)]).float()
        tgt = batch["img"].flatten(0, 1)[:: max(1, T // 4), :3].float() / 255.0
        l_me = F.mse_loss(pic, tgt)
        self.opt_me.zero_grad(set_to_none=True)
        l_me.backward()
        self.opt_me.step()

        # ------------------------------------------------ imagination: actor and critic
        for p_ in self.wm_params:
            p_.requires_grad_(False)
        h = hs.detach().flatten(0, 1)
        z = zs.detach().flatten(0, 1)
        fts, ents = [torch.cat([h, z], -1)], []
        for _ in range(a.horizon):
            with amp:
                act, ent = m.act(fts[-1])
                h = m.step_h(h, z, act)
                z = m.sample(m.imagine_z(h))
            fts.append(torch.cat([h, z], -1))
            ents.append(ent)
        ft = torch.stack(fts)                                              # H+1, N, FEAT
        with amp:
            sen = m.sense_head(ft).float()
            cont = torch.sigmoid(m.cont_head(ft).float().squeeze(-1))
            v_slow = m.slow_critic(ft).float().squeeze(-1)
        for p_ in self.wm_params:
            p_.requires_grad_(True)
        r = wellbeing(sen[1:])                                             # reward on arrival
        c = cont[1:]
        scale = 1.0 / (1 - a.gamma)
        ret = lambda_returns(r, c, v_slow * scale, a.gamma, a.lam)         # H, N
        start_alive = batch["cont"].flatten()[None]                     # don't imagine from a dead moment
        w = (torch.cumprod(torch.cat([torch.ones_like(c[:1]), c[:-1]], 0), 0) * start_alive).detach()
        # return normalisation (Dreamer v3): divide by the 5-95% range, at least 1
        lo, hi = torch.quantile(ret.detach().flatten()[:65536], torch.tensor([0.05, 0.95], device=dev))
        if self.ret_lo is None:
            self.ret_lo, self.ret_hi = lo, hi
        self.ret_lo = 0.99 * self.ret_lo + 0.01 * lo
        self.ret_hi = 0.99 * self.ret_hi + 0.01 * hi
        S = torch.clamp(self.ret_hi - self.ret_lo, min=1.0)
        ent = torch.stack(ents)
        l_actor = -(w * ret / S).mean() - a.ent * (w * ent).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        l_actor.backward(inputs=list(m.actor.parameters()))
        torch.nn.utils.clip_grad_norm_(m.actor.parameters(), 100.0)
        self.opt_actor.step()
        with amp:
            v = m.critic(ft[:-1].detach()).float().squeeze(-1)
        target = (ret.detach() / scale)
        l_critic = (w * ((v - target) ** 2 + (v - v_slow[:-1].detach()) ** 2)).mean()
        self.opt_critic.zero_grad(set_to_none=True)
        l_critic.backward()
        torch.nn.utils.clip_grad_norm_(m.critic.parameters(), 100.0)
        self.opt_critic.step()
        with torch.no_grad():
            for ps, pc in zip(m.slow_critic.parameters(), m.critic.parameters()):
                ps.lerp_(pc, 0.02)
        return dict(sen=l_sen.item(), cont=l_cont.item(), kl=dyn.mean().item(), var=var.item(), cov=cov.item(),
                    estd=estd.item(), me=l_me.item(), actor=l_actor.item(), critic=l_critic.item(),
                    wb=r.mean().item(), ret=(ret.mean() / scale).item(), ent=ent.mean().item(), gn=gn.item(),
                    _emb=emb)


class Actor:
    """Runs the B creatures in their worlds with the current mind."""

    def __init__(self, mind, world, res, dev, deterministic=False):
        self.m, self.w, self.res, self.dev, self.det = mind, world, res, dev, deterministic
        B = world.B
        self.h, self.z = mind.initial(B, dev)
        self.prev_a = torch.zeros(B, ACT, device=dev)
        self.first = torch.ones(B, dtype=torch.bool, device=dev)
        self.cont = torch.ones(B, dtype=torch.bool, device=dev)
        self.life_t = torch.zeros(B, dtype=torch.long, device=dev)
        self.life_food = torch.zeros(B, 3, device=dev)

    @torch.no_grad()
    def observe_act(self, bf16=True):
        m, B = self.m, self.w.B
        img = render(self.w, self.res)
        sen = self.w.senses.clone()
        f = self.first.float()[:, None]
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(self.dev.type == "cuda" and bf16))
        with amp:
            h = m.step_h(self.h * (1 - f), self.z * (1 - f), self.prev_a * (1 - f))
            emb = m.eye(img)
            q = m.observe(h, emb, sen)
        z = m.sample(q)
        with amp:
            a, _ = m.act(torch.cat([h, z], -1), deterministic=self.det)
        a = a.float()
        rec = (img, sen, self.prev_a.clone(), self.first.clone(), self.cont.clone(), h.half())
        self.h, self.z = h.float(), z.float()
        return a, rec, emb

    def advance(self, a):
        """For tests that step the world themselves: remember the muscles, the life goes on."""
        self.prev_a = a
        self.first = torch.zeros_like(self.first)

    @torch.no_grad()
    def env_step(self, a):
        """Step the worlds. Creatures whose last observation was terminal get a fresh life."""
        w = self.w
        was_terminal = ~self.cont
        done, trunc, cause, ate = w.step(a)
        self.life_t += 1
        okf = ate >= 0
        self.life_food[okf, ate[okf]] += 1
        ended = []
        # finished lives (death -> next obs is terminal; truncation -> fresh life right away)
        newly_dead = done & ~was_terminal
        stats = dict(dead=newly_dead, trunc=trunc & ~was_terminal, cause=cause, t=self.life_t.clone(), food=self.life_food.clone())
        reset = was_terminal | (trunc & ~was_terminal)
        idx = reset.nonzero().squeeze(-1)
        if len(idx):
            w.reset(idx)
            self.life_t[idx] = 0
            self.life_food[idx] = 0
        self.first = reset
        self.cont = ~(done & ~was_terminal)
        self.cont[idx] = True
        self.prev_a = torch.where(reset[:, None], torch.zeros_like(a), a)
        return stats


def quick_test(mind, res, dev, mode="test", n=64, seed=12345, bf16=True):
    """n lives in unseen worlds, acting on habit. Returns mean steps alive and food."""
    w = World(n, dev, mode=mode, seed=seed)
    act = Actor(mind, w, res, dev, deterministic=True)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    steps = torch.zeros(n, device=dev)
    food = torch.zeros(n, device=dev)
    causes = torch.zeros(n, dtype=torch.long, device=dev)
    for _ in range(1000):
        a, _, _ = act.observe_act(bf16)
        act.advance(a)
        done, trunc, cause, ate = w.step(a)
        steps += alive.float()
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
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--replay-gb", type=float, default=8.0)
    ap.add_argument("--every", type=int, default=1, help="env steps per learning update")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=0.985)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr-wm", type=float, default=3e-4)
    ap.add_argument("--lr-ac", type=float, default=8e-5)
    ap.add_argument("--ent", type=float, default=3e-4)
    ap.add_argument("--vic-var", type=float, default=1.0)
    ap.add_argument("--vic-cov", type=float, default=1.0)
    ap.add_argument("--bf16", type=int, default=1)
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
    mind = Mind(a.res).to(dev)
    print(f"mind: {sum(p.numel() for p in mind.parameters()) / 1e6:.1f}M parameters, eyes {a.res} px", flush=True)
    world = World(a.envs, dev, mode="train", seed=a.seed)
    per = 6 * a.res * a.res + SENSES * 2 + ACT * 2 + 2 + H_DIM * 2
    cap = max(a.seq * 4, int(a.replay_gb * 1e9 / (per * a.envs)))
    print(f"replay: {cap} steps per world ({cap / 10:.0f} s), {cap * a.envs / 1e6:.2f}M moments", flush=True)
    rb = Replay(cap, a.envs, a.res, dev)
    learner = Learner(mind, a)
    actor = Actor(mind, world, a.res, dev)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    last_save = last_test = t0
    steps = updates = 0
    lives = dict(n=0, t=0.0, food=torch.zeros(3), cause=torch.zeros(6))
    stats_acc = {}
    while True:
        el = (time.time() - t0) / 60
        a_, rec, _ = actor.observe_act(a.bf16)
        if steps < a.prefill:
            a_ = torch.rand_like(a_) * 2 - 1
        rb.add(*rec)
        st = actor.env_step(a_)
        steps += 1
        fin = st["dead"] | st["trunc"]
        if fin.any():
            lives["n"] += int(fin.sum())
            lives["t"] += float(st["t"][fin].sum())
            lives["food"] += st["food"][fin].sum(0).cpu()
            cc = torch.where(st["dead"], st["cause"], torch.zeros_like(st["cause"]))[fin]
            lives["cause"] += torch.bincount(cc, minlength=6)[:6].float().cpu()
        if rb.n >= a.seq + 1 and steps >= a.prefill and steps % a.every == 0:
            r = learner.update(rb.sample(a.batch, a.seq))
            emb = r.pop("_emb")
            updates += 1
            for k, v in r.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v
            if updates % 100 == 0:
                row = {k: round(v / 100, 5) for k, v in stats_acc.items()}
                stats_acc = {}
                row.update(min=round(el, 2), updates=updates, steps=steps, frames=updates * a.batch * a.seq,
                           rank=round(eff_rank(emb.flatten(0, 1)[:2048]), 1))
                if lives["n"]:
                    row.update(lives=lives["n"], life_steps=round(lives["t"] / lives["n"], 1),
                               food=[round(x, 2) for x in (lives["food"] / lives["n"]).tolist()],
                               deaths={k: int(v) for k, v in zip(["cap", "hunger", "lava", "predator", "strange", "blows"], lives["cause"].tolist())})
                    lives = dict(n=0, t=0.0, food=torch.zeros(3), cause=torch.zeros(6))
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
        now = time.time()
        done_time = el >= a.minutes
        if now - last_test >= a.test_every * 60 or done_time:
            mind.eval()
            s, f, _ = quick_test(mind, a.res, dev, bf16=a.bf16)
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
