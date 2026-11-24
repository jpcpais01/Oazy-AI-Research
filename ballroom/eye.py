"""Ballroom Eye: the same task from pixels, with the motion carried in the model's own inner state.

  python3 -m ballroom.eye --minutes 20          # train, then score on fresh rooms (one line on a pod: bash runpod/eye.sh)

What it sees: one small picture of the room with three colour channels: the ball a moment ago,
the ball now (1/30 s later), and the platforms. No coordinates, no distances, no object list.
The walls are just the edge of the picture.

What it does (not next-token: next-state, in a learned space):
  eye      the picture is cut in patches; a transformer reads them into a few "register" tokens.
           Those tokens are the model's inner state z.
  step     a small transformer moves z one moment forward: z -> z'. It never sees a picture again.
  readout  from z: where the ball is, and whether it has come to rest.
  one-shot from the first z only: where the ball will end (the "intuition" head, for comparison).

Training rolls z forward in several windows of K steps, one after another, and asks the readout for the
true position at every step, so the inner state has to carry everything the motion needs, for seconds. One more term asks the rolled state to agree
with what the eye would see at that later moment (a JEPA-style target), which keeps long rolls on track.

The cloud of possibilities: the same start is asked many times, with the picture or the first inner state
nudged a little, and every copy is rolled to its end. The cloud then votes: the middle of its biggest clump.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import sim
from .model import Block
from .test import score, vote


def render(pa, pb, boxes, res):
    """Pictures [N,3,res,res]: ball before, ball now, platforms. Soft edges carry sub-pixel position."""
    g = (torch.arange(res, device=pa.device, dtype=torch.float32) + 0.5) / res
    xs = g[None, None, :]; ys = (1 - g)[None, :, None]            # row 0 is the top of the room

    def ball(p):
        d = ((xs - p[:, 0, None, None]) ** 2 + (ys - p[:, 1, None, None]) ** 2).sqrt()
        return ((sim.R - d) * res + 0.5).clamp(0, 1)

    pl = boxes[:, 4:]                                             # walls are not drawn: they are the picture's edge
    ix = ((pl[:, :, 2, None] - (g[None, None, :] - pl[:, :, 0, None]).abs()) * res + 0.5).clamp(0, 1)        # [N,P,res]
    iy = ((pl[:, :, 3, None] - ((1 - g)[None, None, :] - pl[:, :, 1, None]).abs()) * res + 0.5).clamp(0, 1)  # [N,P,res]
    plat = (iy[:, :, :, None] * ix[:, :, None, :]).amax(1)
    return torch.stack([ball(pa), ball(pb), plat], 1)


class Eye(nn.Module):
    def __init__(self, res=128, patch=16, d=192, enc=4, step=3, heads=4, reg=8):
        super().__init__()
        self.cfg = dict(res=res, patch=patch, d=d, enc=enc, step=step, heads=heads, reg=reg)
        self.res, self.patch, self.reg = res, patch, reg
        n = (res // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, d)
        self.where = nn.Parameter(torch.randn(n, d) * 0.02)       # which patch is where
        self.registers = nn.Parameter(torch.randn(reg, d) * 0.02)
        self.enc = nn.ModuleList(Block(d, heads) for _ in range(enc))
        self.enc_norm = nn.LayerNorm(d)
        self.stp = nn.ModuleList(Block(d, heads) for _ in range(step))
        self.stp_norm = nn.LayerNorm(d)
        self.read1 = nn.Linear(d, d); self.read2 = nn.Linear(d, 3)   # x, y, rest logit
        self.shot1 = nn.Linear(d, d); self.shot2 = nn.Linear(d, 2)   # final x, y

    def see(self, img):
        """Pictures -> inner state z [N,reg,d]."""
        N, P = img.shape[0], self.patch
        x = img.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(N, -1, 3 * P * P)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=img.is_cuda):   # the heavy part; the rest stays exact
            x = self.embed(x) + self.where
            x = torch.cat([self.registers.expand(N, -1, -1), x], 1)
            m = torch.ones(N, x.shape[1], dtype=torch.bool, device=x.device)
            for b in self.enc:
                x = b(x, m)
        return self.enc_norm(x[:, :self.reg].float())

    def step(self, z):
        m = torch.ones(z.shape[0], z.shape[1], dtype=torch.bool, device=z.device)
        x = z
        for b in self.stp:
            x = b(x, m)
        return self.stp_norm(x)

    def read(self, z):
        return self.read2(F.gelu(self.read1(z[:, 0])))

    def shot(self, z):
        return self.shot2(F.gelu(self.shot1(z[:, 1])))


@torch.no_grad()
def roll(net, z, steps=360, noise=0.0):
    """Roll the inner state until the readout says the ball rests. Returns the final position."""
    if noise:
        z = z + noise * torch.randn_like(z)
    N = z.shape[0]
    out = torch.zeros(N, 2, device=z.device)
    live = torch.ones(N, dtype=torch.bool, device=z.device)
    for k in range(steps + 1):
        y = net.read(z)
        done = live & ((y[:, 2] > 0) | (k == steps))
        out[done] = y[done, :2]
        live = live & ~done
        if not live.any():
            break
        z = net.step(z)
    return out


def rooms_on(dev, n, plat, seed):
    rng = np.random.default_rng(seed)
    boxes, count, p0, v0 = sim.make_rooms(n, rng, plat)
    P, _, stop = sim.rollout(boxes, p0, v0)
    t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    return boxes, t(boxes), t(P), t(stop, torch.long)


def evaluate(net, dev, a, rooms=3000):
    net.eval(); res = {}
    for label, plat, seed in (("2-3 platforms (like training)", (2, 3), 101), ("4-5 platforms (never seen)", (4, 5), 102)):
        boxes, tb, P, stop = rooms_on(dev, rooms, plat, seed)      # the same rooms ballroom.test uses
        fin = P[-1].cpu().numpy()
        shot, law, seen = [], [], []
        kinds = [("nudged pictures", 0.0)] + [(f"nudged inner state x{n:g}", n) for n in a.noise]
        ends = {k: [] for k, _ in kinds}
        with torch.no_grad():
            for i in range(0, rooms, 500):
                sl = slice(i, i + 500)
                z = net.see(render(P[0, sl], P[1, sl], tb[sl], a.res))
                shot.append(net.shot(z)); law.append(roll(net, z))
                seen.append((net.read(z)[:, :2] - P[1, sl]).norm(dim=1))
                # clouds: the same start asked many times, each copy rolled to its end
                for kind, n in kinds:
                    if n:       # the first inner state itself is nudged
                        e = [roll(net, z, noise=n) for _ in range(a.cloud)]
                    else:       # the picture is nudged: ball a hair away (0.002) with a hair of other speed (0.02), as in the object test
                        e = []
                        for _ in range(a.cloud):
                            pa = (P[0, sl] + 0.002 * torch.randn_like(P[0, sl])).clamp(sim.R, 1 - sim.R)
                            pb = pa + (P[1, sl] - P[0, sl]) + 0.02 / 30 * torch.randn_like(pa)
                            e.append(roll(net, net.see(render(pa, pb, tb[sl], a.res))))
                    ends[kind].append(torch.stack(e))                                             # [C,n,2]
        rows = [score("pixels, one shot", torch.cat(shot).cpu().numpy(), fin, boxes),
                score("pixels, steps in its own space", torch.cat(law).cpu().numpy(), fin, boxes)]
        notes = []
        for kind, _ in kinds:                                         # every cloud votes: the middle of its biggest clump
            pred, sure = vote(torch.cat(ends[kind], 1))
            pred, sure = pred.cpu().numpy(), sure.cpu().numpy()
            rows.append(score("cloud vote, " + kind, pred, fin, boxes))
            hit = np.sqrt(((pred - fin) ** 2).sum(1)) < 2 * sim.R; m = sure >= 0.75
            notes.append(f"  {kind}: 3/4 of the cloud agrees in {100 * m.mean():.0f}% of rooms" + (f", hits {100 * hit[m].mean():.0f}% there" if m.any() else "")
                         + (f", {100 * hit[~m].mean():.0f}% elsewhere" if (~m).any() else ""))
        see_err = torch.cat(seen).median().item()
        res[label] = dict(rows=rows, sees_ball_within=see_err, clouds=notes)
        print(f"\n== {label}, {rooms} fresh rooms ==")
        print(f"{'':40s}{'mean miss':>10s}{'median':>9s}{'within a ball':>15s}{'right surface':>15s}")
        for r in rows:
            print(f"{r['model']:40s}{r['mean']:10.3f}{r['median']:9.3f}{r['within_ball']:14.1f}%{r['right_surface']:14.1f}%")
        print(f"how well it sees the ball it is shown: median error {see_err:.4f} = {see_err / (2 * sim.R):.2f} ball widths")
        print(f"when the cloud of {a.cloud} agrees, is it right?"); print("\n".join(notes))
    net.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--rooms", type=int, default=100000)
    ap.add_argument("--plat", default="2,3")
    ap.add_argument("--batch", type=int, default=1024 if torch.cuda.is_available() else 64)
    ap.add_argument("--unroll", type=int, default=24, help="steps the inner state is rolled in training")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--enc", type=int, default=4)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--reg", type=int, default=8)
    ap.add_argument("--latent", type=float, default=1.0, help="weight of the agree-with-the-eye term")
    ap.add_argument("--test-rooms", type=int, default=3000)
    ap.add_argument("--windows", type=int, default=4, help="windows of --unroll steps rolled one after another in training")
    ap.add_argument("--init", default="", help="start from a saved eye.pt")
    ap.add_argument("--cloud", type=int, default=16, help="nudged copies per room in the test")
    ap.add_argument("--noise", type=float, nargs="*", default=[0.1, 0.3], help="sizes of the nudge to the inner state")
    ap.add_argument("--out", default="ballroom/runs/eye")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    lo, hi = map(int, a.plat.split(","))
    t0 = time.time()
    _, tb, P, stop = rooms_on(dev, a.rooms, (lo, hi), a.seed)
    T = P.shape[0] - 1; K = a.unroll; W = a.windows
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    net = Eye(a.res, a.patch, a.d, a.enc, a.step, reg=a.reg).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["model"]); print(f"starting from {a.init}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=0.01)
    print(f"rooms {a.rooms} ({lo}-{hi} platforms) {time.time() - t0:.0f}s; eye model {sum(p.numel() for p in net.parameters()):,} weights "
          f"on {dev}; {a.res}x{a.res} pictures, inner state {a.reg} tokens of {a.d}, rolled {W} x {K} steps", flush=True)
    budget = a.minutes * 60; it = 0; t0 = time.time()
    ks = torch.arange(K + 1, device=dev)
    while (it < a.iters) if a.iters else (time.time() - t0 < budget):
        frac = it / a.iters if a.iters else (time.time() - t0) / budget
        lr = a.lr * min(1.0, (it + 1) / 500) * (0.5 + 0.5 * np.cos(np.pi * min(1.0, frac))) + 1e-5
        for g in opt.param_groups:
            g["lr"] = lr
        r = torch.randint(0, a.rooms, (a.batch,), device=dev)
        # a moment on the true path: anywhere before rest (a quarter of the time the very start)
        t = (torch.rand(a.batch, device=dev) * (stop[r] + 3).float()).long().clamp(max=T - W * K - 2)
        t = torch.where(torch.rand(a.batch, device=dev) < 0.25, torch.zeros_like(t), t)
        br = tb[r]
        z = net.see(render(P[t, r], P[t + 1, r], br, a.res))
        l_shot = F.smooth_l1_loss(net.shot(z) / 0.3, P[stop[r], r] / 0.3, beta=0.1)
        l_pos = 0.0; l_rest = 0.0; l_lat = 0.0
        for w in range(W):                                                 # a long roll, learned one window at a time
            t1 = t + 1 + w * K
            tgt = P[t1[None, :] + ks[:, None], r[None, :]]                 # [K+1,B,2] true positions
            at_rest = (t1[None, :] + ks[:, None] >= stop[r][None, :]).float()
            for k in range(K + 1):
                y = net.read(z)
                l_pos = l_pos + F.smooth_l1_loss(y[:, :2] / 0.05, tgt[k] / 0.05, beta=0.2)
                l_rest = l_rest + F.binary_cross_entropy_with_logits(y[:, 2], at_rest[k])
                if k < K:
                    z = net.step(z)
            with torch.no_grad():                                          # what the eye sees at that later moment
                z_eye = net.see(render(P[t1 + K - 1, r], P[t1 + K, r], br, a.res))
            l_lat = l_lat + F.mse_loss(z, z_eye)
            z = z.detach()                                                 # the next window starts from the rolled state, never from the eye
        l_pos = l_pos / (W * (K + 1)); l_rest = l_rest / (W * (K + 1)); l_lat = l_lat / W
        loss = l_pos + 0.3 * l_rest + l_shot + a.latent * l_lat
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        it += 1
        if it % 100 == 0:
            el = time.time() - t0
            print(f"it {it} loss {loss.item():.4f} pos {l_pos.item():.4f} rest {l_rest.item():.4f} shot {l_shot.item():.4f} "
                  f"agree {l_lat.item():.4f} {el:.0f}s ({it / el:.1f} steps/s)", flush=True)
    torch.save({"model": net.state_dict(), "cfg": net.cfg, "plat": [lo, hi], "steps": it}, os.path.join(a.out, "eye.pt"))
    print(f"saved {a.out}/eye.pt after {it} steps", flush=True)
    res = evaluate(net, dev, a, a.test_rooms)
    json.dump(res, open(os.path.join(a.out, "eye.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
