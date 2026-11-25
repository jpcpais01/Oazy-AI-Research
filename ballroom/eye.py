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
import argparse, hashlib, json, math, os, time
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


class Look(nn.Module):
    """The inner state looks at the room: its tokens ask, the room's patches answer. Starts silent (zero output)."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def keys(self, room):
        """The room's answers, worked out once per picture: [2,B,h,S,dh]."""
        B, S, D = room.shape
        return self.kv(self.nk(room)).view(B, S, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4).contiguous()   # contiguous: no copy per step

    def forward(self, x, kv):
        B, T, D = x.shape; h = self.h
        q = self.q(self.nq(x)).view(B, T, h, D // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, kv[0], kv[1])          # fused: keeps nothing big for the backward pass
        return x + self.proj(o.transpose(1, 2).reshape(B, T, D))


class Eye(nn.Module):
    def __init__(self, res=128, patch=16, d=192, enc=4, step=3, heads=4, reg=8, dn=0, view=0):
        super().__init__()
        self.cfg = dict(res=res, patch=patch, d=d, enc=enc, step=step, heads=heads, reg=reg, dn=dn, view=view)
        self.res, self.patch, self.reg, self.dn, self.view = res, patch, reg, dn, view
        if view:    # the room stays in view: every imagined step looks at the room picture again (the ball is only imagined)
            self.room_embed = nn.Linear(patch * patch, d)
            self.room_where = nn.Parameter(torch.randn((res // patch) ** 2, d) * 0.02)
            self.room_enc = nn.ModuleList(Block(d, heads) for _ in range(view))
            self.room_norm = nn.LayerNorm(d)
            self.look = nn.ModuleList(Look(d, heads) for _ in range(step))
        self.room = None    # the room in view: set by see(), read by every step() until the next see()
        if dn:      # chance: every step draws a little fresh noise, so the same picture can have several futures
            self.noise_in = nn.Linear(dn, d)
            nn.init.zeros_(self.noise_in.weight); nn.init.zeros_(self.noise_in.bias)   # starts as the plain model
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

    def see(self, img, look=True):
        """Pictures -> inner state z [N,reg,d]. With the room in view, it also puts this room in view for the steps
        that follow (look=False: a glance that leaves the room in view as it is)."""
        N, P = img.shape[0], self.patch
        x = img.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(N, -1, 3 * P * P)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=img.is_cuda):   # the heavy part; the rest stays exact
            x = self.embed(x) + self.where
            x = torch.cat([self.registers.expand(N, -1, -1), x], 1)
            m = torch.ones(N, x.shape[1], dtype=torch.bool, device=x.device)
            for b in self.enc:
                x = b(x, m)
            if self.view and look:   # the room alone (the platforms channel), read once; every step asks it again
                r = img[:, 2:3].unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(N, -1, P * P)
                r = self.room_embed(r) + self.room_where
                mr = torch.ones(N, r.shape[1], dtype=torch.bool, device=r.device)
                for b in self.room_enc:
                    r = b(r, mr)
        if self.view and look:
            r = self.room_norm(r.float())
            self.room = [lk.keys(r) for lk in self.look]
        return self.enc_norm(x[:, :self.reg].float())

    def hold_room(self):
        """Training learns one window at a time: the room in view becomes a leaf whose gradient adds up over the windows."""
        if not self.view:
            return None
        src, self.room = self.room, [kv.detach().requires_grad_() for kv in self.room]
        return src

    def release_room(self, src):
        """After the last window: pass the summed gradient on into the room encoder, once."""
        if src is not None:
            torch.autograd.backward(src, [kv.grad if kv.grad is not None else torch.zeros_like(kv) for kv in self.room])

    def repeat_room(self, c):
        """The same rooms for c copies of each inner state (copy-major, like z.repeat(c, 1, 1))."""
        if self.view:
            self.room = [kv.repeat(1, c, 1, 1, 1) for kv in self.room]

    def step(self, z, room=None):
        room = self.room if room is None else room
        x = z
        m = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=z.device)
        if self.dn:
            x = x + self.noise_in(torch.randn(z.shape[0], self.dn, device=z.device))[:, None]
        for i, b in enumerate(self.stp):
            x = b(x, m)
            if self.view:
                x = self.look[i](x, room[i])
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


def window(net, z, K, room=None):
    """Read, step, read ... K steps. Returns the K+1 readouts [K+1,N,3] and the last state."""
    ys = []
    for k in range(K + 1):
        ys.append(net.read(z))
        if k < K:
            z = net.step(z, room)
    return torch.stack(ys), z


def cache_dir():
    """Where simulated paths are kept between runs ($BALLROOM_CACHE, else the runs folder's ballroom/cache)."""
    d = os.environ.get("BALLROOM_CACHE")
    if not d:
        try:
            from oazy.data import default_dir
            d = os.path.join(default_dir("runs"), "ballroom", "cache")
        except Exception:
            d = os.path.join("ballroom", "runs", "cache")
    return d


def cached(name, make):
    """The simulator is exact and seeded, so the same rooms give the same paths: work them out once, keep them on disk."""
    path = os.path.join(cache_dir(), f"{name}-v1.npz")
    try:
        with np.load(path) as f:
            return tuple(f[f"arr_{i}"] for i in range(len(f.files)))
    except Exception:
        pass
    out = make()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path + ".tmp.npz", *out); os.replace(path + ".tmp.npz", path)
    except Exception as e:
        print(f"(could not keep {name} on disk: {e})", flush=True)
    return out


def rooms_on(dev, n, plat, seed, starts=False):
    rng = np.random.default_rng(seed)
    boxes, count, p0, v0 = sim.make_rooms(n, rng, plat)

    def make():                                   # on every CPU core
        from . import robot
        robot.land(boxes[:1], p0[:1], v0[:1])    # makes sure the pool of simulator processes is up
        parts = list(robot._pool.map(_paths, [(boxes[i:i + 2000], p0[i:i + 2000], v0[i:i + 2000]) for i in range(0, n, 2000)]))
        return np.concatenate([q for q, _ in parts], 1), np.concatenate([q for _, q in parts])
    P, stop = cached(f"rooms-{n}-{plat[0]}{plat[1]}-{seed}", make)
    t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    return (boxes, t(boxes), t(P), t(stop, torch.long)) + ((p0, v0) if starts else ())


def _paths(args):
    b, p, v = args
    P, _, stop = sim.rollout(b, p, v)
    return P.astype(np.float32), stop


def nudged_paths(boxes, p0, v0, copies, seed=1):
    """The exact simulator's own cloud: each room started `copies` times a hair away (0.002, 0.02).
    Returns paths [T+1,copies,n,2] and rest steps [copies,n]."""
    key = f"nudged-{len(p0)}-{copies}-{seed}-{hashlib.md5(p0.tobytes() + v0.tobytes()).hexdigest()[:10]}"
    return cached(key, lambda: _nudged_paths(boxes, p0, v0, copies, seed))


def _nudged_paths(boxes, p0, v0, copies, seed):
    from . import robot
    robot.land(boxes[:1], p0[:1], v0[:1])                               # makes sure the pool of simulator processes is up
    rng = np.random.default_rng(seed); n = len(p0)
    cp = np.clip(p0[None] + 0.002 * rng.standard_normal((copies, n, 2)), sim.R, 1 - sim.R).reshape(-1, 2)
    cv = (v0[None] + 0.02 * rng.standard_normal((copies, n, 2))).reshape(-1, 2)
    B = np.tile(boxes, (copies, 1, 1))
    parts = list(robot._pool.map(_paths, [(B[i:i + 4000], cp[i:i + 4000], cv[i:i + 4000]) for i in range(0, len(cp), 4000)]))
    P = np.concatenate([q for q, _ in parts], 1)
    return P.reshape(P.shape[0], copies, n, 2), np.concatenate([q for _, q in parts]).reshape(copies, n)


def learnable(boxes, p0, v0, fin, copies=8, seed=0):
    """How predictable each room is at all: the share of the exact simulator's own nudged copies
    (0.002 in position, 0.02 in speed) that rest within a ball of the true rest. Low = a knife-edge."""
    key = f"learnable-{len(p0)}-{copies}-{seed}-{hashlib.md5(p0.tobytes() + v0.tobytes()).hexdigest()[:10]}"
    return cached(key, lambda: (_learnable(boxes, p0, v0, fin, copies, seed),))[0]


def _learnable(boxes, p0, v0, fin, copies, seed):
    from .robot import land                       # the simulator on every CPU core
    rng = np.random.default_rng(seed); n = len(p0)
    cp = np.clip(p0[None] + 0.002 * rng.standard_normal((copies, n, 2)), sim.R, 1 - sim.R).reshape(-1, 2)
    cv = (v0[None] + 0.02 * rng.standard_normal((copies, n, 2))).reshape(-1, 2)
    ends = land(np.tile(boxes, (copies, 1, 1)), cp, cv)[0].reshape(copies, n, 2)
    return (np.linalg.norm(ends - fin[None], axis=-1) < 2 * sim.R).mean(0)


def evaluate(net, dev, a, rooms=3000):
    net.eval(); res = {}
    for label, plat, seed in (("2-3 platforms (like training)", (2, 3), 101), ("4-5 platforms (never seen)", (4, 5), 102)):
        boxes, tb, P, stop, p0, v0 = rooms_on(dev, rooms, plat, seed, starts=True)      # the same rooms ballroom.test uses
        fin = P[-1].cpu().numpy()
        true_odds = learnable(boxes, p0, v0, fin.astype(np.float64), copies=16, seed=7)  # the real chance that a nudged ball rests on the true spot
        shot, law, seen = [], [], []
        kinds = [("nudged pictures", 0.0)] + [(f"nudged inner state x{n:g}", n) for n in a.noise] + ([("its own chance", -1.0)] if net.dn else [])
        ends = {k: [] for k, _ in kinds}
        with torch.no_grad():
            for i in range(0, rooms, 500):
                sl = slice(i, i + 500)
                z = net.see(render(P[0, sl], P[1, sl], tb[sl], a.res))
                shot.append(net.shot(z)); law.append(roll(net, z))
                seen.append((net.read(z)[:, :2] - P[1, sl]).norm(dim=1))
                # clouds: the same start asked many times, each copy rolled to its end
                for kind, n in kinds:
                    if n < 0:   # the model's own chance: every step draws fresh noise
                        e = [roll(net, z) for _ in range(a.cloud)]
                    elif n:     # the first inner state itself is nudged
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
            odds = (torch.cat(ends[kind], 1) - torch.as_tensor(fin, device=dev)[None]).norm(dim=2).lt(2 * sim.R).float().mean(0).cpu().numpy()
            notes.append(f"  {kind}: 3/4 of the cloud agrees in {100 * m.mean():.0f}% of rooms" + (f", hits {100 * hit[m].mean():.0f}% there" if m.any() else "")
                         + (f", {100 * hit[~m].mean():.0f}% elsewhere" if (~m).any() else "")
                         + f"; its odds of the true spot are off the real odds by {100 * np.abs(odds - true_odds).mean():.0f} points")
        see_err = torch.cat(seen).median().item()
        res[label] = dict(rows=rows, sees_ball_within=see_err, clouds=notes)
        print(f"\n== {label}, {rooms} fresh rooms ==")
        print(f"{'':40s}{'mean miss':>10s}{'median':>9s}{'within a ball':>15s}{'right surface':>15s}")
        for r in rows:
            print(f"{r['model']:40s}{r['mean']:10.3f}{r['median']:9.3f}{r['within_ball']:14.1f}%{r['right_surface']:14.1f}%")
        print(f"how well it sees the ball it is shown: median error {see_err:.4f} = {see_err / (2 * sim.R):.2f} ball widths")
        print(f"when the cloud of {a.cloud} agrees, is it right? (real odds: the simulator's own cloud lands on the true spot "
              f"{100 * true_odds.mean():.0f}% of the time, and splits in {100 * (true_odds < 0.75).mean():.0f}% of rooms)"); print("\n".join(notes))
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
    ap.add_argument("--view", type=int, default=0, help="layers of a room encoder that every imagined step looks at again (0 = off)")
    ap.add_argument("--latent", type=float, default=1.0, help="weight of the agree-with-the-eye term")
    ap.add_argument("--test-rooms", type=int, default=3000)
    ap.add_argument("--windows", type=int, default=4, help="windows of --unroll steps rolled one after another in training")
    ap.add_argument("--init", default="", help="start from a saved eye.pt")
    ap.add_argument("--emotion", type=int, default=0, help="1 = learning strength = surprise x learnable x importance")
    ap.add_argument("--odds", type=int, default=0, help="copies per room: learn the spread of futures (a model with chance), 0 = off")
    ap.add_argument("--odds-rooms", type=int, default=50000, help="rooms whose simulator clouds are kept for that")
    ap.add_argument("--compile", default="", help="torch.compile mode for the rolled window: default, reduce-overhead, max-autotune")
    ap.add_argument("--cloud", type=int, default=16, help="nudged copies per room in the test")
    ap.add_argument("--noise", type=float, nargs="*", default=[0.1, 0.3], help="sizes of the nudge to the inner state")
    ap.add_argument("--out", default="ballroom/runs/eye")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", type=int, default=0, help="parallel training: short chains of this many moments, side by side (0 = off)")
    ap.add_argument("--fast-batch", type=int, default=4096, help="rooms per fast step")
    ap.add_argument("--long-every", type=int, default=8, help="with --fast: every this-many steps one long chain as before")
    ap.add_argument("--bf16", type=int, default=0, help="1 = the rolled steps in bf16 too (the eye always is)")
    ap.add_argument("--glances", type=int, default=0, help="windows whose end is checked against the eye (0 = all); fewer = less eye work")
    ap.add_argument("--bench", type=int, default=0, help="speed test: run this many steps, print the speed, save nothing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    lo, hi = map(int, a.plat.split(","))
    t0 = time.time()
    boxes, tb, P, stop, p0, v0 = rooms_on(dev, a.rooms, (lo, hi), a.seed, starts=True)
    T = P.shape[0] - 1; K = a.unroll; W = a.windows
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    net = Eye(a.res, a.patch, a.d, a.enc, a.step, reg=a.reg, dn=16 if a.odds else 0, view=a.view).to(dev)
    if a.init:
        miss = net.load_state_dict(torch.load(a.init, map_location=dev)["model"], strict=False).missing_keys
        print(f"starting from {a.init}" + (f" (new: {', '.join(miss)})" if miss else ""), flush=True)
    if a.odds:
        # learn the odds: from the start picture, its own clouds must spread like the simulator's own cloud
        t1_ = time.time(); NO = min(a.odds_rooms, a.rooms)
        PO, SO = nudged_paths(boxes[:NO], p0[:NO], v0[:NO], a.odds)
        PO = torch.as_tensor(PO, device=dev); SO = torch.as_tensor(SO, dtype=torch.long, device=dev)
        print(f"odds on: {a.odds} nudged copies of {NO:,} rooms by the simulator in {time.time() - t1_:.0f}s; "
              f"half of each batch learns the spread of futures", flush=True)
    win = window
    a.glances = min(a.glances or W, W)
    graphs = a.compile == "reduce-overhead"         # CUDA graphs: each replay overwrites the last one's outputs
    if a.compile:                # fuse the many small operations of a window; reduce-overhead also replays them as one CUDA graph
        win = torch.compile(window, mode=a.compile)
        print(f"compiling the rolled window ({a.compile}); the first steps are slow while it compiles", flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=0.01)
    print(f"rooms {a.rooms} ({lo}-{hi} platforms) {time.time() - t0:.0f}s; eye model {sum(p.numel() for p in net.parameters()):,} weights "
          f"on {dev}; {a.res}x{a.res} pictures, inner state {a.reg} tokens of {a.d}" + (", the room stays in view" if a.view else "") + f", rolled {W} x {K} steps", flush=True)
    budget = a.minutes * 60; it = 0; t0 = time.time()
    ks = torch.arange(K + 1, device=dev)
    if a.emotion:
        # learning strength = surprise x learnable x importance (João's "emotion" idea)
        #   surprise   per room and moment, the model's own recent error there; half of each batch is drawn by it
        #   learnable  the simulator's own cloud: a room where nudged copies split cannot be predicted, so it pulls less
        #   importance the steps where the ball has come to rest count double: that is the answer
        t1_ = time.time()
        learn = torch.as_tensor(learnable(boxes, p0, v0, P[-1].cpu().numpy().astype(np.float64)), dtype=torch.float32, device=dev)
        nb = (T - W * K - 2) // K + 1                                   # moments per room, one per window length
        surp = torch.ones(a.rooms, nb, device=dev)
        last = ((stop + 3).clamp(max=T - W * K - 2) // K)              # the last useful moment of each room
        valid = torch.arange(nb, device=dev)[None, :] <= last[:, None]
        print(f"emotion on: {100 * (learn < 0.5).float().mean().item():.0f}% of rooms are knife-edges (their own cloud splits), "
              f"measured in {time.time() - t1_:.0f}s", flush=True)
    def fast_step():
        """Learn many moments at once: from what it sees at moment t, imagine only a.fast moments, and check
        every one of them against the truth and against what the eye would see there. Short chains run side by side
        on the GPU, so far more moments are learned per second than with one long chain."""
        S, B2 = a.fast, a.fast_batch
        r = torch.randint(0, a.rooms, (B2,), device=dev)
        t = (torch.rand(B2, device=dev) * (stop[r] + 3).float()).long().clamp(max=T - S - 2)
        t = torch.where(torch.rand(B2, device=dev) < 0.1, torch.zeros_like(t), t)
        br = tb[r]
        z = net.see(render(P[t, r], P[t + 1, r], br, a.res))
        room_src = net.hold_room()
        l_shot = F.smooth_l1_loss(net.shot(z) / 0.3, P[stop[r], r] / 0.3, beta=0.1)
        l_pos = l_rest = l_lat = 0.0
        for k in range(S + 1):
            y = net.read(z); now = t + 1 + k
            l_pos = l_pos + F.smooth_l1_loss(y[:, :2] / 0.05, P[now, r] / 0.05, beta=0.2) / (S + 1)
            l_rest = l_rest + F.binary_cross_entropy_with_logits(y[:, 2], (now >= stop[r]).float()) / (S + 1)
            if k:
                with torch.no_grad():
                    z_eye = net.see(render(P[now - 1, r], P[now, r], br, a.res), look=False)
                l_lat = l_lat + F.mse_loss(z, z_eye) / S
            if k < S:
                z = net.step(z)
        loss = l_pos + 0.3 * l_rest + a.latent * l_lat + l_shot
        loss.backward()
        net.release_room(room_src)
        return l_shot, dict(pos=l_pos.detach(), rest=l_rest.detach(), lat=l_lat.detach(), loss=loss.detach(), odds=0.0)

    while (it < a.iters) if a.iters else (time.time() - t0 < budget):
        frac = it / a.iters if a.iters else (time.time() - t0) / budget
        lr = a.lr * min(1.0, (it + 1) / 500) * (0.5 + 0.5 * np.cos(np.pi * min(1.0, frac))) + 1e-5
        for g in opt.param_groups:
            g["lr"] = lr
        if a.fast and it % a.long_every:     # most steps: many short chains side by side (parallel), see fast_step
            opt.zero_grad(set_to_none=True)
            l_shot, tot = fast_step()
        else:                                 # every long_every-th step: the long chain that keeps long imaginations on track
            r = torch.randint(0, a.rooms, (a.batch,), device=dev)
            # a moment on the true path: anywhere before rest (a quarter of the time the very start)
            t = (torch.rand(a.batch, device=dev) * (stop[r] + 3).float()).long().clamp(max=T - W * K - 2)
            t = torch.where(torch.rand(a.batch, device=dev) < 0.25, torch.zeros_like(t), t)
            if a.emotion:                                                   # the other half: where it hurt, and can be learned
                h = a.batch // 2
                pri = (surp * (0.1 + learn[:, None]) * valid).flatten()
                idx = torch.multinomial(pri, h, replacement=True)
                r = torch.cat([r[:a.batch - h], idx // nb]); b = idx % nb
                tb_ = (b * K + torch.randint(0, K, (h,), device=dev)).clamp(max=T - W * K - 2)
                t = torch.cat([t[:a.batch - h], torch.minimum(tb_, (stop[idx // nb] + 3).clamp(max=T - W * K - 2))])
            br = tb[r]
            z = net.see(render(P[t, r], P[t + 1, r], br, a.res))
            room_src = net.hold_room()
            l_shot = F.smooth_l1_loss(net.shot(z) / 0.3, P[stop[r], r] / 0.3, beta=0.1)
            opt.zero_grad(set_to_none=True)
            tot = dict(pos=0.0, rest=0.0, lat=0.0, loss=0.0)
            for w in range(W):                                                 # each window frees its memory before the next
                t1 = t + 1 + w * K                                             # a long roll, learned one window at a time
                tgt = P[t1[None, :] + ks[:, None], r[None, :]]                 # [K+1,B,2] true positions
                at_rest = (t1[None, :] + ks[:, None] >= stop[r][None, :]).float()
                if graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.bf16) and dev.startswith("cuda")):
                    ys, z = win(net, z, K, net.room)                       # bf16: the imagining in half precision too
                ys, z = ys.float(), z.float()
                if a.emotion:
                    e = F.smooth_l1_loss(ys[..., :2] / 0.05, tgt / 0.05, beta=0.2, reduction="none").mean(2)   # [K+1,B]
                    l_pos = (e * (1 + at_rest)).sum() / (1 + at_rest).sum()          # importance: the resting answer counts double
                    with torch.no_grad():                                          # remember the surprise of this moment
                        bi = ((t1 - 1) // K).clamp(max=nb - 1)
                        surp[r, bi] = 0.7 * surp[r, bi] + 0.3 * e.mean(0).clamp(max=10)
                else:
                    l_pos = F.smooth_l1_loss(ys[..., :2] / 0.05, tgt / 0.05, beta=0.2)
                l_rest = F.binary_cross_entropy_with_logits(ys[..., 2], at_rest)
                if w >= W - a.glances:                                         # what the eye sees at that later moment
                    with torch.no_grad():
                        z_eye = net.see(render(P[t1 + K - 1, r], P[t1 + K, r], br, a.res), look=False)
                    l_lat = F.mse_loss(z, z_eye) * W / a.glances
                else:
                    l_lat = torch.zeros((), device=dev)
                part = (l_pos + 0.3 * l_rest + a.latent * l_lat) / W + (l_shot if w == 0 else 0.0)
                part.backward()
                z = z.detach().clone() if graphs else z.detach()               # (a CUDA graph reuses its output memory) the next window starts from the rolled state, never from the eye
                for k_, v_ in (("pos", l_pos), ("rest", l_rest), ("lat", l_lat), ("loss", part * W)):
                    tot[k_] = tot[k_] + v_.detach() / W                        # kept on the GPU: no waiting until a line is printed
            net.release_room(room_src)
            if a.odds:          # the other half: whole clouds from the very start, scored as distributions (energy score)
                C = a.odds; nbo = max(1, a.batch // (2 * C)); ro = torch.randint(0, NO, (nbo,), device=dev)
                zo = net.see(render(P[0, ro], P[1, ro], tb[ro], a.res)).repeat(C, 1, 1)          # [C*nbo,reg,d], copy-major
                net.repeat_room(C); room_o = net.hold_room()
                for w in range(W):
                    if graphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    ys, zo = win(net, zo, K, net.room)
                    when = 1 + w * K + ks                                                       # [K+1] moments
                    X = ys[..., :2].view(K + 1, C, nbo, 2)
                    Y = PO[when][:, :, ro]                                                      # [K+1,C,nbo,2] the simulator's cloud
                    dist = lambda A, B: ((A[:, :, None] - B[:, None]) ** 2).sum(-1).add(1e-10).sqrt().mean((1, 2))
                    l_o = ((dist(X, Y) - 0.5 * dist(X, X)) / 0.05).mean()                     # a proper score: only the true spread wins
                    at_o = (when[:, None, None] >= SO[:, ro][None]).float().mean(1)            # [K+1,nbo] share of the cloud at rest
                    l_ro = F.binary_cross_entropy_with_logits(ys[..., 2].view(K + 1, C, nbo), at_o[:, None].expand(-1, C, -1))
                    ((l_o + 0.3 * l_ro) / W).backward()
                    zo = zo.detach().clone() if graphs else zo.detach()
                    tot["odds"] = tot.get("odds", 0.0) + l_o.detach() / W
                net.release_room(room_o)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        it += 1
        if a.bench:                  # speed test: time the second half only (the first half warms up and compiles)
            if it == a.bench // 2:
                torch.cuda.synchronize() if dev.startswith("cuda") else None; tb0 = time.time()
            elif it == a.bench:
                torch.cuda.synchronize() if dev.startswith("cuda") else None
                sps = (a.bench - a.bench // 2) / (time.time() - tb0)
                per = ((a.long_every - 1) * a.fast_batch * (a.fast + 1) + a.batch * W * K) / a.long_every if a.fast else a.batch * W * K
                mem = torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else 0
                print(f"BENCH batch {a.batch:5d} compile {a.compile or 'off':8s} view {a.view} bf16 {a.bf16} glances {a.glances} res {a.res}: {sps:5.2f} steps/s = "
                      f"{sps * per:8.0f} moments learned/s" + (f" (fast {a.fast} x {a.fast_batch}, long chain every {a.long_every})" if a.fast else "")
                      + f", peak memory {mem:.1f} GB", flush=True)
                return
        if it % 100 == 0:
            el = time.time() - t0
            print(f"it {it} loss {float(tot['loss']):.4f} pos {float(tot['pos']):.4f} rest {float(tot['rest']):.4f} shot {l_shot.item():.4f} "
                  f"agree {float(tot['lat']):.4f} " + (f"odds {float(tot['odds']):.4f} " if a.odds else "") + f"{el:.0f}s ({it / el:.1f} steps/s)", flush=True)
    torch.save({"model": net.state_dict(), "cfg": net.cfg, "plat": [lo, hi], "steps": it}, os.path.join(a.out, "eye.pt"))
    print(f"saved {a.out}/eye.pt after {it} steps", flush=True)
    res = evaluate(net, dev, a, a.test_rooms)
    json.dump(res, open(os.path.join(a.out, "eye.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
