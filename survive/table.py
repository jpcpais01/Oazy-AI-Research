"""Creature two: a robot arm on a tabletop (drop-in for survive.world; select with SURVIVE_WORLD=table).

The world: a 1 m x 1 m table with a low rim, one ball (r 3 cm) and an open box, "the basket"
(14 cm inside, 5 cm walls). Simple but real physics: gravity, bounces, rolling friction, walls,
the ball pushed by the claw plates, a friction grasp when both inner faces squeeze it.

The robot: a fixed base at the near edge, two eyes on a head above it looking down at the table
(the same ray caster as creature one, pitched down), and one arm:
    shoulder yaw, shoulder pitch, elbow, wrist pitch, and a claw of two flat plates.
Muscles (5, in -1..1): the four joints are speeds (relaxed = the arm holds still); the claw is a
spring that is open when relaxed and closes as its muscle pulls (pulling past the ball = squeezing). Body feelings (senses): energy, health, 4 touch sensors (inner and
outer face of each plate), the 4 joint angles and the claw opening it feels.

Its only drive is the same as creature one's: wellbeing = (energy + health) / 2. Energy drains
slowly, faster when the arm works hard; at 0 the life ends. Slamming the claw into the table or
basket hurts (health). The world has ONE rule that feeds it: a ball that lands in the basket gives
energy (like a feeder), and a new ball appears. Nothing tells it to grasp, lift or carry.

To make the first discovery possible, the world (never the reward) starts kind: some lives start
with the ball already in its claw, sometimes right above the basket; some new balls appear just
under the claw. Test worlds ("test") use the same mix; "hard" never starts it with a ball.
"""
from __future__ import annotations

import math
import os

import torch

# ---- the contract with survive.train / render ---------------------------------------------------
ACT = 5
SENSES = 11
DEATH_CAUSES = ["cap", "hunger", "pain"]
NF = 1
DT = 0.1
SUB = 5                      # physics substeps per step
MAX_STEPS = 1200             # 120 s life cap
LIGHT = 0
L = 1.0                      # the table (the renderer's "room")
WALL_H = 0.04                # the rim
EYE_H = 0.0                  # the camera height lives in pos[:, 2]
EYE_SEP = 0.03               # half the distance between the eyes
CAM = (0.5, -0.22, 0.82)     # a tall head behind the base: the arm can't reach up in front of its eyes (t2 parked the claw there)
CAM_PITCH = math.radians(47)
FOV_DEG = 70                 # narrower eyes than creature one: it looks at a table, not a room
TILE = 0.1                   # the table top: 10 cm tiles (depth cues)
OUTSIDE = (0.30, 0.30, 0.34) # the floor far below, beyond the table

# ---- body -----------------------------------------------------------------------------------------
BASE = (0.5, 0.06)
SH_Z = 0.12                  # shoulder height (on a short post)
L1, L2, L3, LF = 0.46, 0.44, 0.05, 0.07     # upper arm, forearm, hand, finger plates (was .30/.28: half the 1 m table was out of reach)
PT, PW = 0.010, 0.040        # plate thickness, plate width
G_MAX = max(0.11, 2 * float(os.environ.get("BALL_R", "0.035")) + 0.04)   # widest claw opening (gap between inner faces)
LO = torch.tensor([-1.75, -0.35, 0.0, -1.9, 0.0])
HI = torch.tensor([1.75, 1.4, 2.7, 1.9, G_MAX])      # shoulder lifts to 80 deg: the long arm still stays >= 33 cm from the eyes
RATE = torch.tensor([0.15, 0.15, 0.18, 0.3, 0.025])   # rad/step; slower than before for the longer arm (tip ~1.4 m/s)   # per step
# ---- things ---------------------------------------------------------------------------------------
BALL_R = float(os.environ.get("BALL_R", "0.035"))   # 0.035 = 2-3 px far away in 64 px eyes; 0.05 doubles its area
B_IN, B_WALL, B_H = max(0.07, float(os.environ.get("BALL_R", "0.035")) + 0.035), 0.01, 0.05        # basket inner half size, wall thickness, height
BALL_E = 1.0                 # a ball in the basket fills the tank (0.4 paid less than fetching it cost; at 0.7 a scripted arm only broke even)
DRAIN = 1 / 600              # per step (60 s from full, resting)
EFFORT = 0.001               # extra drain at full joint speed (all joints)
GRAV, REST, ROLL = 9.81, 0.35, 2.5
P_HELD_OVER = float(os.environ.get("P_HELD_OVER", "0.2"))   # life starts: ball in the claw above the basket
P_HELD = float(os.environ.get("P_HELD", "0.1"))             # ... ball in the claw anywhere
NEAR = float(os.environ.get("NEAR", "0"))                   # share of balls that appear right beside the basket (a gentler world;
                                                            # training may fade it as the creature succeeds: World.near)
P_AROUND = float(os.environ.get("P_AROUND", "0"))   # life starts: ball on the table between the OPEN plates (one squeeze grabs it)
MUSCLE_K = float(os.environ.get("MUSCLE_K", "1"))   # muscles follow the command with a lag: m += K*(a-m); 1 = no lag
P_UNDER = 0.4                # a new ball appears just under the claw


def set_gaze(deg):
    pass


def set_light(n):
    pass


def set_arm_seen(on):
    pass


def set_body(*a, **k):
    pass


def _dirs(th0):
    """Horizontal forward (yaw 0 = +y) and side vectors."""
    f = torch.stack([-torch.sin(th0), torch.cos(th0)], -1)
    s = torch.stack([torch.cos(th0), torch.sin(th0)], -1)
    return f, s


def fk(q):
    """q: (..., 5) -> dict of 3D points (..., 3) and frames for the plates."""
    th0, a1 = q[..., 0], q[..., 1]
    a2 = a1 - q[..., 2]
    a3 = a2 - q[..., 3]
    f, s = _dirs(th0)
    dev = q.device
    base = torch.tensor(BASE, device=dev).expand(*q.shape[:-1], 2)

    def pt(r, z):
        return torch.cat([base + f * r[..., None], z[..., None]], -1)
    r1, z1 = L1 * torch.cos(a1), SH_Z + L1 * torch.sin(a1)
    r2, z2 = r1 + L2 * torch.cos(a2), z1 + L2 * torch.sin(a2)
    r3, z3 = r2 + L3 * torch.cos(a3), z2 + L3 * torch.sin(a3)
    r4, z4 = r3 + LF * torch.cos(a3), z3 + LF * torch.sin(a3)
    sh = pt(torch.zeros_like(r1), torch.full_like(r1, SH_Z))
    el, wr, root, tip = pt(r1, z1), pt(r2, z2), pt(r3, z3), pt(r4, z4)
    hand = torch.stack([f[..., 0] * torch.cos(a3), f[..., 1] * torch.cos(a3), torch.sin(a3)], -1)   # along the plates
    side = torch.cat([s, torch.zeros_like(s[..., :1])], -1)
    up = torch.cross(hand, side, dim=-1)
    g = q[..., 4]
    mid = (root + tip) / 2
    off = (g / 2 + PT / 2)[..., None] * side
    return dict(sh=sh, el=el, wr=wr, root=root, tip=tip, hand=hand, side=side, up=up,
                pl=mid - off, pr=mid + off, grip=root + hand * (LF * 0.55))


def ik_pitch(xyz, g, phi):
    """Joint angles that put the grip point at xyz with the hand pitched phi (rad, -pi/2 = straight down)."""
    base = torch.tensor(BASE, device=xyz.device)
    d = xyz[:, :2] - base
    th0 = torch.atan2(-d[:, 0], d[:, 1])
    reach = L3 + LF * 0.55
    R = d.norm(dim=-1) - reach * math.cos(phi)
    Z = xyz[:, 2] - reach * math.sin(phi) - SH_Z
    D = ((R ** 2 + Z ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)).clamp(-1, 1)
    el = torch.acos(D)
    a1 = torch.atan2(Z, R) + torch.atan2(L2 * torch.sin(el), L1 + L2 * torch.cos(el))
    th3 = a1 - el - phi
    return torch.stack([th0, a1, el, th3, torch.full_like(th0, g)], -1)


def ik_down(xyz, g):
    """Grip at xyz, hand as close to straight down as the joint limits allow (tries -90 deg, then flatter)."""
    lo, hi = LO.to(xyz.device), HI.to(xyz.device)
    best = ik_pitch(xyz, g, -math.pi / 2)
    done = ((best >= lo) & (best <= hi)).all(-1) & ((fk(best)["grip"] - xyz).norm(dim=-1) < 0.01)
    for deg in (-75, -60, -45, -30, -15):
        q = ik_pitch(xyz, g, math.radians(deg))
        fit = ((q >= lo) & (q <= hi)).all(-1) & ((fk(q)["grip"] - xyz).norm(dim=-1) < 0.01)
        use = fit & ~done
        best = torch.where(use[:, None], q, best)
        done = done | fit
    return best


def obb_contact(p, c, ax, half):
    """Sphere centre p (B,3) vs an oriented box (centre c, axes ax = 3 x (B,3), half extents).
    Returns the distance from p to the box, the unit normal (box -> p) and p in box coordinates."""
    d = p - c
    loc = torch.stack([(d * a).sum(-1) for a in ax], -1)
    hh = torch.tensor(half, device=p.device)
    cl = torch.maximum(torch.minimum(loc, hh), -hh)
    diff = loc - cl
    dist = diff.norm(dim=-1)
    inside = dist < 1e-7
    # inside the box: push out along the side axis (the plates are thin)
    sgn = torch.where(loc[..., 1] >= 0, 1.0, -1.0)
    diff = torch.where(inside[..., None], torch.stack([torch.zeros_like(sgn), sgn, torch.zeros_like(sgn)], -1), diff)
    nl = diff / diff.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    n = nl[..., :1] * ax[0] + nl[..., 1:2] * ax[1] + nl[..., 2:3] * ax[2]
    dist = torch.where(inside, -(hh[1] - loc[..., 1].abs()), dist)
    return dist, n, loc


def held_pos(P, off):
    """Where a held ball sits: fixed in the claw's frame (where it was caught), between the plates."""
    mid = (P["pl"] + P["pr"]) / 2
    return mid + off[..., 0:1] * P["hand"] + off[..., 1:2] * P["side"] + off[..., 2:3] * P["up"]


def link_points(P):
    """Spheres along the upper arm, forearm and hand that push the ball (r LINK_R)."""
    pts = []
    for a_, b_, k in ((P["sh"], P["el"], 6), (P["el"], P["wr"], 6), (P["wr"], P["root"], 1)):
        for i in range(k):
            pts.append(a_ + (b_ - a_) * ((i + 0.5) / k))
    return torch.stack(pts, -2)


LINK_R = 0.02
GRIP_OFF = (LF * 0.05, 0.0, 0.0)


class Mode:
    def __init__(self, name):
        self.name = name
        self.curriculum = name != "hard"
        self.far = name == "hard"


def _boxes(p, v, mn, mx):
    """Ball (centre p, vel v) against solid boxes mn/mx (B,K,3): pushed out, bounced. A centre that got
    inside a box leaves along the axis of least penetration (not straight up: the post is taller than the ball)."""
    for j in range(mn.shape[1]):
        lo, hi = mn[:, j], mx[:, j]
        qn = torch.maximum(torch.minimum(p, hi), lo)
        d = p - qn
        dist = d.norm(dim=-1)
        inside = dist < 1e-7
        pen = torch.cat([p - lo, hi - p], -1)                    # B,6: distance to each face
        pen[:, 2] = 1e9                                          # never out through the bottom (the table)
        k = pen.argmin(-1)
        ax, sg = k % 3, torch.where(k < 3, -1.0, 1.0)
        n_in = torch.nn.functional.one_hot(ax, 3).float() * sg[:, None]
        face = torch.where(k[:, None] < 3, lo.gather(1, ax[:, None]), hi.gather(1, ax[:, None]))
        q_in = p.clone().scatter(1, ax[:, None], face)
        n = torch.where(inside[:, None], n_in, d / dist.clamp(min=1e-7)[:, None])
        qn = torch.where(inside[:, None], q_in, qn)
        hit = inside | (dist < BALL_R)
        p = torch.where(hit[:, None], qn + n * BALL_R, p)
        vn = (v * n).sum(-1)
        v = torch.where((hit & (vn < 0))[:, None], v - (1 + REST) * vn[:, None] * n, v)
    return p, v


class Demo:
    """A scripted pick-and-place arm (for optional demonstrations: it shows what the body can do;
    the creature's pay is unchanged). Phases: above ball, down, squeeze, up, over basket, open."""

    def __init__(self, B, dev, noise=0.2):
        self.phase = torch.zeros(B, dtype=torch.long, device=dev)
        self.timer = torch.zeros(B, device=dev)
        self.noise = noise

    def reset(self, idx):
        self.phase[idx] = 0
        self.timer[idx] = 0

    def act(self, w):
        B, dev = w.B, w.dev
        b = w.ball
        above = torch.cat([b[:, :2], torch.full((B, 1), 0.14, device=dev)], -1)
        low = torch.cat([b[:, :2], torch.full((B, 1), BALL_R, device=dev)], -1)
        bk = torch.cat([w.basket, torch.full((B, 1), 0.16, device=dev)], -1)
        ph = self.phase
        tgt = torch.where((ph == 0)[:, None], above, torch.where((ph <= 2)[:, None], low,
                          torch.where((ph == 3)[:, None], above, bk)))
        q = ik_down(tgt, G_MAX)
        a = ((q - w.q) / w.rate).clamp(-1, 1)
        a[:, 4] = torch.where((ph >= 2) & (ph <= 4), torch.ones_like(a[:, 4]), -torch.ones_like(a[:, 4]))
        a = (a + self.noise * torch.randn_like(a)).clamp(-1, 1)
        err = (fk(w.q)["grip"] - tgt).norm(dim=-1)
        self.timer += 1
        t = self.timer
        nxt = ((ph == 0) & (err < 0.03)) | ((ph == 1) & (err < 0.02)) | ((ph == 2) & (t > 3)) | \
              ((ph == 3) & (err < 0.04)) | ((ph == 4) & (err < 0.03)) | ((ph == 5) & (t > 4)) | (t > 40)
        self.phase = torch.where(nxt, (ph + 1) % 6, ph)
        self.timer = torch.where(nxt, torch.zeros_like(t), t)
        lost = (self.phase >= 3) & (self.phase <= 4) & ~w.held
        self.phase = torch.where(lost, torch.zeros_like(self.phase), self.phase)
        return a


class World:
    def __init__(self, B, dev, mode="train", seed=0, compile=False, rooms=0):
        self.B, self.dev = B, dev
        self.mode = Mode(mode)
        self.g = torch.Generator(device=dev)
        self.g.manual_seed(int(seed) * 7919 + {"train": 0, "test": 1, "hard": 2}.get(mode, 3))
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)
        self.lo, self.hi, self.rate = LO.to(dev), HI.to(dev), RATE.to(dev)
        self.q = z(B, 5)
        self.qv = z(B, 5)                               # the arm's real motion last step
        self.m = z(B, 5)                                # muscle activation (follows the command with a lag)
        self.ball, self.bv = z(B, 3), z(B, 3)
        self.held = z(B, dt=torch.bool)
        self.hold_off = torch.tensor(GRIP_OFF, device=dev).expand(B, 3).clone()   # the held ball in the claw's frame
        self.basket = z(B, 2)
        self.energy, self.health = z(B), z(B)
        self.touch = z(B, 4)
        self.t = z(B, dt=torch.long)
        self.dead = z(B, dt=torch.bool)
        self.cause = z(B, dt=torch.long)
        self.dying = z(B, dt=torch.long)
        self.score_t = z(B, dt=torch.long)             # >0: the ball sits in the basket, a new one soon
        self.near = NEAR if self.mode.name == "train" else 0.0     # balls beside the basket (train worlds only, so tests stay comparable; training may fade it)
        self.ball_col = z(B, 3)
        self.basket_col = z(B, 3)
        self.floor_col = z(B, 2, 3)
        self.wall_col = z(B, 4, 3)
        self.arm_col = z(B, 3)
        self.senses = z(B, SENSES)
        self.pos = torch.tensor(CAM, device=dev).expand(B, 3).clone()
        self.head = torch.full((B,), math.pi / 2, device=dev)
        self.pitch = torch.full((B,), CAM_PITCH, device=dev)
        self.reset(torch.arange(B, device=dev))

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    # ------------------------------------------------------------------ generation
    def _reach_xy(self, n, rmin, rmax, yaw):
        r = rmin + self._rand(n) * (rmax - rmin)
        th = (self._rand(n) * 2 - 1) * yaw
        f, _ = _dirs(th)
        xy = torch.tensor(BASE, device=self.dev) + f * (r[:, None] + 0.0)
        return xy.clamp(0.08, L - 0.08)                  # on the table, clear of the rim

    def _pose(self, n):
        """A random arm pose with the claw 8-25 cm above the table, pointing down-ish."""
        q = torch.zeros(n, 5, device=self.dev)
        q[:, 0] = (self._rand(n) * 2 - 1) * 1.2
        q[:, 1] = 0.2 + self._rand(n) * 0.9
        q[:, 2] = 0.9 + self._rand(n) * 1.2
        q[:, 3] = 0.3 + self._rand(n) * 0.9
        q[:, 4] = G_MAX * (0.6 + 0.4 * self._rand(n))
        return q

    def reset(self, idx):
        if len(idx) == 0:
            return
        n, dev = len(idx), self.dev
        far = self.mode.far
        self.basket[idx] = self._reach_xy(n, 0.50 if far else 0.32, 0.82 if far else 0.70, 1.3)
        q = self._pose(n)
        # the arm must clear the table; redraw bad poses a few times, else a safe default
        for _ in range(4):
            bad = ~self._ok(q, idx)
            if not bad.any():
                break
            q[bad] = self._pose(int(bad.sum()))
        bad = ~self._ok(q, idx)
        q[bad] = torch.tensor([0.0, 0.8, 1.3, 0.5, G_MAX], device=dev)   # claw 13 cm above the table (the old one had a plate 9 mm inside it)
        self.q[idx] = q
        self.qv[idx] = 0
        self.m[idx] = 0
        self.held[idx] = False
        self.bv[idx] = 0
        self._new_ball(idx, start=True)
        self.energy[idx] = 0.8 + 0.2 * self._rand(n)
        self.health[idx] = 1.0
        self.t[idx] = 0
        self.dead[idx] = False
        self.cause[idx] = 0
        self.score_t[idx] = 0
        self.touch[idx] = 0
        jit = lambda *s: 1 + (self._rand(n, *s) * 2 - 1) * 0.08
        self.ball_col[idx] = torch.tensor([1.0, 0.25, 0.12], device=dev) * jit(3)
        self.basket_col[idx] = torch.tensor([0.45, 0.55, 0.85], device=dev) * jit(3)
        self.floor_col[idx] = torch.tensor([[0.80, 0.72, 0.58], [0.76, 0.68, 0.54]], device=dev) * jit(1, 3)
        self.wall_col[idx] = torch.tensor([0.55, 0.45, 0.35], device=dev) * jit(4, 3)
        self.arm_col[idx] = torch.tensor([0.22, 0.26, 0.45], device=dev) * jit(3)   # dark blue: stands out on the beige table (light grey nearly vanished)
        self._sense()

    def _new_ball(self, idx, start=False):
        """Place a ball for envs idx: on the table, under the claw, or (life start) in the claw."""
        n, dev = len(idx), self.dev
        u = self._rand(n)
        cur = self.mode.curriculum
        P = fk(self.q[idx])
        held = torch.zeros(n, dtype=torch.bool, device=dev)
        if start and cur:
            over = u < P_HELD_OVER
            held = u < P_HELD_OVER + P_HELD
            # "over": move the basket under the claw (if the claw is above the table, inside the reach)
            gxy = P["grip"][:, :2].clamp(B_IN + B_WALL + 0.02, L - B_IN - B_WALL - 0.02)
            ok_over = over & (P["grip"][:, 2] > B_H + BALL_R + 0.02)
            b = self.basket[idx]
            self.basket[idx] = torch.where(ok_over[:, None], gxy, b)
        xy = self._reach_xy(n, 0.25, 0.85, 1.4)
        if self.near > 0:
            # a gentler world: some balls appear just outside the basket wall (3-12 cm gap), on the table
            ang = self._rand(n) * 2 * math.pi
            d = B_IN + B_WALL + BALL_R + 0.03 + self._rand(n) * 0.09
            nxy = (self.basket[idx] + d[:, None] * torch.stack([ang.cos(), ang.sin()], -1)).clamp(0.08, L - 0.08)
            r = (nxy - torch.tensor(BASE, device=dev)).norm(dim=-1)
            ok = (self._rand(n) < self.near) & (r > 0.25) & (r < 0.85)
            xy = torch.where(ok[:, None], nxy, xy)
        # keep loose balls out of the basket footprint
        for _ in range(4):
            inb = ((xy - self.basket[idx]).abs() < B_IN + B_WALL + BALL_R + 0.02).all(-1)
            if not inb.any():
                break
            xy = torch.where(inb[:, None], self._reach_xy(n, 0.25, 0.85, 1.4), xy)
        if cur and not start:
            under = (self._rand(n) < P_UNDER) & (P["grip"][:, 2] > 0.02)
            gxy = P["grip"][:, :2].clamp(0.05, L - 0.05)
            inb = ((gxy - self.basket[idx]).abs() < B_IN + B_WALL + BALL_R + 0.02).all(-1)
            xy = torch.where((under & ~inb)[:, None], gxy, xy)
        pos = torch.cat([xy, torch.full((n, 1), BALL_R, device=dev)], -1)
        if start and cur and P_AROUND > 0:
            # the open claw already around the ball on the table (if that pose is reachable and clear)
            around = (u >= P_HELD_OVER + P_HELD) & (u < P_HELD_OVER + P_HELD + P_AROUND)
            qa = ik_down(pos, G_MAX)
            use = torch.zeros_like(around)
            for _ in range(6):                     # near balls (< 30 cm) can't be reached hand-down: redraw farther
                fits = ((qa >= self.lo) & (qa <= self.hi)).all(-1) & ((fk(qa)["grip"] - pos).norm(dim=-1) < 0.01)
                fits = fits & self._ok(qa, idx) & ~(((pos[:, :2] - self.basket[idx]).abs() < B_IN + B_WALL + BALL_R + 0.02).all(-1))
                use = use | (around & fits)
                redo = around & ~use
                if not redo.any():
                    break
                p2 = torch.cat([self._reach_xy(n, 0.30, 0.80, 1.4), torch.full((n, 1), BALL_R, device=dev)], -1)
                pos = torch.where(redo[:, None], p2, pos)
                qa = torch.where(redo[:, None], ik_down(pos, G_MAX), qa)
            self.q[idx] = torch.where(use[:, None], qa, self.q[idx])
        off0 = torch.tensor(GRIP_OFF, device=dev).expand(n, 3)
        pos = torch.where(held[:, None], held_pos(P, off0), pos)
        self.hold_off[idx] = off0
        self.ball[idx] = pos
        self.bv[idx] = 0
        self.held[idx] = held
        q = self.q[idx]
        q[:, 4] = torch.where(held, torch.full_like(q[:, 4], 2 * BALL_R - 0.004), q[:, 4])
        self.q[idx] = q

    # ------------------------------------------------------------------ geometry
    def _walls(self, bxy, post=True):
        """The basket's 4 walls (+ the arm's post, solid for the ball and the arm) as AABBs: (B,4|5,3) min and max."""
        dev = bxy.device
        o, w, h = B_IN + B_WALL / 2, B_WALL / 2, B_H
        c = torch.tensor([[-o, 0], [o, 0], [0, -o], [0, o]], device=dev)
        hx = torch.tensor([[w, B_IN + B_WALL], [w, B_IN + B_WALL], [B_IN + B_WALL, w], [B_IN + B_WALL, w]], device=dev)
        cen = bxy[:, None] + c
        mn = torch.cat([cen - hx, torch.zeros(*cen.shape[:2], 1, device=dev)], -1)
        mx = torch.cat([cen + hx, torch.full((*cen.shape[:2], 1), h, device=dev)], -1)
        if post:
            n = bxy.shape[0]
            mn = torch.cat([mn, torch.tensor([BASE[0] - 0.03, BASE[1] - 0.03, 0.0], device=dev).expand(n, 1, 3)], 1)
            mx = torch.cat([mx, torch.tensor([BASE[0] + 0.03, BASE[1] + 0.03, SH_Z - 0.02], device=dev).expand(n, 1, 3)], 1)
        return mn, mx

    def _arm_points(self, P):
        """Points that must stay above the table and outside the basket walls (B,K,3)."""
        lo = -P["up"] * (PW / 2)
        pts = [P["el"], P["wr"], P["root"], P["tip"]]
        for c in (P["pl"], P["pr"]):
            for a in (-1, 1):
                pts.append(c + P["hand"] * (a * LF / 2) + lo)
                pts.append(c + P["hand"] * (a * LF / 2) - lo)
        return torch.stack(pts, 1)

    def _ok(self, q, idx=None):
        P = fk(q)
        pts = self._arm_points(P)
        ok = (pts[..., 2] > 0.003).all(-1)
        inside_table = ((pts[..., :2] > 0.0) & (pts[..., :2] < L)).all(-1)    # beyond the rim: no table under it
        ok = ok | ~inside_table.any(-1) & False
        bxy = self.basket if idx is None else self.basket[idx]
        mn, mx = self._walls(bxy)
        inw = ((pts[:, :, None] > mn[:, None] - 0.004) & (pts[:, :, None] < mx[:, None] + 0.004)).all(-1).any(-1).any(-1)
        ok = ok & ~inw
        if idx is None:      # a ball in the claw is solid too: it can't be pushed into the table or a wall
            bp = held_pos(P, self.hold_off)
            qn = torch.maximum(torch.minimum(bp[:, None], mx), mn)
            into = ((bp[:, None] - qn).norm(dim=-1) < BALL_R - 0.004).any(-1) | (bp[:, 2] < BALL_R - 0.003)
            ok = ok & ~(self.held & into)
        return ok

    # ------------------------------------------------------------------ step
    def step(self, a):
        B, dev = self.B, self.dev
        a = a.clamp(-1, 1)
        alive = ~self.dead
        # a real muscle can't flip every 0.1 s: activation follows the command with a lag (body, not shaping)
        self.m = self.m + MUSCLE_K * (a - self.m)
        a = self.m
        # muscles: the 4 joints are speeds (relaxed = the arm holds its pose); the claw is a spring that
        # stays open when relaxed (a <= 0) and closes as the muscle pulls (a = 1: fully shut, a firm squeeze)
        target = (self.q + a * self.rate).clamp(self.lo, self.hi)
        target[:, 4] = G_MAX * (1 - a[:, 4].clamp(min=0))
        dq = (target - self.q).clamp(-self.rate, self.rate) * alive[:, None]
        # the ball blocks the claw from closing while it is between the plates
        q_new = self.q + dq
        q_new = q_new.clamp(self.lo, self.hi)
        q_new[:, 4] = torch.where(self.held, q_new[:, 4].clamp(min=2 * BALL_R - 0.004), q_new[:, 4])
        # solid table and basket: try the full move, then half, then none (per env)
        q_half = (self.q + q_new) / 2
        ok_full, ok_half = self._ok(q_new), self._ok(q_half)
        blocked = ~ok_full
        q_go = torch.where(ok_full[:, None], q_new, torch.where(ok_half[:, None], q_half, self.q))
        # the claw opening is never blocked by the table (it opens sideways)
        q_go[:, 4] = q_new[:, 4]
        # a blow hurts: the arm was really moving fast (last step) and now hits something; steady pushing doesn't
        speed_hit = (self.qv[:, :4].abs() / self.rate[:4]).max(-1).values
        pain = blocked & ~ok_half & (speed_hit > 0.7)
        q0, q1 = self.q, q_go
        self.qv = q_go - self.q
        self.q = q_go
        effort = (dq[:, :4].abs() / self.rate[:4]).mean(-1)

        # ---- physics substeps
        h = DT / SUB
        touch = torch.zeros(B, 4, device=dev)
        on_ball = torch.zeros(B, 2, dtype=torch.bool, device=dev)     # the BALL on each plate's inner face (grasping)
        P_prev = fk(q0)
        mn, mx = self._walls(self.basket)
        for k in range(1, SUB + 1):
            P = fk(q0 + (q1 - q0) * (k / SUB))
            vpl = (P["pl"] - P_prev["pl"]) / h
            vpr = (P["pr"] - P_prev["pr"]) / h
            p, v = self.ball, self.bv
            held = self.held
            # held: the ball rides in the claw
            gp = held_pos(P, self.hold_off)
            v_h = (gp - p) / h
            p_h = gp
            # free: gravity and motion
            v = v - torch.tensor([0, 0, GRAV * h], device=dev)
            p = p + v * h
            # table
            on_tab = ((p[:, :2] > 0) & (p[:, :2] < L)).all(-1)
            low = (p[:, 2] < BALL_R) & on_tab & (p[:, 2] > -0.05)
            vz = torch.where(low & (v[:, 2] < 0), -REST * v[:, 2], v[:, 2])
            vz = torch.where(low & (vz.abs() < 0.15), torch.zeros_like(vz), vz)
            p = torch.cat([p[:, :2], torch.where(low, torch.full_like(p[:, 2], BALL_R), p[:, 2:3].squeeze(-1))[:, None]], -1)
            roll = torch.where(low, 1 - ROLL * h, torch.ones_like(vz))
            v = torch.cat([v[:, :2] * roll[:, None], vz[:, None]], -1)
            # rim
            for ax in (0, 1):
                # the rim stops a ball rolling on the table; one carried higher than the rim passes over it
                # (and falls off the table: lost, a new one appears)
                rim = p[:, 2] < WALL_H + BALL_R
                lo_ = rim & (p[:, ax] < BALL_R) & (p[:, ax] > -0.05)
                hi_ = rim & (p[:, ax] > L - BALL_R) & (p[:, ax] < L + 0.05)
                pa = torch.where(lo_ | hi_, p[:, ax].clamp(BALL_R, L - BALL_R), p[:, ax])
                va = torch.where((lo_ & (v[:, ax] < 0)) | (hi_ & (v[:, ax] > 0)), -0.5 * v[:, ax], v[:, ax])
                p = torch.cat([p[:, :ax], pa[:, None], p[:, ax + 1:]], -1)
                v = torch.cat([v[:, :ax], va[:, None], v[:, ax + 1:]], -1)
            p, v = _boxes(p, v, mn, mx)
            # the plates push the ball (kinematic)
            ax = (P["hand"], P["side"], P["up"])
            for side, (c, vp) in enumerate(((P["pl"], vpl), (P["pr"], vpr))):
                dist, n, loc = obb_contact(p, c, ax, (LF / 2, PT / 2, PW / 2))
                hit = dist < BALL_R
                p = torch.where(hit[:, None], p + n * (BALL_R - dist)[:, None], p)
                vr = v - vp
                vn = (vr * n).sum(-1)
                v = torch.where((hit & (vn < 0))[:, None], v - (1 + REST) * vn[:, None] * n, v)
                near = (dist < BALL_R + 0.004).float() * ((BALL_R + 0.004 - dist) / 0.008).clamp(0.3, 1)
                inner = loc[:, 1] > 0 if side == 0 else loc[:, 1] < 0     # left plate's inner face looks to +side
                touch[:, 2 * side] = torch.maximum(touch[:, 2 * side], near * inner.float())
                on_ball[:, side] |= (near > 0) & inner
                touch[:, 2 * side + 1] = torch.maximum(touch[:, 2 * side + 1], near * (~inner).float())
            # the arm's links push it too (no passing through the forearm)
            lp = link_points(P)                                              # B,K,3
            d = p[:, None] - lp
            dist = d.norm(dim=-1).clamp(min=1e-6)
            hit = dist < BALL_R + LINK_R
            n = d / dist[..., None]
            push = (n * (BALL_R + LINK_R - dist).clamp(min=0)[..., None] * hit[..., None]).sum(1)
            p = p + push
            nn_ = push / push.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            vn = (v * nn_).sum(-1)
            v = torch.where((hit.any(-1) & (vn < 0))[:, None], v - vn[:, None] * nn_, v)
            # the arm can't shove it into the table, the basket or the post
            p, v = _boxes(p, v, mn, mx)
            on_tab = ((p[:, :2] > 0) & (p[:, :2] < L)).all(-1)
            p = torch.cat([p[:, :2], torch.where(on_tab[:, None], p[:, 2:3].clamp(min=BALL_R), p[:, 2:3])], -1)
            p = torch.where(held[:, None], p_h, p)
            v = torch.where(held[:, None], v_h, v)
            self.ball, self.bv = p, v
            P_prev = P
        # plates on the table or basket rim: felt on both faces
        tp = self._arm_points(P)[:, 4:]                                     # plate corners, 4 per plate
        on = (tp[..., 2] < 0.006) | blocked[:, None]
        # (felt on the OUTER face only: the inner faces feel only the ball, so "something between my plates"
        #  is never confused with the table; before, table and ball both read ~0.6 on all four faces)
        touch[:, 1:2] = torch.maximum(touch[:, 1:2], on[:, :4].any(-1, keepdim=True).float() * 0.6)
        touch[:, 3:4] = torch.maximum(touch[:, 3:4], on[:, 4:].any(-1, keepdim=True).float() * 0.6)
        # grasp: both inner faces on the ball and the claw nearly closed on it
        g = self.q[:, 4]
        both = on_ball[:, 0] & on_ball[:, 1]          # the ball itself between both inner faces (not the table)
        grab = ~self.held & both & (g < 2 * BALL_R + 0.012) & alive
        self.held = self.held | grab
        # caught where it is: remember the ball in the claw's frame (centred between the plates)
        Pg = fk(self.q)
        mid = (Pg["pl"] + Pg["pr"]) / 2
        dlt = self.ball - mid
        off = torch.stack([(dlt * Pg["hand"]).sum(-1).clamp(-LF / 2, LF / 2), torch.zeros_like(dlt[:, 0]),
                           (dlt * Pg["up"]).sum(-1).clamp(-PW / 2, PW / 2)], -1)
        self.hold_off = torch.where(grab[:, None], off, self.hold_off)
        # how hard it squeezes: how far past the ball the claw is told to close
        squeeze = ((2 * BALL_R + 0.004 - target[:, 4]) / 0.03).clamp(0, 0.5)
        # it lets go when the claw opens; a weak squeeze on a fast swing slips
        swing = self.bv.norm(dim=-1)
        slip = self.held & ~grab & (squeeze < 0.12) & (swing > 1.2)
        drop = self.held & ((g > 2 * BALL_R + 0.015) | slip)
        self.held = self.held & ~drop
        # a ball let go a few mm low (the held-ball check allows 3 mm) sits on the table, not in it
        on_tab = ((self.ball[:, :2] > 0) & (self.ball[:, :2] < L)).all(-1) & ~self.held & (self.ball[:, 2] > -0.05)
        self.ball = torch.cat([self.ball[:, :2], torch.where(on_tab[:, None], self.ball[:, 2:3].clamp(min=BALL_R), self.ball[:, 2:3])], -1)
        self.q[:, 4] = torch.where(self.held, g.clamp(min=2 * BALL_R - 0.004), g)
        touch[:, 0] = torch.where(self.held, 0.5 + squeeze, touch[:, 0])
        touch[:, 2] = torch.where(self.held, 0.5 + squeeze, touch[:, 2])
        self.touch = touch

        # ---- the basket feeds it
        bl = self.ball
        inb = ((bl[:, :2] - self.basket).abs() < B_IN).all(-1) & (bl[:, 2] < max(B_H, BALL_R + 0.015)) & ~self.held
        scored = inb & (self.score_t == 0) & alive
        self.energy = torch.where(scored, (self.energy + BALL_E).clamp(max=1.0), self.energy)
        self.score_t = torch.where(scored, torch.full_like(self.score_t, 12), self.score_t)
        due = (self.score_t == 1)
        self.score_t = (self.score_t - 1).clamp(min=0)
        # a ball dropped off the table is lost: a new one appears; also a new ball after scoring
        lost = ~self.held & (self.ball[:, 2] < -0.1) & (self.score_t == 0)
        idx = (due | lost).nonzero().squeeze(-1)
        if len(idx):
            self._new_ball(idx)

        # ---- body
        self.energy = self.energy - (DRAIN + EFFORT * effort) * alive
        self.health = (self.health - 0.03 * pain.float() + 0.003).clamp(max=1.0)
        self.t += 1
        starve, hurt = self.energy <= 0, self.health <= 0
        newly = alive & (starve | hurt)
        self.cause = torch.where(newly, torch.where(starve, 1, 2), self.cause)
        self.dead = self.dead | newly
        trunc = alive & ~newly & (self.t >= MAX_STEPS)
        ate = torch.where(scored, torch.zeros_like(self.t), torch.full_like(self.t, -1))
        self._sense()
        return self.dead.clone(), trunc, self.cause.clone(), ate

    def _sense(self):
        qn = ((self.q - self.lo) / (self.hi - self.lo) * 2 - 1).clamp(-1, 1)
        self.senses = torch.cat([self.energy.clamp(0, 1)[:, None], self.health.clamp(0, 1)[:, None],
                                 self.touch, qn], -1)

    # ------------------------------------------------------------------ the eyes
    def scene(self):
        B, dev = self.B, self.dev
        mn, mx = self._walls(self.basket, post=False)
        post_mn = torch.tensor([BASE[0] - 0.03, BASE[1] - 0.03, 0.0], device=dev).expand(B, 1, 3)
        post_mx = torch.tensor([BASE[0] + 0.03, BASE[1] + 0.03, SH_Z - 0.02], device=dev).expand(B, 1, 3)
        amin = torch.cat([mn, post_mn], 1)
        amax = torch.cat([mx, post_mx], 1)
        acol = torch.cat([self.basket_col[:, None].expand(B, 4, 3), (self.arm_col * 0.6)[:, None]], 1)
        glow = torch.zeros(B, 5, device=dev)
        P = fk(self.q)
        beads, rad, col = [], [], []

        def chain(a, b, k, r, c):
            for i in range(k):
                beads.append(a + (b - a) * ((i + 0.5) / k)); rad.append(r); col.append(c)
        joint = self.arm_col * 0.75
        for pnt in (P["sh"], P["el"], P["wr"]):
            beads.append(pnt); rad.append(0.026); col.append(joint)
        chain(P["sh"], P["el"], 14, 0.022, self.arm_col)          # beads closer than their size: a solid limb, not dots
        chain(P["el"], P["wr"], 14, 0.019, self.arm_col)
        chain(P["wr"], P["root"], 2, 0.018, self.arm_col)
        plate_col = torch.tensor([0.12, 0.12, 0.16], device=dev).expand(B, 3)
        chain(P["pl"] - P["hand"] * (LF / 2), P["pr"] - P["hand"] * (LF / 2), 5, PT * 1.4, plate_col)   # the palm: plates hang from it (one claw, not two floating prongs)
        for c in (P["pl"], P["pr"]):
            a_, b_ = c - P["hand"] * (LF / 2), c + P["hand"] * (LF / 2)
            chain(a_, b_, 5, PT * 1.2, plate_col)
        beads.append(self.ball); rad.append(BALL_R); col.append(self.ball_col)
        xyz = torch.stack(beads, 1)
        sph_r = torch.tensor(rad, device=dev).expand(B, len(rad))
        return dict(pos=self.pos, head=self.head, pitch=self.pitch, amin=amin, amax=amax, acol=acol, glow=glow,
                    sph=xyz, sph_z=xyz[..., 2], sph_r=sph_r, sph_col=torch.stack(col, 1),
                    floor_col=self.floor_col, wall_col=self.wall_col,
                    lava=torch.full((B, 1, 4), -10.0, device=dev))
