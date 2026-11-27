"""The v2 world: many 3D arenas at once, batched on the GPU.

Each arena (one per "env") holds:
  - an 11.3 m square floor (two-tone checker) closed by 2 m walls, sky above,
  - static boxes (some are 1.4 m "ledges" too tall to jump onto from the floor),
  - movable crates (0.6 m; pushable, and carryable with the grip),
  - lava patches on the floor,
  - food of 3 kinds (cube, stick, flat tile), some of it on top of ledges,
  - three beings: a predator that chases and bites, a harmless wanderer, and a strange
    being that is harmless at first but drains health if the creature stays near it.

The creature has five muscles, each in [-1, 1]: push (forward/back), turn, jump, grip, and gaze
(turns both eyes left/right of the body by up to GAZE; off by default, then the muscle does nothing).
Its senses (all produced here, nothing privileged):
  - two eyes: 120 degree stereo pictures (render.py),
  - body feelings: energy, health, touch (pushing against something), on-ground, holding,
    and (last of all senses) stamina: 1 unless set_body(stamina=True), and hunger: a feeling that
    grows louder as energy nears 0 (log scale: 0.1 -> 0.43, 0.03 -> 0.66, 0.01 -> 0.87), so
    "almost empty" and "empty" feel very different (energy itself barely differs there),
  - intensity: 4 sensors (left, right, front, bottom) x 4 bands (heat, pressure, hum, field),
  - ears: left and right x 4 bands; the outer ear dulls high bands coming from behind.
Its life ends when energy or health reaches 0, or after 100 s (a time limit, not a death).

Step = 0.1 s. Energy is spent by a metabolism (idle drain, more for moving, jumping and
carrying) and refilled by food. Health is lost to lava heat, bites, hard blows and the
strange being, and slowly recovers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

DT = 0.1
L = 11.3            # arena side (m)
WALL_H = 2.0
R_BODY = 0.25       # creature radius
EYE_H = 0.45        # eye height above the feet
EYE_SEP = 0.04      # half the distance between the eyes
STEP_UP = 0.3       # it can walk onto anything this low
MAX_STEPS = 1000    # 100 s life cap
NB = 24             # box slots (static boxes + crates)
NL = 7              # lava slots
NF = 8              # food slots
SENSES = 33         # 5 body + 16 intensity + 8 ears + stamina + hunger + 2 nostrils (smell; 0 unless set_body(smell=1))
BODY = 5
ACT = 5            # push, turn, jump, grip, gaze (gaze is ignored unless set_gaze(deg) > 0)
GAZE = 0.0         # radians the eyes can turn left/right of the body


def set_gaze(deg):
    global GAZE
    GAZE = math.radians(float(deg or 0))


LIGHT = 0           # steps of white light seen after dying, before the life ends (0 = off)


def set_light(n):
    """When a creature dies, its eyes fade to white for n steps (muscles do nothing, body frozen),
    then the life ends. Lifetime (bench) counts only up to the moment of death."""
    global LIGHT
    LIGHT = int(n or 0)

STAMINA = False     # a fast tiredness pool next to the slow food energy
METAB = 1.0         # how fast the food energy drains (1 = a full tank lasts 30 s standing still)


BACK = 1.0          # top speed walking backward, relative to forward
HEAT = 1            # 1: heat felt from far (inverse square, bottom sensor x5 everywhere: always "on")
                    # 2: heat fades within ~1.5 m, and the bottom sensor burns only over lava


WORLD_V = 1         # 2: audit fixes (Nov 27), see set_body
SMELL = False       # two nostrils smell food (any kind), stronger the closer and the more it lies to that side


def set_body(stamina=0, metab=1.0, life_cap=100, back_speed=1.0, heat=1, world_v=1, smell=0):
    """Stamina: sprinting, turning hard and jumping drain it in seconds; easy moving refills it
    in ~8 s. A tired body is weaker (top speed down to 35%, no jump below 15%); it never kills.
    metab: food-energy drain (0.2 = a full tank lasts 150 s standing still). life_cap in s.
    world_v 3: also the predator lunges straight when close (corners and walls were safe spots),
    and the rooms are more open (half the fixed blocks; crates and ledges as before).
    world_v 2: crates spawn clear of other boxes and their overlap tests see height; a carry that
    can't move the crate lets go; the back sensor feels bumps when reversing; blows and stamina
    follow the real drive (slow reversing is gentle); the dying body stops sliding."""
    global STAMINA, METAB, MAX_STEPS, BACK, HEAT, WORLD_V, SMELL
    WORLD_V = int(world_v or 1)
    SMELL = bool(smell)
    HEAT = int(heat or 1)
    STAMINA, METAB, BACK = bool(stamina), float(metab or 1.0), float(back_speed if back_speed is not None else 1.0)
    MAX_STEPS = int(round(float(life_cap or 100) / DT))


FOOD_HALF = torch.tensor([[0.15, 0.15, 0.15], [0.05, 0.05, 0.25], [0.22, 0.22, 0.05]])
FOOD_COL = torch.tensor([[0.95, 0.35, 0.25], [0.95, 0.85, 0.20], [0.40, 0.95, 0.30]])
PRED_R, WAND_R, STRANGE_HALF = 0.35, 0.30, 0.30
PRED_SEE, PRED_HEAR, PRED_FORGET = 6.0, 2.0, 50    # sight range (m, needs a clear line), hearing range, memory (steps)

DEATH_CAUSES = ["alive", "hunger", "lava", "predator", "strange", "blows"]


@dataclass
class Mode:
    n_static: int = 16
    n_ledge: int = 3
    n_crate: int = 6
    n_lava: int = 4
    n_food: int = 8
    n_ledge_food: int = 2
    pred_speed: float = 1.6


MODES = {
    "train": Mode(),
    # unseen worlds: same rules, a different mix of things
    "test": Mode(n_static=18, n_ledge=2, n_crate=5, n_lava=5, n_food=7, n_ledge_food=1, pred_speed=1.7),
    "hard": Mode(n_static=17, n_ledge=3, n_crate=4, n_lava=7, n_food=6, n_ledge_food=2, pred_speed=2.0),
}


def _jit(g, shape, dev, a=0.05):
    return 1 + (torch.rand(shape, generator=g, device=dev) * 2 - 1) * a


class World:
    """B arenas stepped together. All state lives in tensors on `device`."""

    def __init__(self, B: int, device="cuda", mode="train", seed=0, metabolism=True, compile=False,
                 rooms=0, room_seed=0):
        """rooms > 0: lives are drawn from a fixed list of `rooms` worlds made from room_seed, and
        world slot e on its k-th life always gets the same one, in every run (whatever the mind
        does, whenever it dies). Twins trained with different seeds then live in the same rooms."""
        self.compile, self._step_c = compile, None
        self.B, self.dev, self.metabolism = B, torch.device(device), metabolism
        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(seed)
        self.mode_name = mode
        self.mode = MODES[mode]
        self.food_half = FOOD_HALF.to(self.dev)
        self.food_col = FOOD_COL.to(self.dev)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=self.dev, dtype=dt)
        self.pos, self.vel = z(B, 3), z(B, 3)
        self.head = z(B)
        self.gaze = z(B)
        self.on_ground = z(B, dt=torch.bool)
        self.energy, self.health = z(B), z(B)
        self.stamina = z(B)
        self.t = z(B, dt=torch.long)
        self.dying, self.dcause = z(B, dt=torch.long), z(B, dt=torch.long)   # light steps seen; cause
        self.bmin, self.bmax, self.bcol = z(B, NB, 3), z(B, NB, 3), z(B, NB, 3)
        self.bact, self.bmov = z(B, NB, dt=torch.bool), z(B, NB, dt=torch.bool)
        self.lmin, self.lmax, self.lact = z(B, NL, 2), z(B, NL, 2), z(B, NL, dt=torch.bool)
        self.fpos, self.fkind = z(B, NF, 3), z(B, NF, dt=torch.long)
        self.fslot, self.fup, self.fledge = z(B, NF, dt=torch.bool), z(B, NF, dt=torch.bool), z(B, NF, dt=torch.long)
        self.ftimer, self.fcol = z(B, NF, dt=torch.long), z(B, NF, 3)
        self.pred, self.wand, self.strange = z(B, 2), z(B, 2), z(B, 2)
        self.wand_dir, self.strange_goal = z(B), z(B, 2)
        self.bite_cd, self.near_strange = z(B, dt=torch.long), z(B, dt=torch.long)
        self.pred_speed = z(B)
        self.pred_stam = z(B)          # the predator tires too (only with STAMINA): a chase is a sprint
        self.pred_mem, self.pred_mem_t = z(B, 2), z(B, dt=torch.long)   # where/when it last noticed the creature
        self.pred_dir, self.pred_side = z(B), z(B)                         # patrol heading; which way it goes around
        self.held = torch.full((B,), -1, device=self.dev, dtype=torch.long)
        self.floor_col, self.wall_col = z(B, 2, 3), z(B, 4, 3)
        self.pred_col, self.wand_col, self.strange_col = z(B, 3), z(B, 3), z(B, 3)
        # per-step outputs
        self.touch, self.pressure = z(B), z(B)
        self.bump_back = z(B, dt=torch.bool)
        self.senses = z(B, SENSES)
        self.bank = None
        if rooms:
            self.bank = World(rooms, device, mode, seed=room_seed, metabolism=metabolism)
            self.life = z(B, dt=torch.long)
            self._copy = [k for k, v in vars(self).items() if isinstance(v, torch.Tensor) and k != "life"
                          and v.dim() and v.shape[0] == B and getattr(self.bank, k).shape[0] == rooms]
        self.reset(torch.arange(B, device=self.dev))

    # ------------------------------------------------------------------ world generation
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    def _sel(self, x, idx):
        return x if idx is None else x[idx]

    def _free_points(self, idx, k, avoid_lava=True, margin=0.6, gen=True):
        """For each env in idx (None = all), k random floor points not inside boxes (and lava).
        gen=False draws from the global generator (used inside steps, which may be compiled)."""
        n, C = (self.B if idx is None else len(idx)), 24
        r = self._rand(n, k, C, 2) if gen else torch.rand(n, k, C, 2, device=self.dev)
        p = 0.8 + r * (L - 1.6)
        S = lambda x: self._sel(x, idx)
        bmin, bmax = S(self.bmin)[:, None, None], S(self.bmax)[:, None, None]   # n,1,1,NB,3
        inside = ((p[..., None, :] > bmin[..., :2] - margin) & (p[..., None, :] < bmax[..., :2] + margin)).all(-1)
        bad = (inside & S(self.bact)[:, None, None]).any(-1)
        if avoid_lava:
            lmin, lmax = S(self.lmin)[:, None, None], S(self.lmax)[:, None, None]
            inl = ((p[..., None, :] > lmin - 0.3) & (p[..., None, :] < lmax + 0.3)).all(-1)
            bad |= (inl & S(self.lact)[:, None, None]).any(-1)
        tie = self._rand(n, k, C) if gen else torch.rand(n, k, C, device=self.dev)
        score = (~bad).float() + tie * 0.01      # first good candidate (random tie-break)
        best = score.argmax(-1)
        return torch.gather(p, 2, best[..., None, None].expand(n, k, 1, 2)).squeeze(2)

    def reset(self, idx):
        """New random worlds for the envs in idx (a fresh life in a fresh layout)."""
        if len(idx) == 0:
            return
        if self.bank is not None:
            j = (idx * 7919 + self.life[idx] * 104729 + 1) % self.bank.B
            for k in self._copy:
                getattr(self, k)[idx] = getattr(self.bank, k)[j]
            self.life[idx] += 1
            self._sense(self.pos)
            return
        n, m, dev = len(idx), self.mode, self.dev
        g = self.g
        # boxes: static [0, n_static), ledges are the first n_ledge statics, crates after
        half = 0.3 + self._rand(n, NB, 2) * 0.5
        hgt = 0.35 + self._rand(n, NB) * 0.65
        j = torch.arange(NB, device=dev)
        is_ledge = j < m.n_ledge
        # world_v 3: a more open room (about half the blocks): long sight lines, food in view
        ns = m.n_static if WORLD_V < 3 else max(m.n_ledge + 2, round(m.n_static * 0.5))
        is_crate = (j >= ns) & (j < ns + m.n_crate)
        hgt = torch.where(is_ledge, torch.full_like(hgt, 1.4), hgt)
        hgt = torch.where(is_crate, torch.full_like(hgt, 0.6), hgt)
        half = torch.where(is_crate[:, None], torch.full_like(half, 0.3), half)
        ctr = 1.0 + self._rand(n, NB, 2) * (L - 2.0)
        if WORLD_V >= 2:      # each crate takes the first of 16 candidate spots clear of every other box
            C = 16
            cc = 1.0 + self._rand(n, NB, C, 2) * (L - 2.0)
            act = (j < ns + m.n_crate)
            for c_ in range(ns, ns + m.n_crate):
                lo_o, hi_o = ctr - half, ctr + half                                   # n,NB,2 (crates already moved)
                other = act & (j != c_)
                lo_c = cc[:, c_] - half[:, c_, None]                                 # n,C,2
                hi_c = cc[:, c_] + half[:, c_, None]
                ov = ((lo_c[:, :, None] < hi_o[:, None] + 0.05) & (hi_c[:, :, None] > lo_o[:, None] - 0.05)).all(-1)
                bad = (ov & other[None, None]).any(-1)                               # n,C
                best = ((~bad).float() + self._rand(n, C) * 0.01).argmax(-1)
                ctr[:, c_] = cc[torch.arange(n, device=dev), c_, best]
        self.bmin[idx] = torch.cat([ctr - half, torch.zeros(n, NB, 1, device=dev)], -1)
        self.bmax[idx] = torch.cat([ctr + half, hgt[..., None]], -1)
        self.bact[idx] = (j < ns + m.n_crate).expand(n, NB)
        self.bmov[idx] = is_crate.expand(n, NB)
        base = torch.where(is_crate[:, None], torch.tensor([0.78, 0.62, 0.42], device=dev),
                           torch.where(is_ledge[:, None], torch.tensor([0.42, 0.40, 0.48], device=dev),
                                       torch.tensor([0.55, 0.50, 0.44], device=dev)))
        self.bcol[idx] = base * _jit(g, (n, NB, 3), dev, 0.08)
        # lava
        lh = 0.4 + self._rand(n, NL, 2) * 0.6
        lc = 1.0 + self._rand(n, NL, 2) * (L - 2.0)
        self.lmin[idx], self.lmax[idx] = lc - lh, lc + lh
        self.lact[idx] = (torch.arange(NL, device=dev) < m.n_lava).expand(n, NL)
        # colours
        self.floor_col[idx] = torch.tensor([[0.62, 0.70, 0.58], [0.55, 0.63, 0.52]], device=dev) * _jit(g, (n, 1, 3), dev)
        self.wall_col[idx] = torch.tensor([0.70, 0.70, 0.74], device=dev) * _jit(g, (n, 4, 3), dev)
        self.pred_col[idx] = torch.tensor([0.50, 0.28, 0.78], device=dev) * _jit(g, (n, 3), dev)
        self.wand_col[idx] = torch.tensor([0.30, 0.62, 0.66], device=dev) * _jit(g, (n, 3), dev)
        self.strange_col[idx] = torch.tensor([0.90, 0.88, 0.72], device=dev) * _jit(g, (n, 3), dev)
        # creature
        self.pos[idx] = torch.cat([self._free_points(idx, 1)[:, 0], torch.zeros(n, 1, device=dev)], -1)
        self.vel[idx] = 0
        self.head[idx] = self._rand(n) * 2 * math.pi
        self.on_ground[idx] = True
        self.energy[idx] = 0.7 + self._rand(n) * 0.3
        self.health[idx] = 1.0
        self.stamina[idx] = 1.0
        self.t[idx] = 0
        self.dying[idx], self.dcause[idx] = 0, 0
        self.held[idx] = -1
        # food
        kinds = (self._rand(n, NF) * 3).long()
        self.fkind[idx] = kinds
        self.fcol[idx] = self.food_col[kinds] * _jit(g, (n, NF, 3), dev)
        k = torch.arange(NF, device=dev)
        self.fslot[idx] = (k < m.n_food).expand(n, NF)
        self.fledge[idx] = torch.where(k < m.n_ledge_food, k % max(m.n_ledge, 1), torch.full_like(k, -1)).expand(n, NF).clone()
        self.ftimer[idx] = 0
        self.fup[idx] = self.fslot[idx]
        self._place_food(idx, torch.ones(n, NF, dtype=torch.bool, device=dev))
        # beings: start away from the creature
        cand = self._free_points(idx, 6, avoid_lava=False)                      # n,6,2
        d = (cand - self.pos[idx, None, :2]).norm(dim=-1)
        far = d.argmax(-1)
        self.pred[idx] = cand[torch.arange(n, device=dev), far]
        self.wand[idx] = cand[:, 0]
        self.strange[idx] = cand[:, 1]
        self.wand_dir[idx] = self._rand(n) * 2 * math.pi
        self.strange_goal[idx] = self._free_points(idx, 1, avoid_lava=False)[:, 0]
        self.bite_cd[idx] = 0
        self.near_strange[idx] = 0
        self.pred_speed[idx] = m.pred_speed
        self.pred_stam[idx] = 1.0
        self.pred_mem[idx] = self.pred[idx]
        self.pred_mem_t[idx] = PRED_FORGET
        self.pred_dir[idx] = self._rand(n) * 2 * math.pi
        self.pred_side[idx] = torch.where(self._rand(n) < 0.5, -1.0, 1.0)
        self.touch[idx] = 0
        self.pressure[idx] = 0
        self._sense(self.pos)

    def _place_food(self, idx, which, gen=True):
        """(Re)place food slots `which` (n,NF bool) of envs idx (None = all): on their ledge or on free floor."""
        S = lambda x: self._sel(x, idx)
        floor = self._free_points(idx, NF, gen=gen)                              # n,NF,2
        n = floor.shape[0]
        kinds = S(self.fkind)
        hz = self.food_half[kinds][..., 2]
        led = S(self.fledge)
        lidx = led.clamp(min=0)
        bmin = torch.gather(S(self.bmin), 1, lidx[..., None].expand(n, NF, 3))
        bmax = torch.gather(S(self.bmax), 1, lidx[..., None].expand(n, NF, 3))
        on_ledge = led >= 0
        xy = torch.where(on_ledge[..., None], (bmin[..., :2] + bmax[..., :2]) / 2, floor)
        zc = torch.where(on_ledge, bmax[..., 2] + hz, hz)
        new = torch.cat([xy, zc[..., None]], -1)
        if idx is None:
            self.fpos = torch.where(which[..., None], new, self.fpos)
        else:
            self.fpos[idx] = torch.where(which[..., None], new, self.fpos[idx])

    # ------------------------------------------------------------------ physics
    def step(self, action):
        """action (B,4) in [-1,1]. Returns (done, trunc, cause, ate_kind) for this step."""
        if self.compile:
            if self._step_c is None:
                self._step_c = torch.compile(self._step, dynamic=False)
            return self._step_c(action)
        return self._step(action)

    def _step(self, action):
        B, dev = self.B, self.dev
        a = action.clamp(-1, 1)
        was_dying = self.dying > 0
        if LIGHT:
            e0, h0 = self.energy, self.health
            a = torch.where(was_dying[:, None], torch.zeros_like(a), a)
        push, turn = a[:, 0], a[:, 1]
        jump = (a[:, 2] > 0.5) & self.on_ground
        if STAMINA:
            jump = jump & (self.stamina > 0.15)
        grip = a[:, 3] > 0
        self.gaze = a[:, 4] * GAZE
        self.head = (self.head + turn * 2.5 * DT) % (2 * math.pi)
        fwd = torch.stack([torch.cos(self.head), torch.sin(self.head)], -1)
        drive = torch.where(push < 0, push * BACK, push) if BACK != 1.0 else push   # a body built to walk forward
        want = fwd * (drive * 2.5 * ((0.35 + 0.65 * self.stamina) if STAMINA else 1.0))[:, None]
        ctrl = torch.where(self.on_ground, 0.5, 0.1)[:, None]
        vxy = self.vel[:, :2] + ctrl * (want - self.vel[:, :2])
        if LIGHT and WORLD_V >= 2:     # the dying body stops where it fell
            vxy = torch.where(was_dying[:, None], torch.zeros_like(vxy), vxy)
        vz = torch.where(jump, torch.full_like(self.vel[:, 2], 4.2), self.vel[:, 2]) - 9.8 * DT
        old = self.pos.clone()
        p = self.pos + torch.cat([vxy, vz[:, None]], -1) * DT
        holding = self.held >= 0
        ar = torch.arange(B, device=dev)

        # --- horizontal collisions with boxes (circle approximated by a square of side 2r)
        feet = p[:, 2]
        bmin, bmax = self.bmin, self.bmax
        solid = self.bact & (bmax[..., 2] > feet[:, None] + STEP_UP)
        solid &= ~(holding[:, None] & (torch.arange(NB, device=dev)[None] == self.held[:, None]))
        lo = bmin[..., :2] - R_BODY
        hi = bmax[..., :2] + R_BODY
        pxy = p[:, None, :2]
        inside = ((pxy > lo) & (pxy < hi)).all(-1) & solid                    # B,NB
        pen_neg = pxy - lo                                                     # push toward -axis
        pen_pos = hi - pxy                                                     # push toward +axis
        pen = torch.cat([pen_neg, pen_pos], -1)                                # B,NB,4
        k = pen.argmin(-1)
        amt = pen.gather(-1, k[..., None]).squeeze(-1)
        dirs = torch.tensor([[-1., 0], [0, -1.], [1., 0], [0, 1.]], device=dev)
        disp = dirs[k] * amt[..., None] * inside[..., None]                    # creature displacement per box
        # crates move instead of the creature, unless they would run into something
        crate = inside & self.bmov
        cmove = -disp * crate[..., None]
        nmin = bmin[..., :2] + cmove
        nmax = bmax[..., :2] + cmove
        blocked_c = (nmin < 0.02).any(-1) | (nmax > L - 0.02).any(-1)
        # crate vs any other box (footprint overlap)
        ov = ((nmin[:, :, None] < bmax[:, None, :, :2]) & (nmax[:, :, None] > bmin[:, None, :, :2])).all(-1)
        if WORLD_V >= 2:      # boxes at different heights (a crate on a ledge, one under a carried one) don't collide
            ov &= (bmin[:, :, None, 2] < bmax[:, None, :, 2] - 1e-3) & (bmax[:, :, None, 2] > bmin[:, None, :, 2] + 1e-3)
        ov &= self.bact[:, None, :] & ~torch.eye(NB, dtype=torch.bool, device=dev)[None]
        blocked_c |= ov.any(-1)
        ok_c = crate & ~blocked_c
        self.bmin[..., :2] = torch.where(ok_c[..., None], nmin, bmin[..., :2])
        self.bmax[..., :2] = torch.where(ok_c[..., None], nmax, bmax[..., :2])
        disp = disp * (~ok_c)[..., None]
        p_xy = p[:, :2] + disp.sum(1)
        hit_box = (inside & ~ok_c).any(-1)
        # walls
        clamped = p_xy.clamp(R_BODY, L - R_BODY)
        hit_wall = (clamped != p_xy).any(-1)
        p = torch.cat([clamped, p[:, 2:]], -1)
        self.touch = ((hit_box | hit_wall) & (push.abs() > 0.05)).float()

        # --- vertical: ground under the creature (floor or a box top it is over)
        over = ((p[:, None, :2] > bmin[..., :2]) & (p[:, None, :2] < bmax[..., :2])).all(-1) & self.bact
        over &= bmax[..., 2] <= old[:, None, 2] + STEP_UP + 1e-4
        over &= ~(holding[:, None] & (torch.arange(NB, device=dev)[None] == self.held[:, None]))
        ground = torch.where(over, bmax[..., 2], torch.zeros_like(bmax[..., 2])).amax(-1)
        land = p[:, 2] <= ground
        p[:, 2] = torch.where(land, ground, p[:, 2])
        vz = torch.where(land, torch.zeros_like(vz), vz)
        self.vel = torch.cat([(p[:, :2] - old[:, :2]) / DT, vz[:, None]], -1)
        self.pos = p
        self.on_ground = land

        # --- grip: pick up the crate in front, carry it, drop it when the grip opens
        fwd = torch.stack([torch.cos(self.head), torch.sin(self.head)], -1)
        ctr = (self.bmin[..., :2] + self.bmax[..., :2]) / 2
        rel = ctr - p[:, None, :2]
        dist = rel.norm(dim=-1)
        front = (rel * fwd[:, None]).sum(-1) > 0
        cand = self.bmov & self.bact & (dist < 0.75) & front & ((self.bmin[..., 2] - p[:, None, 2]).abs() < 0.5)
        dsel = torch.where(cand, dist, torch.full_like(dist, 1e9))
        pick = dsel.argmin(-1)
        can = cand.any(-1) & grip & ~holding
        self.held = torch.where(can, pick, self.held)
        self.held = torch.where(~grip, torch.full_like(self.held, -1), self.held)
        holding = self.held >= 0
        if True:                       # (mask-safe for every env: no GPU->CPU wait)
            hi_ = self.held.clamp(min=0)
            size = (self.bmax[ar, hi_, :2] - self.bmin[ar, hi_, :2])
            target = p[:, :2] + fwd * (R_BODY + 0.32)
            nmin = target - size / 2
            nmax = target + size / 2
            okw = (nmin > 0.02).all(-1) & (nmax < L - 0.02).all(-1)
            ov = ((nmin[:, None] < self.bmax[..., :2]) & (nmax[:, None] > self.bmin[..., :2])).all(-1) & self.bact
            if WORLD_V >= 2:  # only boxes reaching the carried crate's height (it rides at the feet's height) block it
                hgt_c = self.bmax[ar, hi_, 2] - self.bmin[ar, hi_, 2]
                ov &= (self.bmax[..., 2] > p[:, None, 2] + 1e-3) & (self.bmin[..., 2] < (p[:, 2] + hgt_c)[:, None] - 1e-3)
            ov[ar, hi_] = False
            okm = holding & okw & ~ov.any(-1)
            self.bmin[ar, hi_, :2] = torch.where(okm[:, None], nmin, self.bmin[ar, hi_, :2])
            self.bmax[ar, hi_, :2] = torch.where(okm[:, None], nmax, self.bmax[ar, hi_, :2])
            # a carried crate rides at the creature's height (lifted onto ledges with it)
            zb = torch.where(okm, p[:, 2], self.bmin[ar, hi_, 2])
            hgt = self.bmax[ar, hi_, 2] - self.bmin[ar, hi_, 2]
            self.bmin[ar, hi_, 2] = zb
            self.bmax[ar, hi_, 2] = zb + hgt
            if WORLD_V >= 2:  # the crate can't come along: the grip slips
                self.held = torch.where(holding & ~okm, torch.full_like(self.held, -1), self.held)
                holding = self.held >= 0
        # dropped crates fall to whatever is under them
        free_c = self.bmov & self.bact & ~(holding[:, None] & (torch.arange(NB, device=dev)[None] == self.held[:, None]))
        if True:
            cmin, cmax = self.bmin[..., :2], self.bmax[..., :2]
            ov = ((cmin[:, :, None] < self.bmax[:, None, :, :2]) & (cmax[:, :, None] > self.bmin[:, None, :, :2])).all(-1)
            ov &= self.bact[:, None, :] & ~torch.eye(NB, dtype=torch.bool, device=dev)[None]
            below = ov & (self.bmax[:, None, :, 2] <= self.bmin[:, :, None, 2] + 1e-3)
            floor_z = torch.where(below, self.bmax[:, None, :, 2], torch.zeros_like(below, dtype=torch.float)).amax(-1)
            hgt = self.bmax[..., 2] - self.bmin[..., 2]
            nz = torch.where(free_c, floor_z, self.bmin[..., 2])
            self.bmin[..., 2] = nz
            self.bmax[..., 2] = nz + hgt

        # --- metabolism
        d0 = METAB / 300
        speed2 = (self.vel[:, :2] ** 2).sum(-1)
        if self.metabolism:
            moving = ((push.abs() > 0.05) | (turn.abs() > 0.05)).float()
            move = 0.2 * moving + 0.15 * push.abs() + 0.05 * turn.abs() + 0.0375 * speed2
            move = torch.where(holding, 2 * move + 0.1, move)
            cost = d0 * (1 + move) + 12 * d0 * jump.float()
        else:
            cost = torch.full_like(push, d0)
        self.energy = self.energy - cost
        if STAMINA:
            pe = drive if WORLD_V >= 2 else push                 # reversing slowly is easy
            effort = 0.015 * pe * pe + 0.004 * turn.abs() + 0.15 * jump.float()
            rest = 0.012 * (1 - pe.abs()) + 0.002     # walking calmly still recovers a little
            self.stamina = (self.stamina - effort + rest).clamp(0, 1)

        # --- food
        fc = self.fpos
        dxy = (fc[..., :2] - p[:, None, :2]).norm(dim=-1)
        fz = fc[..., 2]
        reach = (dxy < R_BODY + 0.25) & (fz > p[:, None, 2] - 0.2) & (fz < p[:, None, 2] + 0.9) & self.fup
        first = reach.float().argmax(-1)
        ate = reach.any(-1) & ~was_dying
        ate_kind = torch.where(ate, self.fkind[ar, first], torch.full_like(first, -1))
        self.energy = torch.where(ate, (self.energy + 0.3).clamp(max=1.0), self.energy)
        self.health = torch.where(ate, (self.health + 0.05).clamp(max=1.0), self.health)
        eaten = torch.zeros_like(self.fup)
        eaten[ar, first] = ate
        self.fup &= ~eaten
        self.ftimer = torch.where(eaten, torch.full_like(self.ftimer, 200), (self.ftimer - 1).clamp(min=0))
        respawn = self.fslot & ~self.fup & (self.ftimer == 0)
        self._place_food(None, respawn, gen=False)      # all envs, only the slots due (no sync)
        self.fup |= respawn

        # --- beings
        self._move_beings(p)
        cause = torch.zeros(B, dtype=torch.long, device=dev)
        # lava: heat under the feet
        on_floor = p[:, 2] < 0.05
        in_lava = (((p[:, None, :2] > self.lmin) & (p[:, None, :2] < self.lmax)).all(-1) & self.lact).any(-1) & on_floor
        dmg_lava = in_lava.float() * 0.019
        # predator bite
        dpred = (self.pred - p[:, :2]).norm(dim=-1)
        bite = (dpred < R_BODY + PRED_R + 0.05) & (p[:, 2] < 0.6) & (self.bite_cd == 0)
        self.bite_cd = torch.where(bite, torch.full_like(self.bite_cd, 10), (self.bite_cd - 1).clamp(min=0))
        dmg_bite = bite.float() * 0.34
        # strange being: harmless at first, then drains while the creature stays near
        dstr = (self.strange - p[:, :2]).norm(dim=-1)
        near = (dstr < 1.5) & (p[:, 2] < 1.0)
        self.near_strange = torch.where(near, self.near_strange + 1, torch.zeros_like(self.near_strange))
        dmg_str = (self.near_strange > 30).float() * 0.02
        # blows: pressure above a threshold hurts
        pf = drive.abs() if WORLD_V >= 2 else push.abs()          # a bump hurts as hard as the body really pushed
        self.pressure = self.touch * 6 * pf + bite.float() * 10
        self.bump_back = (push < 0) & (WORLD_V >= 2)
        dmg_blow = (self.touch * 6 * pf - 5).clamp(min=0) * 0.01
        dmg = dmg_lava + dmg_bite + dmg_str + dmg_blow
        self.health = (self.health - dmg + 0.001).clamp(max=1.0)

        if LIGHT:                                # the dying keep the body they died with
            self.energy = torch.where(was_dying, e0, self.energy)
            self.health = torch.where(was_dying, h0, self.health)
        self.t += (~was_dying).long()
        starve = self.energy <= 0
        killed = self.health <= 0
        done = starve | killed
        # cause of death: the biggest harm this step if killed, else hunger
        harms = torch.stack([dmg_lava, dmg_bite, dmg_str, dmg_blow], -1)
        cause = torch.where(killed, harms.argmax(-1) + 2, cause)
        cause = torch.where(starve & ~killed, torch.ones_like(cause), cause)
        if LIGHT:                                # death starts the light; the life ends after it
            newly = done & ~was_dying
            self.dcause = torch.where(newly, cause, self.dcause)
            done = was_dying & (self.dying >= LIGHT)
            cause = torch.where(done, self.dcause, torch.zeros_like(cause))
            self.dying = torch.where(newly | was_dying, self.dying + 1, self.dying)
        trunc = (self.t >= MAX_STEPS) & ~done & (self.dying == 0)
        self._sense(p)
        return done, trunc, cause, ate_kind

    def _slide(self, q, r):
        """Push circles (B,2) of radius r out of active boxes taller than 0.3 m and keep them inside."""
        lo = self.bmin[..., :2] - r
        hi = self.bmax[..., :2] + r
        qq = q[:, None]
        inside = ((qq > lo) & (qq < hi)).all(-1) & self.bact & (self.bmax[..., 2] > 0.3)
        pen = torch.cat([qq - lo, hi - qq], -1)
        k = pen.argmin(-1)
        amt = pen.gather(-1, k[..., None]).squeeze(-1)
        dirs = torch.tensor([[-1., 0], [0, -1.], [1., 0], [0, 1.]], device=self.dev)
        q = q + (dirs[k] * amt[..., None] * inside[..., None]).sum(1)
        return q.clamp(r, L - r)

    def _blocked(self, q, r):
        """q (B,K,2): inside a tall active box (grown by r), or outside the arena."""
        lo = self.bmin[:, None, :, :2] - r
        hi = self.bmax[:, None, :, :2] + r
        qq = q[:, :, None]
        tall = (self.bact & (self.bmax[..., 2] > 0.3))[:, None]
        inside = (((qq > lo) & (qq < hi)).all(-1) & tall).any(-1)
        return inside | (q < r).any(-1) | (q > L - r).any(-1)

    def _clear(self, a, b):
        """Line of sight from a to b (B,2) at body height: no tall box crosses the segment."""
        d = b - a
        dsafe = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
        inv = 1.0 / dsafe[:, None]
        t1 = (self.bmin[..., :2] - a[:, None]) * inv
        t2 = (self.bmax[..., :2] - a[:, None]) * inv
        tn = torch.minimum(t1, t2).max(-1).values
        tf = torch.maximum(t1, t2).min(-1).values
        hit = (tn < tf) & (tf > 0) & (tn < 1) & self.bact & (self.bmax[..., 2] > 0.6)
        return ~hit.any(-1)

    def _move_beings(self, p):
        B = self.B
        # predator: a simple hunter (no learning). It notices the creature only if it can see it
        # (within 6 m with no tall box in between) or hear it (within 2 m). Then it chases; when it
        # loses it, it goes to where it last noticed it, and gives up after 5 s and patrols.
        # It steers around boxes: if the way ahead is blocked it tries turning more and more to
        # one side (keeping that side while blocked, like following a wall).
        to = p[:, :2] - self.pred
        d = to.norm(dim=-1).clamp(min=1e-4)
        clear = self._clear(self.pred, p[:, :2])
        notice = ((d < PRED_SEE) & clear) | (d < PRED_HEAR)
        self.pred_mem = torch.where(notice[:, None], p[:, :2], self.pred_mem)
        self.pred_mem_t = torch.where(notice, torch.zeros_like(self.pred_mem_t), self.pred_mem_t + 1)
        goal = self.pred_mem - self.pred
        gd = goal.norm(dim=-1)
        hunting = (self.pred_mem_t < PRED_FORGET) & (gd > 0.3)
        self.pred_mem_t = torch.where(~hunting & (gd <= 0.3), torch.full_like(self.pred_mem_t, PRED_FORGET), self.pred_mem_t)
        self.pred_dir = self.pred_dir + (torch.rand(B, device=self.dev) - 0.5) * 0.5
        want = torch.where(hunting, torch.atan2(goal[:, 1], goal[:, 0]), self.pred_dir)
        if STAMINA:     # it can sprint ~6 s at full speed, then slows to 40%; rests back while patrolling
            self.pred_stam = (self.pred_stam + torch.where(hunting, -0.016, 0.008)).clamp(0, 1)
            speed = torch.where(hunting, self.pred_speed * (0.4 + 0.6 * self.pred_stam), self.pred_speed * 0.4) * DT
        else:
            speed = torch.where(hunting, self.pred_speed, self.pred_speed * 0.4) * DT
        # try headings: straight, then 30..150 deg to the current side, then to the other side
        offs = torch.tensor([0, 30, 60, 90, 120, 150, -30, -60, -90, -120, -150], device=self.dev) * (math.pi / 180)
        ang = want[:, None] + offs[None] * self.pred_side[:, None]                       # B,K
        look = 0.25 + PRED_R
        probe = self.pred[:, None] + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * look
        free = ~self._blocked(probe, PRED_R)                                             # B,K
        k = torch.where(free.any(-1), free.float().argmax(-1), torch.zeros_like(free[:, 0], dtype=torch.long))
        self.pred_side = torch.where(k >= 6, -self.pred_side, self.pred_side)          # went the other way: remember it
        head = ang.gather(1, k[:, None]).squeeze(1)
        if WORLD_V >= 3:      # close and in sight: lunge straight at it (walls/boxes still stop it via _slide);
            # the 0.6 m look-ahead saw the wall behind a creature as "blocked" and swerved: corners were safe
            lunge = hunting & (d < 1.5) & clear
            head = torch.where(lunge, want, head)
        self.pred_dir = torch.where(hunting, self.pred_dir, head)                       # patrol keeps its new heading
        step = torch.stack([torch.cos(head), torch.sin(head)], -1) * speed[:, None]
        step = torch.where((hunting & (d <= 0.55))[:, None], torch.zeros_like(step), step)
        self.pred = self._slide(self.pred + step, PRED_R)
        # wanderer: a slow random walk
        self.wand_dir = self.wand_dir + (torch.rand(B, device=self.dev) - 0.5) * 0.8
        w = torch.stack([torch.cos(self.wand_dir), torch.sin(self.wand_dir)], -1) * 0.8 * DT
        nw = self._slide(self.wand + w, WAND_R)
        stuck = (nw - (self.wand + w)).norm(dim=-1) > 1e-3
        self.wand_dir = torch.where(stuck, self.wand_dir + math.pi * 0.7, self.wand_dir)
        self.wand = nw
        # strange being: drifts slowly between random spots
        to = self.strange_goal - self.strange
        d = to.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        self.strange = self._slide(self.strange + to / d * 0.4 * DT, STRANGE_HALF)
        arrived = d.squeeze(-1) < 0.5
        new_goal = self._free_points(None, 1, avoid_lava=False, gen=False)[:, 0]
        self.strange_goal = torch.where(arrived[:, None], new_goal, self.strange_goal)

    # ------------------------------------------------------------------ senses
    def _sense(self, p):
        B, dev = self.B, self.dev
        fwd = torch.stack([torch.cos(self.head), torch.sin(self.head)], -1)
        left = torch.stack([-fwd[:, 1], fwd[:, 0]], -1)
        # intensity sensors: left, right, front side points (normals), and the bottom
        pts = torch.stack([p[:, :2] + left * R_BODY, p[:, :2] - left * R_BODY, p[:, :2] + fwd * R_BODY, p[:, :2]], 1)
        nrm = torch.stack([left, -left, fwd], 1)                                       # B,3,2
        # band 0, heat from lava: falls off with distance to each patch
        q = pts[:, :, None]                                                            # B,4,1,2
        dx = torch.maximum(self.lmin[:, None] - q, q - self.lmax[:, None]).clamp(min=0).norm(dim=-1)  # B,4,NL
        lift = torch.exp(-p[:, 2] / 0.3)
        if HEAT == 2:
            per = 4.0 / (1 + (dx / 0.4) ** 2) * torch.exp(-dx / 0.7)                   # 1 m: 0.13, 2 m: 0.01
            bottom = (torch.arange(4, device=dev) == 3)[None, :, None]
            under = torch.where(bottom, 5 * (dx == 0).float(), torch.ones_like(per))     # the feet burn only on lava
            heat = (per * under * self.lact[:, None]).sum(-1) * lift[:, None]
        else:
            heat = (4.0 / (1 + (dx / 0.4) ** 2) * self.lact[:, None]).sum(-1)          # B,4
            heat = heat * torch.stack([lift, lift, lift, lift * 5], -1)
        # band 1, pressure: blows on the front, bites on the side facing the predator
        to_p = self.pred - p[:, :2]
        dp = to_p.norm(dim=-1).clamp(min=1e-3)
        u_p = to_p / dp[:, None]
        gain_p = 0.5 + 0.5 * (nrm * u_p[:, None]).sum(-1).clamp(min=0)                # B,3
        bite = (self.pressure >= 10).float()
        press = torch.zeros(B, 4, device=dev)
        blow = self.pressure - bite * 10
        press[:, :3] = bite[:, None] * 10 * gain_p
        if WORLD_V >= 2:      # reversing into something is felt on the back (the bottom sensor, otherwise unused)
            back = self.bump_back.float()
            press[:, 2] += blow * (1 - back)
            press[:, 3] += blow * back
        else:
            press[:, 2] += blow
        # bands 2-3: the predator's hum and the strange being's field, through the side sensors
        hum = torch.zeros(B, 4, device=dev)
        hum[:, :3] = 2.0 / (1 + dp[:, None] ** 2) * gain_p
        hum[:, 3] = 2.0 / (1 + dp ** 2 + p[:, 2] ** 2) * 0.5
        to_s = self.strange - p[:, :2]
        ds = to_s.norm(dim=-1).clamp(min=1e-3)
        gain_s = 0.5 + 0.5 * (nrm * (to_s / ds[:, None])[:, None]).sum(-1).clamp(min=0)
        fld = torch.zeros(B, 4, device=dev)
        fld[:, :3] = 2.0 / (1 + ds[:, None] ** 2) * gain_s
        fld[:, 3] = 2.0 / (1 + ds ** 2) * 0.5
        intensity = torch.stack([heat, press, hum, fld], -1)                           # B,4 sensors,4 bands
        # ears: left/right, 4 bands; predator low+mid, wanderer mid+high, strange highest
        ears = torch.zeros(B, 2, 4, device=dev)
        srcs = [(self.pred, torch.tensor([1.0, 0.6, 0.1, 0.0], device=dev) * 3),
                (self.wand, torch.tensor([0.0, 0.5, 0.8, 0.2], device=dev) * 2),
                (self.strange, torch.tensor([0.0, 0.0, 0.3, 1.0], device=dev) * 2)]
        ear_pos = torch.stack([p[:, :2] + left * 0.15, p[:, :2] - left * 0.15], 1)
        ear_n = torch.stack([left, -left], 1)
        for src, spec in srcs:
            to = src[:, None] - ear_pos
            d = to.norm(dim=-1).clamp(min=0.1)
            u = to / d[..., None]
            gain = 0.6 + 0.4 * (u * ear_n).sum(-1)                                     # B,2
            behind = (u * fwd[:, None]).sum(-1) < 0
            pinna = torch.where(behind[..., None], torch.tensor([1.0, 1.0, 0.5, 0.5], device=dev),
                                torch.ones(4, device=dev))
            ears += spec * pinna * (gain / (1 + d ** 2))[..., None]
        holding = (self.held >= 0).float()
        body = torch.stack([self.energy.clamp(0, 1), self.health.clamp(0, 1), self.touch,
                            self.on_ground.float(), holding], -1)
        stam = self.stamina if STAMINA else torch.ones_like(self.energy)
        hunger = (-torch.log(self.energy.clamp(0.005, 1.0)) / 5.3).clamp(0, 1)          # 0 full ... 1 at energy 0.005
        nose = torch.zeros(B, 2, device=dev)
        if SMELL:     # like the ears: nostrils 12 cm left/right of the snout, odour 2/(1+d^2) per food, a bit directional
            sn = torch.stack([p[:, :2] + fwd * R_BODY + left * 0.12, p[:, :2] + fwd * R_BODY - left * 0.12], 1)   # B,2,2
            to = self.fpos[:, None, :, :2] - sn[:, :, None]                                  # B,2,NF,2
            d = to.norm(dim=-1).clamp(min=0.05)
            u = to / d[..., None]
            side = torch.stack([left, -left], 1)[:, :, None]                                 # B,2,1,2
            gain = 0.5 + 0.5 * (u * side).sum(-1).clamp(min=0) + 0.25 * (u * fwd[:, None, None]).sum(-1).clamp(min=0)
            nose = (2.0 * gain / (1 + d ** 2) * self.fup[:, None].float()).sum(-1)
        self.senses = torch.cat([body, torch.log1p(intensity.flatten(1)), torch.log1p(ears.flatten(1)),
                                 stam[:, None], hunger[:, None], torch.log1p(nose)], -1)

    # ------------------------------------------------------------------ for the renderer
    def scene(self):
        """Everything the renderer needs as tensors (boxes incl. food and the strange being)."""
        B, dev = self.B, self.dev
        fh = self.food_half[self.fkind]
        fmin = self.fpos - fh
        fmax = self.fpos + fh
        far = torch.full_like(fmin, -100.0)
        fmin = torch.where(self.fup[..., None], fmin, far)
        fmax = torch.where(self.fup[..., None], fmax, far)
        smin = torch.cat([self.strange - STRANGE_HALF, torch.zeros(B, 1, device=dev)], -1)
        smax = torch.cat([self.strange + STRANGE_HALF, torch.full((B, 1), 2 * STRANGE_HALF, device=dev)], -1)
        bmin = torch.where(self.bact[..., None], self.bmin, torch.full_like(self.bmin, -100.0))
        bmax = torch.where(self.bact[..., None], self.bmax, torch.full_like(self.bmax, -100.0))
        amin = torch.cat([bmin, fmin, smin[:, None]], 1)
        amax = torch.cat([bmax, fmax, smax[:, None]], 1)
        acol = torch.cat([self.bcol, self.fcol, self.strange_col[:, None]], 1)
        glow = torch.cat([torch.zeros(B, NB, device=dev), torch.ones(B, NF, device=dev), torch.zeros(B, 1, device=dev)], 1)
        sph = torch.stack([torch.cat([self.pred, torch.full((B, 1), PRED_R, device=dev)], -1),
                           torch.cat([self.wand, torch.full((B, 1), WAND_R, device=dev)], -1)], 1)   # centre xy, z=r
        sph_r = torch.tensor([PRED_R, WAND_R], device=dev).expand(B, 2)
        sph_col = torch.stack([self.pred_col, self.wand_col], 1)
        lava = torch.cat([self.lmin, self.lmax], -1)
        lava = torch.where(self.lact[..., None], lava, torch.full_like(lava, -100.0))
        return dict(pos=self.pos, head=self.head + self.gaze, amin=amin, amax=amax, acol=acol, glow=glow,
                    sph=sph, sph_r=sph_r, sph_col=sph_col, lava=lava,
                    floor_col=self.floor_col, wall_col=self.wall_col)
