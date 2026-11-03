"""GPU reacting-flow solver for a candle flame, written with NVIDIA Warp.

Model (low-Mach, Boussinesq buoyancy, thermal expansion as a divergence source):

  * MAC staggered velocity, 2nd-order (RK2 back-trace + MacCormack, clamped)
    advection of velocity and all scalars.
  * Scalars: temperature T, paraffin vapour Yf, oxygen Yo, soot, condensed-wax
    smoke.  The flame is a genuine diffusion flame: fuel evaporates from the
    wick, mixes with air, and burns with single-step Arrhenius chemistry
    r = A Yf Yo exp(-Ta/T), so the flame sheet sits where fuel and oxygen meet
    (dark fuel-rich core, blue reaction skirt at the base, yellow soot body).
  * Soot forms in hot fuel-rich gas and is oxidised at the flame tip; if a gust
    quenches the tip it escapes as a dark wisp.  Radiative heat loss scales with
    soot loading.
  * After blow-out the glowing wick keeps evaporating paraffin; the vapour
    condenses into the white smoke ribbon when it cools (exactly what real
    candle smoke is).
  * Temperature-dependent molecular diffusion and viscosity (implicit,
    red-black Gauss-Seidel) keep the flame laminar like a real candle, while
    vorticity confinement (restricted to cool smoke) keeps the plume detailed.
  * Pressure projection with a multigrid-preconditioned conjugate gradient
    (MGPCG).  Open (Dirichlet p=0) sides and top, solid floor.
  * Breaths of air are soft turbulent jets (curl-noise modulated) defined in
    candle/timeline.py.

The solver advances in physical time and writes the raw fields at the exact
simulated time of every output frame (speed ramps are applied by the timeline),
so look development (exposure, colour) can be re-done later without
re-simulating.  Checkpoints are written periodically and on SIGTERM/SIGINT;
re-running the same command resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import warp as wp

from . import config as C
from . import spectral
from . import timeline as TL

wp.config.quiet = True

T_AMB = 293.0
YO_AMB = 0.233
GRAV = 9.81

vec3 = wp.vec3


# =========================================================================== #
# Physical parameters of the flame model (tunable without touching kernels)
# =========================================================================== #
PHYS = dict(
    # chemistry
    A_fuel=5e7, Ta_fuel=11000.0, stoich=3.4, heat=26000.0,   # K per unit Yf burned
    A_soot=2.0e4, Ta_soot=8000.0, A_ox=3.5e6, Ta_ox=10000.0,
    soot_yield=1.0,
    # radiation [K/s at 2000 K]
    rad_gas=200.0, rad_soot=800.0,
    # transport
    D0=2.2e-5, nu0=1.6e-5, transport_exp=1.7, soot_D_frac=0.02,
    # expansion
    expansion=0.55, max_div=350.0,
    # wick
    pilot_T=1550.0,
    fuel_rate=5.5e-6,        # paraffin vapour volume flow at the wick [m^3/s of Yf=1 gas]
    wick_T=640.0, wick_emit_from=0.18,
    # smoke condensation
    T_cond=540.0, k_cond=60.0,
    # extinction by strain
    u_extinct=2.8,
    # ambient draft
    draft_speed=0.025,
)


# =========================================================================== #
# Helpers
# =========================================================================== #
@wp.func
def sample(f: wp.array3d(dtype=float), q: vec3) -> float:
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    x = wp.clamp(q[0], 0.0, float(nx - 1))
    y = wp.clamp(q[1], 0.0, float(ny - 1))
    z = wp.clamp(q[2], 0.0, float(nz - 1))
    i = wp.min(int(x), nx - 2)
    j = wp.min(int(y), ny - 2)
    k = wp.min(int(z), nz - 2)
    fx = x - float(i)
    fy = y - float(j)
    fz = z - float(k)
    c000 = f[i, j, k]
    c100 = f[i + 1, j, k]
    c010 = f[i, j + 1, k]
    c110 = f[i + 1, j + 1, k]
    c001 = f[i, j, k + 1]
    c101 = f[i + 1, j, k + 1]
    c011 = f[i, j + 1, k + 1]
    c111 = f[i + 1, j + 1, k + 1]
    c00 = c000 + (c100 - c000) * fx
    c10 = c010 + (c110 - c010) * fx
    c01 = c001 + (c101 - c001) * fx
    c11 = c011 + (c111 - c011) * fx
    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy
    return c0 + (c1 - c0) * fz


@wp.func
def sample_minmax(f: wp.array3d(dtype=float), q: vec3) -> wp.vec2:
    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    x = wp.clamp(q[0], 0.0, float(nx - 1))
    y = wp.clamp(q[1], 0.0, float(ny - 1))
    z = wp.clamp(q[2], 0.0, float(nz - 1))
    i = wp.min(int(x), nx - 2)
    j = wp.min(int(y), ny - 2)
    k = wp.min(int(z), nz - 2)
    mn = f[i, j, k]
    mx = mn
    for a in range(2):
        for b in range(2):
            for c in range(2):
                v = f[i + a, j + b, k + c]
                mn = wp.min(mn, v)
                mx = wp.max(mx, v)
    return wp.vec2(mn, mx)


@wp.func
def vel(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), X: vec3) -> vec3:
    return vec3(sample(u, X - vec3(0.0, 0.5, 0.5)),
                sample(v, X - vec3(0.5, 0.0, 0.5)),
                sample(w, X - vec3(0.5, 0.5, 0.0)))


@wp.func
def backtrace(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
              X0: vec3, dtdx: float) -> vec3:
    v1 = vel(u, v, w, X0)
    Xm = X0 - 0.5 * dtdx * v1
    v2 = vel(u, v, w, Xm)
    return X0 - dtdx * v2


@wp.func
def smoothstep(e0: float, e1: float, x: float) -> float:
    t = wp.clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# =========================================================================== #
# Advection
# =========================================================================== #
@wp.kernel
def k_advect_sl(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float),
                u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
                off: vec3, dtdx: float):
    i, j, k = wp.tid()
    X0 = vec3(float(i), float(j), float(k)) + off
    X = backtrace(u, v, w, X0, dtdx)
    dst[i, j, k] = sample(src, X - off)


@wp.kernel
def k_maccormack(src: wp.array3d(dtype=float), hat: wp.array3d(dtype=float), bar: wp.array3d(dtype=float),
                 dst: wp.array3d(dtype=float),
                 u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
                 off: vec3, dtdx: float):
    i, j, k = wp.tid()
    X0 = vec3(float(i), float(j), float(k)) + off
    X = backtrace(u, v, w, X0, dtdx)
    mm = sample_minmax(src, X - off)
    val = hat[i, j, k] + 0.5 * (src[i, j, k] - bar[i, j, k])
    if val < mm[0] or val > mm[1]:
        val = hat[i, j, k]
    dst[i, j, k] = val


# =========================================================================== #
# Chemistry, sources, thermodynamics
# =========================================================================== #
@wp.kernel
def k_sources(T: wp.array3d(dtype=float), Yf: wp.array3d(dtype=float), Yo: wp.array3d(dtype=float),
              wick_w: wp.array3d(dtype=float), wick_tip: wp.array3d(dtype=float),
              pilot: wp.array3d(dtype=float), pilot_T: float,
              evap: float, wick_fuel: float, wick_T: float, ember_T: float, burning: int,
              ignite: float, ign_c: vec3, ign_r: float, dt: float):
    i, j, k = wp.tid()
    ww = wick_w[i, j, k]
    if ww > 0.0:
        a = wp.min(1.0, ww)
        # prescribed vapour mass flow (resolution independent flame height)
        Yf[i, j, k] = wp.min(Yf[i, j, k] + wick_fuel * evap * ww * dt, 1.0)
        Yo[i, j, k] = Yo[i, j, k] * (1.0 - a)
        tt = wick_T
        tip = wick_tip[i, j, k]
        if burning == 0:
            # glowing ember at the top of the wick, cooling body below
            tt = wick_T + (ember_T - wick_T) * tip
            tt = wp.min(tt, ember_T)
        T[i, j, k] = T[i, j, k] + (tt - T[i, j, k]) * a
    # Flame holder: a burning candle's flame base is anchored ~1 mm beside the
    # wick, in the quench layer where conduction keeps the gas hot.  That layer
    # is thinner than any affordable voxel, so we keep the gas there at
    # ignition temperature wherever fuel and oxygen actually coexist.
    pw = pilot[i, j, k]
    if burning == 1 and pw > 0.0:
        mix = smoothstep(0.002, 0.03, Yf[i, j, k]) * smoothstep(0.005, 0.05, Yo[i, j, k])
        T[i, j, k] = wp.max(T[i, j, k], 293.0 + (pilot_T - 293.0) * pw * mix)
    if ignite > 0.0:
        X = vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
        d = wp.length(X - ign_c)
        if d < ign_r:
            T[i, j, k] = wp.max(T[i, j, k], 300.0 + ignite * 1900.0 * (1.0 - d / ign_r))


@wp.kernel
def k_react(T: wp.array3d(dtype=float), Yf: wp.array3d(dtype=float), Yo: wp.array3d(dtype=float),
            soot: wp.array3d(dtype=float), smoke: wp.array3d(dtype=float), react: wp.array3d(dtype=float),
            target_div: wp.array3d(dtype=float),
            u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
            dt: float, kill: float,
            A_fuel: float, Ta_fuel: float, stoich: float, heat: float,
            A_soot: float, Ta_soot: float, A_ox: float, Ta_ox: float, soot_yield: float,
            rad_gas: float, rad_soot: float, expansion: float, max_div: float,
            T_cond: float, k_cond: float, u_ext: float, smoke_on: float):
    i, j, k = wp.tid()
    t = wp.max(T[i, j, k], 250.0)
    yf = wp.max(Yf[i, j, k], 0.0)
    yo = wp.max(Yo[i, j, k], 0.0)
    s = wp.max(soot[i, j, k], 0.0)

    # local strain extinction (blow-out) -- centre velocity magnitude
    uc = 0.5 * (u[i, j, k] + u[i + 1, j, k])
    vc = 0.5 * (v[i, j, k] + v[i, j + 1, k])
    wc = 0.5 * (w[i, j, k] + w[i, j, k + 1])
    sp = wp.sqrt(uc * uc + vc * vc + wc * wc)
    ext = wp.exp(-(sp / u_ext) * (sp / u_ext) * (sp / u_ext) * (sp / u_ext))

    # fuel oxidation: dYf/dt = -k Yo Yf, exact exponential update, limited by O2
    kr = A_fuel * wp.exp(-Ta_fuel / t) * kill * ext
    burn = yf * (1.0 - wp.exp(-kr * yo * dt))
    burn = wp.min(burn, yo / stoich)
    yf = yf - burn
    yo = yo - stoich * burn
    dT = heat * burn

    # soot formation in hot fuel-rich gas (no oxygen), oxidation where O2 is present
    rich = smoothstep(0.02, 0.0, yo)
    form = soot_yield * A_soot * yf * wp.exp(-Ta_soot / t) * rich * dt
    s = s + form
    ox = s * (1.0 - wp.exp(-A_ox * yo * wp.exp(-Ta_ox / t) * dt))
    s = s - ox

    # radiative loss (soot dominated), implicit-ish clamp
    tn = t / 2000.0
    ta = 293.0 / 2000.0
    rad = (rad_gas + rad_soot * s) * (tn * tn * tn * tn - ta * ta * ta * ta) * dt
    t_new = t + dT - rad
    t_new = wp.max(t_new, 293.0)

    # condensation of paraffin vapour into white smoke when it cools
    # (a steadily burning candle does not smoke: any vapour that escapes the
    # flame on a coarse grid would otherwise condense into a grey haze)
    cnd = yf * (1.0 - wp.exp(-k_cond * smoke_on * smoothstep(T_cond + 90.0, T_cond - 90.0, t_new) * dt))
    yf = yf - cnd

    T[i, j, k] = t_new
    Yf[i, j, k] = yf
    Yo[i, j, k] = yo
    soot[i, j, k] = s
    smoke[i, j, k] = smoke[i, j, k] + cnd * 40.0
    react[i, j, k] = dT / dt
    target_div[i, j, k] = wp.min(expansion * (dT / dt) / t_new, max_div)


@wp.kernel
def k_sponge(T: wp.array3d(dtype=float), Yf: wp.array3d(dtype=float), Yo: wp.array3d(dtype=float),
             width: int, strength: float):
    i, j, k = wp.tid()
    nx = T.shape[0]
    ny = T.shape[1]
    nz = T.shape[2]
    d = wp.min(wp.min(i, nx - 1 - i), wp.min(j, ny - 1 - j))
    d = wp.min(d, nz - 1 - k)
    if d < width:
        a = strength * (1.0 - float(d) / float(width))
        T[i, j, k] = T[i, j, k] + (293.0 - T[i, j, k]) * a
        Yo[i, j, k] = Yo[i, j, k] + (0.233 - Yo[i, j, k]) * a
        Yf[i, j, k] = Yf[i, j, k] * (1.0 - a)


# =========================================================================== #
# Diffusion (implicit, red-black Gauss-Seidel, temperature dependent)
# =========================================================================== #
@wp.kernel
def k_diffuse_rbgs(phi: wp.array3d(dtype=float), phi0: wp.array3d(dtype=float), T: wp.array3d(dtype=float),
                   off: vec3, coef0: float, texp: float, dtdx2: float, color: int):
    i, j, k = wp.tid()
    if (i + j + k) % 2 != color:
        return
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    X = vec3(float(i), float(j), float(k)) + off
    t = wp.max(sample(T, X - vec3(0.5, 0.5, 0.5)), 250.0)
    a = coef0 * wp.pow(t / 300.0, texp) * dtdx2
    c = phi[i, j, k]
    s = phi[wp.max(i - 1, 0), j, k] + phi[wp.min(i + 1, nx - 1), j, k]
    s = s + phi[i, wp.max(j - 1, 0), k] + phi[i, wp.min(j + 1, ny - 1), k]
    s = s + phi[i, j, wp.max(k - 1, 0)] + phi[i, j, wp.min(k + 1, nz - 1)]
    phi[i, j, k] = (phi0[i, j, k] + a * s) / (1.0 + 6.0 * a)


# =========================================================================== #
# Forces
# =========================================================================== #
@wp.kernel
def k_buoyancy(w: wp.array3d(dtype=float), T: wp.array3d(dtype=float), soot: wp.array3d(dtype=float),
               smoke: wp.array3d(dtype=float), dt: float):
    i, j, k = wp.tid()
    nz = T.shape[2]
    if k == 0 or k >= nz:
        return
    t = 0.5 * (T[i, j, k - 1] + T[i, j, k])
    sm = 0.5 * (smoke[i, j, k - 1] + smoke[i, j, k])
    b = 9.81 * (t - 293.0) / t - 9.81 * 0.004 * sm
    w[i, j, k] = w[i, j, k] + b * dt


@wp.kernel
def k_center_vel(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
                 vc: wp.array3d(dtype=vec3)):
    i, j, k = wp.tid()
    vc[i, j, k] = vec3(0.5 * (u[i, j, k] + u[i + 1, j, k]),
                       0.5 * (v[i, j, k] + v[i, j + 1, k]),
                       0.5 * (w[i, j, k] + w[i, j, k + 1]))


@wp.kernel
def k_vorticity(vc: wp.array3d(dtype=vec3), om: wp.array3d(dtype=vec3), omag: wp.array3d(dtype=float), inv2dx: float):
    i, j, k = wp.tid()
    nx = vc.shape[0]
    ny = vc.shape[1]
    nz = vc.shape[2]
    xp = vc[wp.min(i + 1, nx - 1), j, k]
    xm = vc[wp.max(i - 1, 0), j, k]
    yp = vc[i, wp.min(j + 1, ny - 1), k]
    ym = vc[i, wp.max(j - 1, 0), k]
    zp = vc[i, j, wp.min(k + 1, nz - 1)]
    zm = vc[i, j, wp.max(k - 1, 0)]
    o = vec3((yp[2] - ym[2]) - (zp[1] - zm[1]),
             (zp[0] - zm[0]) - (xp[2] - xm[2]),
             (xp[1] - xm[1]) - (yp[0] - ym[0])) * inv2dx
    om[i, j, k] = o
    omag[i, j, k] = wp.length(o)


@wp.kernel
def k_confine(om: wp.array3d(dtype=vec3), omag: wp.array3d(dtype=float), T: wp.array3d(dtype=float),
              smoke: wp.array3d(dtype=float), soot: wp.array3d(dtype=float),
              force: wp.array3d(dtype=vec3), eps: float, dx: float):
    i, j, k = wp.tid()
    nx = om.shape[0]
    ny = om.shape[1]
    nz = om.shape[2]
    g = vec3(omag[wp.min(i + 1, nx - 1), j, k] - omag[wp.max(i - 1, 0), j, k],
             omag[i, wp.min(j + 1, ny - 1), k] - omag[i, wp.max(j - 1, 0), k],
             omag[i, j, wp.min(k + 1, nz - 1)] - omag[i, j, wp.max(k - 1, 0)])
    gl = wp.length(g)
    f = vec3(0.0, 0.0, 0.0)
    if gl > 1e-8:
        N = g / gl
        # only in cool, visible smoke: the flame itself stays laminar
        m = (1.0 - smoothstep(600.0, 1300.0, T[i, j, k]))
        m = m * wp.min(1.0, 4.0 * (smoke[i, j, k] + soot[i, j, k]) + 0.15)
        f = eps * dx * m * wp.cross(N, om[i, j, k])
    force[i, j, k] = f


@wp.kernel
def k_apply_center_force(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
                         force: wp.array3d(dtype=vec3), dt: float, comp: int):
    i, j, k = wp.tid()
    nx = force.shape[0]
    ny = force.shape[1]
    nz = force.shape[2]
    if comp == 0:
        if i == 0 or i >= nx:
            return
        u[i, j, k] = u[i, j, k] + 0.5 * (force[i - 1, j, k][0] + force[i, j, k][0]) * dt
    elif comp == 1:
        if j == 0 or j >= ny:
            return
        v[i, j, k] = v[i, j, k] + 0.5 * (force[i, j - 1, k][1] + force[i, j, k][1]) * dt
    else:
        if k == 0 or k >= nz:
            return
        w[i, j, k] = w[i, j, k] + 0.5 * (force[i, j, k - 1][2] + force[i, j, k][2]) * dt


@wp.kernel
def k_puff(f: wp.array3d(dtype=float), off: vec3, comp: int, dx: float, origin: vec3,
           dirn: vec3, centre: vec3, width: float, speed: float, turb: float,
           t: float, relax: float, seed: int, noise_scale: float):
    i, j, k = wp.tid()
    X = (vec3(float(i), float(j), float(k)) + off) * dx + origin
    rel = X - centre
    along = wp.dot(rel, dirn)
    perp = rel - along * dirn
    r2 = wp.dot(perp, perp) / (width * width)
    prof = wp.exp(-r2)
    if prof < 1e-3:
        return
    # turbulent modulation travelling with the jet
    q = (X - dirn * speed * t) / noise_scale
    state = wp.rand_init(seed)
    cn = wp.curlnoise(state, wp.vec4(q[0], q[1], q[2], t * 3.0), wp.uint32(2), 2.0, 0.5)
    target = dirn * speed * prof * (1.0 + 0.35 * turb * cn[0]) + cn * (0.45 * turb * speed * prof)
    cur = f[i, j, k]
    tc = target[comp]
    # relax only towards the jet (never decelerate faster-moving buoyant gas sideways)
    f[i, j, k] = cur + (tc - cur) * relax * prof


@wp.kernel
def k_draft(f: wp.array3d(dtype=float), target: float, relax: float):
    i, j, k = wp.tid()
    f[i, j, k] = f[i, j, k] + (target - f[i, j, k]) * relax


@wp.kernel
def k_wick_solid(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
                 solid: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    nx = solid.shape[0]
    ny = solid.shape[1]
    if solid[i, j, k] > 0.5:
        u[i, j, k] = 0.0
        u[i + 1, j, k] = 0.0
        v[i, j, k] = 0.0
        v[i, j + 1, k] = 0.0
        w[i, j, k] = 0.0
        w[i, j, k + 1] = 0.0


@wp.kernel
def k_floor(w: wp.array3d(dtype=float)):
    i, j = wp.tid()
    w[i, j, 0] = 0.0


# =========================================================================== #
# Pressure projection: MGPCG
# =========================================================================== #
@wp.func
def lap_diag_sum(x: wp.array3d(dtype=float), i: int, j: int, k: int) -> wp.vec2:
    # returns (diag, sum_of_inside_neighbours); Dirichlet ghost (p=0 face) on
    # sides and top, Neumann on the floor.
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]
    diag = float(0.0)
    s = float(0.0)
    if i > 0:
        diag += 1.0
        s += x[i - 1, j, k]
    else:
        diag += 2.0
    if i < nx - 1:
        diag += 1.0
        s += x[i + 1, j, k]
    else:
        diag += 2.0
    if j > 0:
        diag += 1.0
        s += x[i, j - 1, k]
    else:
        diag += 2.0
    if j < ny - 1:
        diag += 1.0
        s += x[i, j + 1, k]
    else:
        diag += 2.0
    if k > 0:
        diag += 1.0
        s += x[i, j, k - 1]
    if k < nz - 1:
        diag += 1.0
        s += x[i, j, k + 1]
    else:
        diag += 2.0
    return wp.vec2(diag, s)


@wp.kernel
def k_lap_matvec(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float), z: wp.array3d(dtype=float),
                 alpha: float, beta: float):
    i, j, k = wp.tid()
    ds = lap_diag_sum(x, i, j, k)
    ax = ds[0] * x[i, j, k] - ds[1]
    z[i, j, k] = alpha * ax + beta * y[i, j, k]


@wp.kernel
def k_mg_smooth(x: wp.array3d(dtype=float), b: wp.array3d(dtype=float), color: int):
    i, j, k = wp.tid()
    if (i + j + k) % 2 != color:
        return
    ds = lap_diag_sum(x, i, j, k)
    x[i, j, k] = (b[i, j, k] + ds[1]) / ds[0]


@wp.kernel
def k_mg_residual(x: wp.array3d(dtype=float), b: wp.array3d(dtype=float), r: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    ds = lap_diag_sum(x, i, j, k)
    r[i, j, k] = b[i, j, k] - (ds[0] * x[i, j, k] - ds[1])


@wp.kernel
def k_mg_restrict(r: wp.array3d(dtype=float), bc: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    s = float(0.0)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                s += r[2 * i + a, 2 * j + b, 2 * k + c]
    bc[i, j, k] = 0.5 * s


@wp.kernel
def k_mg_prolong_add(xc: wp.array3d(dtype=float), xf: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    xf[i, j, k] = xf[i, j, k] + xc[i // 2, j // 2, k // 2]


@wp.kernel
def k_axpby(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float), z: wp.array3d(dtype=float),
            alpha: float, beta: float):
    i, j, k = wp.tid()
    z[i, j, k] = alpha * x[i, j, k] + beta * y[i, j, k]


@wp.kernel
def k_divergence(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
                 target: wp.array3d(dtype=float), b: wp.array3d(dtype=float), dx: float):
    i, j, k = wp.tid()
    div = (u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]) / dx
    b[i, j, k] = -(div - target[i, j, k]) * dx * dx


@wp.kernel
def k_subtract_grad(phi: wp.array3d(dtype=float), f: wp.array3d(dtype=float), comp: int, inv_dx: float):
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    if comp == 0:
        a = float(0.0)
        b = float(0.0)
        if i == 0:
            b = phi[0, j, k]
            a = -b
        elif i == nx:
            a = phi[nx - 1, j, k]
            b = -a
        else:
            a = phi[i - 1, j, k]
            b = phi[i, j, k]
        f[i, j, k] = f[i, j, k] - (b - a) * inv_dx
    elif comp == 1:
        a = float(0.0)
        b = float(0.0)
        if j == 0:
            b = phi[i, 0, k]
            a = -b
        elif j == ny:
            a = phi[i, ny - 1, k]
            b = -a
        else:
            a = phi[i, j - 1, k]
            b = phi[i, j, k]
        f[i, j, k] = f[i, j, k] - (b - a) * inv_dx
    else:
        if k == 0:
            f[i, j, k] = 0.0
            return
        a = float(0.0)
        b = float(0.0)
        if k == nz:
            a = phi[i, j, nz - 1]
            b = -a
        else:
            a = phi[i, j, k - 1]
            b = phi[i, j, k]
        f[i, j, k] = f[i, j, k] - (b - a) * inv_dx


@wp.kernel
def k_absmax_chunk(a: wp.array(dtype=float), out: wp.array(dtype=float), n: int, chunk: int):
    t = wp.tid()
    s = t * chunk
    e = wp.min(s + chunk, n)
    m = float(0.0)
    for idx in range(s, e):
        m = wp.max(m, wp.abs(a[idx]))
    wp.atomic_max(out, 0, m)


@wp.kernel
def k_dot_chunk(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=wp.float64), n: int, chunk: int):
    t = wp.tid()
    s = t * chunk
    e = wp.min(s + chunk, n)
    acc = wp.float64(0.0)
    for idx in range(s, e):
        acc += wp.float64(a[idx] * b[idx])
    wp.atomic_add(out, 0, acc)


@wp.kernel
def k_cg_update_xr(x: wp.array3d(dtype=float), r: wp.array3d(dtype=float), p: wp.array3d(dtype=float),
                   ap: wp.array3d(dtype=float), alpha: float):
    i, j, k = wp.tid()
    x[i, j, k] = x[i, j, k] + alpha * p[i, j, k]
    r[i, j, k] = r[i, j, k] - alpha * ap[i, j, k]


@wp.kernel
def k_cg_update_p(p: wp.array3d(dtype=float), z: wp.array3d(dtype=float), beta: float):
    i, j, k = wp.tid()
    p[i, j, k] = z[i, j, k] + beta * p[i, j, k]


class MGPCG:
    """Conjugate gradient preconditioned with one geometric multigrid V-cycle.

    Cell-centred, red-black Gauss-Seidel smoothing (symmetric ordering so the
    preconditioner is SPD), piecewise-constant prolongation and its scaled
    transpose for restriction, re-discretised coarse operators.
    """

    def __init__(self, shape, levels, device):
        self.device = device
        self.shape = tuple(shape)
        self.levels = []
        s = tuple(shape)
        for l in range(levels):
            self.levels.append(dict(
                shape=s,
                x=wp.zeros(s, dtype=float, device=device),
                b=wp.zeros(s, dtype=float, device=device),
                r=wp.zeros(s, dtype=float, device=device),
            ))
            if any(d % 2 for d in s):
                break
            s = (s[0] // 2, s[1] // 2, s[2] // 2)
            if min(s) < 2:
                break
        self.n = shape[0] * shape[1] * shape[2]
        self.r = wp.zeros(self.shape, dtype=float, device=device)
        self.z = wp.zeros(self.shape, dtype=float, device=device)
        self.p = wp.zeros(self.shape, dtype=float, device=device)
        self.ap = wp.zeros(self.shape, dtype=float, device=device)
        self.acc = wp.zeros(1, dtype=wp.float64, device=device)
        self.last_iters = 0
        self.last_res = 0.0

    def dot(self, a, b):
        chunk = 2048
        self.acc.zero_()
        wp.launch(k_dot_chunk, dim=(self.n + chunk - 1) // chunk,
                  inputs=[a.flatten(), b.flatten(), self.acc, self.n, chunk], device=self.device)
        return float(self.acc.numpy()[0])

    def _vcycle(self, l):
        L = self.levels[l]
        x, b, r, s = L["x"], L["b"], L["r"], L["shape"]
        x.zero_()
        d = self.device
        if l == len(self.levels) - 1:
            for _ in range(20):
                wp.launch(k_mg_smooth, dim=s, inputs=[x, b, 0], device=d)
                wp.launch(k_mg_smooth, dim=s, inputs=[x, b, 1], device=d)
            for _ in range(20):
                wp.launch(k_mg_smooth, dim=s, inputs=[x, b, 1], device=d)
                wp.launch(k_mg_smooth, dim=s, inputs=[x, b, 0], device=d)
            return
        for _ in range(2):
            wp.launch(k_mg_smooth, dim=s, inputs=[x, b, 0], device=d)
            wp.launch(k_mg_smooth, dim=s, inputs=[x, b, 1], device=d)
        wp.launch(k_mg_residual, dim=s, inputs=[x, b, r], device=d)
        Lc = self.levels[l + 1]
        wp.launch(k_mg_restrict, dim=Lc["shape"], inputs=[r, Lc["b"]], device=d)
        self._vcycle(l + 1)
        wp.launch(k_mg_prolong_add, dim=s, inputs=[Lc["x"], x], device=d)
        for _ in range(2):
            wp.launch(k_mg_smooth, dim=s, inputs=[x, b, 1], device=d)
            wp.launch(k_mg_smooth, dim=s, inputs=[x, b, 0], device=d)

    def _precond(self, r, z):
        L0 = self.levels[0]
        wp.copy(L0["b"], r)
        self._vcycle(0)
        wp.copy(z, L0["x"])

    def solve(self, b, x, tol, maxiter):
        d = self.device
        S = self.shape
        r, z, p, ap = self.r, self.z, self.p, self.ap
        # r = b - A x  (warm start from previous pressure)
        wp.launch(k_lap_matvec, dim=S, inputs=[x, b, r, -1.0, 1.0], device=d)
        bnorm = math.sqrt(max(self.dot(b, b), 1e-300))
        rnorm = math.sqrt(self.dot(r, r))
        if rnorm <= tol * bnorm:
            self.last_iters, self.last_res = 0, rnorm / bnorm
            return 0, rnorm / bnorm
        self._precond(r, z)
        wp.copy(p, z)
        rz = self.dot(r, z)
        it = 0
        for it in range(1, maxiter + 1):
            wp.launch(k_lap_matvec, dim=S, inputs=[p, p, ap, 1.0, 0.0], device=d)
            pap = self.dot(p, ap)
            if pap <= 0:
                break
            alpha = rz / pap
            wp.launch(k_cg_update_xr, dim=S, inputs=[x, r, p, ap, alpha], device=d)
            rnorm = math.sqrt(max(self.dot(r, r), 0.0))
            if rnorm <= tol * bnorm:
                break
            self._precond(r, z)
            rz_new = self.dot(r, z)
            beta = rz_new / max(rz, 1e-300)
            rz = rz_new
            wp.launch(k_cg_update_p, dim=S, inputs=[p, z, beta], device=d)
        self.last_iters, self.last_res = it, rnorm / bnorm
        return it, rnorm / bnorm


# =========================================================================== #
# The solver
# =========================================================================== #
FIELDS = ["u", "v", "w", "T", "Yf", "Yo", "soot", "smoke", "phi"]


class FlameSolver:
    def __init__(self, preset: C.FlamePreset, geom: C.CandleGeometry, device=None):
        self.p = preset
        self.g = geom
        self.device = device or wp.get_preferred_device()
        nx, ny, nz = preset.res
        self.shape = (nx, ny, nz)
        self.dx = preset.dx
        # world-space origin of the domain corner: wick at x=y=0, floor at the pool surface
        self.origin = np.array([-nx * self.dx / 2, -ny * self.dx / 2, geom.pool_level], dtype=np.float64)
        d = self.device
        S = self.shape
        self.u = wp.zeros((nx + 1, ny, nz), dtype=float, device=d)
        self.v = wp.zeros((nx, ny + 1, nz), dtype=float, device=d)
        self.w = wp.zeros((nx, ny, nz + 1), dtype=float, device=d)
        self.tmp_vel = {c: [wp.zeros(a.shape, dtype=float, device=d) for _ in range(3)]
                        for c, a in (("u", self.u), ("v", self.v), ("w", self.w))}
        self.T = wp.full(S, T_AMB, dtype=float, device=d)
        self.Yf = wp.zeros(S, dtype=float, device=d)
        self.Yo = wp.full(S, YO_AMB, dtype=float, device=d)
        self.soot = wp.zeros(S, dtype=float, device=d)
        self.smoke = wp.zeros(S, dtype=float, device=d)
        self.react = wp.zeros(S, dtype=float, device=d)
        self.target_div = wp.zeros(S, dtype=float, device=d)
        self.phi = wp.zeros(S, dtype=float, device=d)
        self.rhs = wp.zeros(S, dtype=float, device=d)
        self.s_tmp = [wp.zeros(S, dtype=float, device=d) for _ in range(3)]
        self.vc = wp.zeros(S, dtype=vec3, device=d)
        self.om = wp.zeros(S, dtype=vec3, device=d)
        self.omag = wp.zeros(S, dtype=float, device=d)
        self.force = wp.zeros(S, dtype=vec3, device=d)
        self.maxbuf = wp.zeros(1, dtype=float, device=d)
        self.mg = MGPCG(S, preset.mg_levels, d)
        self.t = 0.0
        self.step_count = 0
        self.burning = True
        self.puffs = TL.puffs()
        self._build_wick()

    # ------------------------------------------------------------------ #
    def _build_wick(self):
        nx, ny, nz = self.shape
        g = self.g
        dx = self.dx
        h = g.wick_height
        # only a small box around the wick is non-zero: build it there (memory!)
        half = 0.006
        i0 = max(0, int((-half - self.origin[0]) / dx))
        i1 = min(nx, int((half - self.origin[0]) / dx) + 2)
        j0 = max(0, int((-half - self.origin[1]) / dx))
        j1 = min(ny, int((half + g.wick_lean - self.origin[1]) / dx) + 2)
        k1 = min(nz, int((h + 0.006) / dx) + 2)
        ii, jj, kk = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), np.arange(0, k1), indexing="ij")
        X = (ii + 0.5) * dx + self.origin[0]
        Y = (jj + 0.5) * dx + self.origin[1]
        Z = (kk + 0.5) * dx
        # wick axis: a curved segment from (0,0,0) up to (0, lean, h), lean ~ quadratic
        zc = np.clip(Z, 0, h)
        axis_y = g.wick_lean * (zc / h) ** 2
        dist = np.sqrt(X ** 2 + (Y - axis_y) ** 2 + np.maximum(Z - h, 0) ** 2)
        r_eff = max(g.wick_radius, 0.55 * dx)
        solid = (dist < r_eff).astype(np.float32)
        # fuel emitting surface: a shell around the upper part of the wick
        shell = np.clip(1.0 - (dist - r_eff) / (0.9 * dx), 0, 1) * (dist < r_eff + 0.9 * dx)
        emit = shell * (Z > PHYS["wick_emit_from"] * h) * (Z < h + 1.2 * dx)
        tip = np.clip(1.0 - np.abs(Z - h) / max(0.0025, 2 * dx), 0, 1) * (dist < r_eff + 1.5 * dx)
        # flame-holder shell around the upper wick (see k_sources)
        pr0, pr1 = r_eff + 0.5 * dx, r_eff + max(0.0018, 1.6 * dx)
        pil = np.clip((pr1 - dist) / max(pr1 - pr0, 1e-9), 0, 1) * (dist > r_eff)
        pil *= (Z > 0.35 * h) * (Z < h + max(0.002, 1.5 * dx))
        if emit.sum() == 0:
            raise RuntimeError("wick not resolved by grid")

        def full(sub):
            a = np.zeros(self.shape, np.float32)
            a[i0:i1, j0:j1, 0:k1] = sub
            return wp.array(a, dtype=float, device=self.device)
        self.wick_w = full(emit)
        self.wick_w_sum = float(emit.sum())
        self.wick_solid = full(solid)
        self.wick_tip = full(tip)
        self.pilot = full(pil)
        self.wick_tip_index = np.array([(-self.origin[0]) / dx, (g.wick_lean - self.origin[1]) / dx, h / dx])

    # ------------------------------------------------------------------ #
    def _absmax(self, a):
        flat = a.flatten()
        n = flat.shape[0]
        chunk = 1024
        self.maxbuf.zero_()
        wp.launch(k_absmax_chunk, dim=(n + chunk - 1) // chunk, inputs=[flat, self.maxbuf, n, chunk], device=self.device)
        return float(self.maxbuf.numpy()[0])

    def max_speed(self):
        return max(self._absmax(self.u), self._absmax(self.v), self._absmax(self.w))

    # ------------------------------------------------------------------ #
    def _advect_scalar(self, name, dtdx):
        src = getattr(self, name)
        hat, bar, new = self.s_tmp
        off = vec3(0.5, 0.5, 0.5)
        args = [self.u, self.v, self.w, off]
        wp.launch(k_advect_sl, dim=self.shape, inputs=[src, hat] + args + [dtdx], device=self.device)
        wp.launch(k_advect_sl, dim=self.shape, inputs=[hat, bar] + args + [-dtdx], device=self.device)
        wp.launch(k_maccormack, dim=self.shape, inputs=[src, hat, bar, new] + args + [dtdx], device=self.device)
        setattr(self, name, new)
        self.s_tmp = [hat, bar, src]

    def _advect_velocity(self, dtdx):
        offs = {"u": vec3(0.0, 0.5, 0.5), "v": vec3(0.5, 0.0, 0.5), "w": vec3(0.5, 0.5, 0.0)}
        new = {}
        for c in ("u", "v", "w"):
            src = getattr(self, c)
            hat, bar, out = self.tmp_vel[c]
            args = [self.u, self.v, self.w, offs[c]]
            wp.launch(k_advect_sl, dim=src.shape, inputs=[src, hat] + args + [dtdx], device=self.device)
            wp.launch(k_advect_sl, dim=src.shape, inputs=[hat, bar] + args + [-dtdx], device=self.device)
            wp.launch(k_maccormack, dim=src.shape, inputs=[src, hat, bar, out] + args + [dtdx], device=self.device)
            new[c] = out
        for c in ("u", "v", "w"):
            old = getattr(self, c)
            hat, bar, out = self.tmp_vel[c]
            setattr(self, c, new[c])
            self.tmp_vel[c] = [hat, bar, old]

    def _diffuse(self, field, coef0, dt, off=vec3(0.5, 0.5, 0.5), iters=None):
        iters = iters or self.p.diffusion_iters
        f0 = self.s_tmp[0] if field.shape == self.shape else self.tmp_vel_for(field)
        wp.copy(f0, field)
        dtdx2 = dt / (self.dx * self.dx)
        for _ in range(iters):
            wp.launch(k_diffuse_rbgs, dim=field.shape,
                      inputs=[field, f0, self.T, off, coef0, PHYS["transport_exp"], dtdx2, 0], device=self.device)
            wp.launch(k_diffuse_rbgs, dim=field.shape,
                      inputs=[field, f0, self.T, off, coef0, PHYS["transport_exp"], dtdx2, 1], device=self.device)

    def tmp_vel_for(self, field):
        for c in ("u", "v", "w"):
            if getattr(self, c) is field:
                return self.tmp_vel[c][0]
        raise KeyError

    # ------------------------------------------------------------------ #
    def _world_to_index(self, p):
        return (np.asarray(p) - self.origin) / self.dx

    def step(self, dt):
        d = self.device
        S = self.shape
        dx = self.dx
        dtdx = dt / dx
        t = self.t

        self._advect_velocity(dtdx)
        for name in ("T", "Yf", "Yo", "soot", "smoke"):
            self._advect_scalar(name, dtdx)

        # --- sources ------------------------------------------------------
        tb = TL.blow_time()
        if self.burning and t > tb + 0.12:
            self.burning = False
        evap = TL.wick_evaporation(t)
        fuel_density_rate = PHYS["fuel_rate"] / (self.wick_w_sum * dx ** 3)
        ember = TL.ember_temperature(t)
        ignite = 1.0 if t < 0.06 else 0.0
        ign_c = vec3(*self.wick_tip_index + np.array([0, 0, 0.003 / dx]))
        wp.launch(k_sources, dim=S, inputs=[self.T, self.Yf, self.Yo, self.wick_w, self.wick_tip,
                                             self.pilot, PHYS["pilot_T"], evap,
                                             fuel_density_rate, PHYS["wick_T"], ember, int(self.burning),
                                             ignite, ign_c, 0.003 / dx, dt], device=d)

        # --- chemistry ----------------------------------------------------
        smoke_on = 0.0 if t < tb else min(1.0, (t - tb) / 0.05)
        kill = 1.0
        if t > tb + 0.015:
            kill = max(0.0, 1.0 - (t - tb - 0.015) / 0.06)
        wp.launch(k_react, dim=S, inputs=[self.T, self.Yf, self.Yo, self.soot, self.smoke, self.react,
                                           self.target_div, self.u, self.v, self.w, dt, kill,
                                           PHYS["A_fuel"], PHYS["Ta_fuel"], PHYS["stoich"], PHYS["heat"],
                                           PHYS["A_soot"], PHYS["Ta_soot"], PHYS["A_ox"], PHYS["Ta_ox"],
                                           PHYS["soot_yield"], PHYS["rad_gas"], PHYS["rad_soot"],
                                           PHYS["expansion"], PHYS["max_div"], PHYS["T_cond"], PHYS["k_cond"],
                                           PHYS["u_extinct"], smoke_on], device=d)
        wp.launch(k_sponge, dim=S, inputs=[self.T, self.Yf, self.Yo, 4, 0.25], device=d)

        # --- molecular transport -------------------------------------------
        self._diffuse(self.T, PHYS["D0"], dt)
        self._diffuse(self.Yf, PHYS["D0"], dt)
        self._diffuse(self.Yo, PHYS["D0"], dt)
        self._diffuse(self.soot, PHYS["D0"] * PHYS["soot_D_frac"], dt, iters=1)

        # --- forces ---------------------------------------------------------
        wp.launch(k_buoyancy, dim=self.w.shape, inputs=[self.w, self.T, self.soot, self.smoke, dt], device=d)
        if self.p.vorticity > 0:
            wp.launch(k_center_vel, dim=S, inputs=[self.u, self.v, self.w, self.vc], device=d)
            wp.launch(k_vorticity, dim=S, inputs=[self.vc, self.om, self.omag, 1.0 / (2 * dx)], device=d)
            wp.launch(k_confine, dim=S, inputs=[self.om, self.omag, self.T, self.smoke, self.soot, self.force,
                                                 self.p.vorticity, dx], device=d)
            for comp, f in enumerate((self.u, self.v, self.w)):
                wp.launch(k_apply_center_force, dim=f.shape, inputs=[self.u, self.v, self.w, self.force, dt, comp],
                          device=d)
        self._apply_puffs(dt)
        self._apply_draft(dt)

        # --- viscosity -------------------------------------------------------
        if self.p.diffusion_iters >= 4:
            for c, off in (("u", vec3(0.0, 0.5, 0.5)), ("v", vec3(0.5, 0.0, 0.5)), ("w", vec3(0.5, 0.5, 0.0))):
                self._diffuse(getattr(self, c), PHYS["nu0"], dt, off=off, iters=max(2, self.p.diffusion_iters // 2))

        wp.launch(k_wick_solid, dim=S, inputs=[self.u, self.v, self.w, self.wick_solid], device=d)
        wp.launch(k_floor, dim=(S[0], S[1]), inputs=[self.w], device=d)

        # --- projection -------------------------------------------------------
        wp.launch(k_divergence, dim=S, inputs=[self.u, self.v, self.w, self.target_div, self.rhs, dx], device=d)
        it, res = self.mg.solve(self.rhs, self.phi, self.p.cg_tol, self.p.cg_maxiter)
        for comp, f in enumerate((self.u, self.v, self.w)):
            wp.launch(k_subtract_grad, dim=f.shape, inputs=[self.phi, f, comp, 1.0 / dx], device=d)
        wp.launch(k_wick_solid, dim=S, inputs=[self.u, self.v, self.w, self.wick_solid], device=d)

        self.t += dt
        self.step_count += 1
        return it

    def _apply_puffs(self, dt):
        t = self.t
        for k, pf in enumerate(self.puffs):
            env = pf.envelope(t)
            if env <= 0:
                continue
            dirn = np.array([-math.cos(pf.azimuth), -math.sin(pf.azimuth), 0.04])
            dirn /= np.linalg.norm(dirn)
            centre = np.array([0.0, 0.0, self.origin[2] + pf.height])
            relax = min(1.0, dt / 0.012)
            for comp, (f, off) in enumerate(((self.u, vec3(0.0, 0.5, 0.5)), (self.v, vec3(0.5, 0.0, 0.5)),
                                              (self.w, vec3(0.5, 0.5, 0.0)))):
                if comp == 2:
                    # the jet barely pushes vertically; keep buoyancy in charge
                    continue
                wp.launch(k_puff, dim=f.shape, inputs=[f, off, comp, self.dx, vec3(*self.origin), vec3(*dirn),
                                                       vec3(*centre), pf.width, pf.speed * env, pf.turbulence,
                                                       t, relax, 1234 + 17 * k, 0.009], device=self.device)

    def _apply_draft(self, dt):
        # a barely perceptible room draft that wanders slowly
        t = self.t
        ax = PHYS["draft_speed"] * (math.sin(0.7 * t + 0.3) + 0.6 * math.sin(1.9 * t + 1.1) + 0.3 * math.sin(4.3 * t))
        ay = PHYS["draft_speed"] * (math.sin(0.5 * t + 2.0) + 0.5 * math.sin(2.3 * t + 0.4))
        relax = min(1.0, dt / 1.5)
        wp.launch(k_draft, dim=self.u.shape, inputs=[self.u, ax, relax], device=self.device)
        wp.launch(k_draft, dim=self.v.shape, inputs=[self.v, ay, relax], device=self.device)

    # ------------------------------------------------------------------ #
    def state_dict(self):
        out = {n: getattr(self, n).numpy() for n in FIELDS}
        out["meta"] = np.array([self.t, self.step_count, int(self.burning)], dtype=np.float64)
        return out

    def load_state(self, st):
        for n in FIELDS:
            a = getattr(self, n)
            a.assign(st[n].astype(np.float32))
        self.t = float(st["meta"][0])
        self.step_count = int(st["meta"][1])
        self.burning = bool(int(st["meta"][2]))


# =========================================================================== #
# Export
# =========================================================================== #
class Exporter:
    """Writes per-frame cropped fields (float16, zlib) on background threads."""

    def __init__(self, out_dir, solver: FlameSolver, nthreads=4):
        self.out_dir = out_dir
        self.s = solver
        self.q = queue.Queue(maxsize=nthreads * 2)
        self.threads = [threading.Thread(target=self._work, daemon=True) for _ in range(nthreads)]
        temps, lut = spectral.soot_lut()
        self.lut_t = temps
        self.lut_lum = (lut @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float32)
        self.errors = []
        for t in self.threads:
            t.start()

    def submit(self, frame, t):
        s = self.s
        data = dict(T=s.T.numpy(), soot=s.soot.numpy(), smoke=s.smoke.numpy(), react=s.react.numpy())
        self.q.put((frame, t, data))

    def _work(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            try:
                self._write(*item)
            except Exception as e:  # pragma: no cover
                self.errors.append(repr(e))
            finally:
                self.q.task_done()

    def _write(self, frame, t, d):
        s = self.s
        T, soot, smoke, react = d["T"], d["soot"], d["smoke"], d["react"]
        mask = (soot > 2e-3) | (smoke > 2e-3) | (react > 50.0)
        stats = dict(frame=frame, t=t, burning=bool(s.burning or t < TL.blow_time() + 0.12))
        if mask.any():
            idx = np.nonzero(mask)
            lo = np.maximum(np.array([a.min() for a in idx]) - 2, 0)
            hi = np.minimum(np.array([a.max() for a in idx]) + 3, np.array(T.shape))
        else:
            lo = np.array([0, 0, 0])
            hi = np.array([2, 2, 2])
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        Tc, sc, smc, rc = T[sl], soot[sl], smoke[sl], react[sl]
        # emission weights for statistics (luminance): soot glow + small blue term
        lum = sc * np.interp(Tc, self.lut_t, self.lut_lum).astype(np.float32)
        tot = float(lum.sum())
        stats["lum_total"] = tot * s.dx ** 3
        stats["react_total"] = float(rc.sum()) * s.dx ** 3
        if tot > 0:
            ii, jj, kk = np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi)), indexing="ij")
            wx = ((ii + 0.5) * s.dx + s.origin[0])
            wy = ((jj + 0.5) * s.dx + s.origin[1])
            wz = ((kk + 0.5) * s.dx + s.origin[2])
            cx, cy, cz = (float((lum * a).sum() / tot) for a in (wx, wy, wz))
            stats["centroid"] = [cx, cy, cz]
            # three light proxies: split luminance along height into terciles
            zprof = lum.sum(axis=(0, 1))
            cdf = np.cumsum(zprof) / max(zprof.sum(), 1e-30)
            bands = []
            edges = [0, int(np.searchsorted(cdf, 1 / 3)), int(np.searchsorted(cdf, 2 / 3)), len(cdf)]
            for b0, b1 in zip(edges[:-1], edges[1:]):
                b1 = max(b1, b0 + 1)
                L = lum[:, :, b0:b1]
                tl = float(L.sum())
                if tl <= 0:
                    continue
                bands.append(dict(
                    pos=[float((L * wx[:, :, b0:b1]).sum() / tl), float((L * wy[:, :, b0:b1]).sum() / tl),
                         float((L * wz[:, :, b0:b1]).sum() / tl)],
                    frac=tl / tot))
            stats["bands"] = bands
            # flame height (95th percentile of luminance along z) and width
            stats["height"] = float(wz[0, 0, min(int(np.searchsorted(cdf, 0.97)), len(cdf) - 1)] - s.origin[2])
        else:
            stats["centroid"] = None
            stats["bands"] = []
        path = os.path.join(self.out_dir, f"fields_{frame:04d}.npz")
        tmp = path + ".tmp.npz"
        dt = np.float16
        np.savez_compressed(
            tmp,
            T=np.where(mask[sl] | (rc > 1), Tc, 293.0).astype(dt),
            soot=np.where(sc > 1e-3, sc, 0).astype(dt),
            smoke=np.where(smc > 1e-3, smc, 0).astype(dt),
            react=(rc / 1e4).astype(dt),
            lo=lo.astype(np.int32), shape=np.array(T.shape, np.int32),
            origin=s.origin, dx=np.array([s.dx]))
        os.replace(tmp, path)
        with open(os.path.join(self.out_dir, f"stats_{frame:04d}.json"), "w") as f:
            json.dump(stats, f)

    def flush(self):
        self.q.join()
        if self.errors:
            raise RuntimeError("export failed: " + "; ".join(self.errors))

    def close(self):
        self.flush()
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join()
        self.threads = []


# =========================================================================== #
# Driver with checkpoint / resume
# =========================================================================== #
def run(out_dir, quality="preview", device=None, checkpoint_every=600.0, max_frames=None, log_every=50):
    preset, _, _ = C.get_preset(quality)
    geom = C.CandleGeometry()
    os.makedirs(out_dir, exist_ok=True)
    ck_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ck_dir, exist_ok=True)
    wp.init()
    solver = FlameSolver(preset, geom, device=device)
    print(f"[flame] device={solver.device} grid={solver.shape} dx={solver.dx*1e3:.3f} mm "
          f"domain={np.array(solver.shape)*solver.dx*100} cm", flush=True)

    frame_times = TL.frame_times()
    n_frames = len(frame_times) if max_frames is None else min(max_frames, len(frame_times))

    # ---- resume -----------------------------------------------------------
    ck_path = os.path.join(ck_dir, "state.npz")
    next_frame = 0
    if os.path.exists(ck_path):
        st = np.load(ck_path)
        solver.load_state(st)
        next_frame = int(st["next_frame"][0])
        print(f"[flame] resumed from checkpoint t={solver.t:.4f}s next_frame={next_frame}", flush=True)
    if next_frame >= n_frames:
        print("[flame] all frames already simulated", flush=True)
        return

    exporter = Exporter(out_dir, solver)
    stop = {"flag": False}

    def _sig(signum, frame):
        print(f"[flame] signal {signum}: checkpointing and exiting", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    def save_ck(nf):
        exporter.flush()
        st = solver.state_dict()
        st["next_frame"] = np.array([nf])
        tmp = ck_path + ".tmp.npz"
        np.savez(tmp, **st)
        os.replace(tmp, ck_path)

    wall0 = time.time()
    last_ck = time.time()
    umax = max(solver.max_speed(), 0.3)
    steps_since = 0
    dead_since = None
    t_log = time.time()
    while next_frame < n_frames:
        target = frame_times[next_frame]
        dt = min(preset.max_dt, preset.cfl * solver.dx / max(umax, 0.05))
        if solver.t + dt >= target - 1e-9:
            dt = target - solver.t
        if dt > 1e-7:
            solver.step(dt)
            steps_since += 1
            if solver.step_count % 4 == 0:
                umax = max(solver.max_speed(), 0.3)
                if not math.isfinite(umax):
                    raise FloatingPointError(f"solver blew up at t={solver.t}")
            # watchdog: a flame that dies before the scripted blow-out would waste
            # the whole render -- stop early and loudly instead.
            if solver.step_count % 25 == 0 and solver.burning and solver.t > 0.3:
                rmax = solver._absmax(solver.react)
                dead_since = dead_since if rmax < 1e3 else None
                if rmax < 1e3:
                    dead_since = dead_since or solver.t
                    if solver.t - dead_since > 0.25:
                        save_ck(next_frame)
                        raise RuntimeError(f"flame extinguished unexpectedly at t={solver.t:.3f}s; "
                                           f"see docs/TUNING.md")
        if solver.t >= target - 1e-7:
            exporter.submit(next_frame, solver.t)
            next_frame += 1
            if next_frame % log_every == 0 or next_frame == n_frames:
                el = time.time() - wall0
                print(f"[flame] frame {next_frame}/{n_frames} t={solver.t:.3f}s steps={solver.step_count} "
                      f"umax={umax:.2f} m/s  wall={el/60:.1f} min  ({(time.time()-t_log)/max(steps_since,1)*1e3:.0f} ms/step)",
                      flush=True)
                t_log = time.time()
                steps_since = 0
        if time.time() - last_ck > checkpoint_every:
            save_ck(next_frame)
            last_ck = time.time()
        if stop["flag"]:
            save_ck(next_frame)
            exporter.close()
            sys.exit(130)
    save_ck(next_frame)
    exporter.close()
    with open(os.path.join(out_dir, "DONE"), "w") as f:
        f.write(json.dumps(dict(frames=n_frames, t=solver.t, steps=solver.step_count,
                                wall_min=(time.time() - wall0) / 60, grid=list(solver.shape), dx=solver.dx,
                                origin=list(solver.origin))))
    print(f"[flame] done: {solver.step_count} steps in {(time.time()-wall0)/60:.1f} min", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", default="preview")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: best available)")
    ap.add_argument("--checkpoint-every", type=float, default=600.0, help="seconds of wall time")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.out, a.quality, a.device, a.checkpoint_every, a.max_frames)


if __name__ == "__main__":
    main()
