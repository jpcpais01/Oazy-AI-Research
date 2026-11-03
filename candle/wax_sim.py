"""Molten wax: melt pool, rim breach (weir overflow) and drips running down the
candle as a cooling thin film that thickens, slows and freezes.

Physics
-------
* Side wall: lubrication (thin-film) equations on the unrolled cylinder
  (azimuth x = R*theta, height z).  Gravity-driven flux per unit width
  q = rho g (h - h_pin)^3 / (3 mu(T)), first-order upwind with positivity
  limiting.  h_pin models contact-line pinning: a thin residual trail stays
  behind and freezes, exactly like real drip tracks.
* Paraffin viscosity mu(T): Arrhenius-like in the liquid, with a steep rise in
  the mushy zone close to the freezing point.
* The film carries heat (energy h*T is advected with the flux) and loses it to
  the colder candle wall through its own thickness, tau ~ h^2 / alpha, so thin
  leading edges freeze first while the thick bead keeps running: the classic
  fat-bulb drip.
* Below the solidus, liquid converts into solid deposit, which adds to the
  candle radius.  Wax reaching the table pools around the base.
* Melt pool: liquid level L(t) with a weir overflow q_w ~ (L - H_rim)^1.5 over
  the rim crest H_rim(theta).  Overflowing hot wax erodes its own channel
  (positive feedback), so a notch opens into a spout.  The rim is heated more
  on the side the flame leans to (read from the flame solver's statistics).
* Before the shot a "history" pass runs a dozen random spills to freezing to
  build up the old drips that cover a well-used pillar candle.

Outputs one compressed .npz per output frame with the deposit, film thickness,
film temperature, rim crest profile and pool level.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import warp as wp

from . import config as C
from . import timeline as TL

wp.config.quiet = True

RHO = 880.0
G = 9.81
ALPHA = 1.3e-7          # thermal diffusivity of paraffin [m^2/s]
T_POOL = 78.0           # degC
T_AMB = 24.0
T_LIQ = 61.0            # liquidus
T_SOL = 54.0            # below this the film freezes fast
RHO_G3 = wp.constant(RHO * G / 3.0)


@wp.func
def viscosity(T: float) -> float:
    mu = 0.0055 * wp.exp(-0.022 * (T - 80.0))
    mush = wp.max(0.0, 63.0 - T)
    return mu * wp.exp(0.9 * mush)


@wp.kernel
def k_flux(h: wp.array2d(dtype=float), E: wp.array2d(dtype=float), q: wp.array2d(dtype=float),
           qe: wp.array2d(dtype=float), h_pin: float, dt: float, dz: float, cmax: wp.array(dtype=float)):
    # face k is between cell k (above) and k-1 (below); flow is downward
    # face 0 drains the lowest cell onto the table
    i, k = wp.tid()
    hh = h[i, k]
    T = 20.0
    if hh > 1e-9:
        T = E[i, k] / hh
    he = wp.max(hh - h_pin, 0.0)
    mu = viscosity(T)
    flux = RHO_G3 * he * he * he / mu
    # positivity: never move more than 60 % of the mobile liquid in one step
    lim = 0.6 * he * dz / dt
    flux = wp.min(flux, lim)
    q[i, k] = flux
    qe[i, k] = flux * T
    c = 3.0 * RHO_G3 * he * he / mu
    wp.atomic_max(cmax, 0, c)



@wp.kernel
def k_update(h: wp.array2d(dtype=float), E: wp.array2d(dtype=float), q: wp.array2d(dtype=float),
             qe: wp.array2d(dtype=float), h2: wp.array2d(dtype=float), E2: wp.array2d(dtype=float),
             dt: float, dz: float):
    i, k = wp.tid()
    nz = h.shape[1]
    qin = float(0.0)
    qein = float(0.0)
    if k < nz - 1:
        qin = q[i, k + 1]
        qein = qe[i, k + 1]
    h2[i, k] = h[i, k] + dt / dz * (qin - q[i, k])
    E2[i, k] = E[i, k] + dt / dz * (qein - qe[i, k])


@wp.kernel
def k_table(q: wp.array2d(dtype=float), table: wp.array(dtype=float), dt: float):
    i = wp.tid()
    table[i] = table[i] + q[i, 0] * dt


@wp.kernel
def k_lateral(h: wp.array2d(dtype=float), E: wp.array2d(dtype=float), h2: wp.array2d(dtype=float),
              E2: wp.array2d(dtype=float), D: float, dt: float, dxl: float, h_pin: float):
    # capillary spreading across the rivulet (periodic in theta), mobile liquid only
    i, k = wp.tid()
    n = h.shape[0]
    ip = (i + 1) % n
    im = (i + n - 1) % n
    a = D * dt / (dxl * dxl)
    hc = wp.max(h[i, k] - h_pin, 0.0)
    hp = wp.max(h[ip, k] - h_pin, 0.0)
    hm = wp.max(h[im, k] - h_pin, 0.0)
    fp = a * (hp - hc)
    fm = a * (hm - hc)
    Tc = 20.0
    if h[i, k] > 1e-9:
        Tc = E[i, k] / h[i, k]
    Tp = Tc
    if h[ip, k] > 1e-9:
        Tp = E[ip, k] / h[ip, k]
    Tm = Tc
    if h[im, k] > 1e-9:
        Tm = E[im, k] / h[im, k]
    ep = fp * Tc
    if fp > 0.0:
        ep = fp * Tp
    em = fm * Tc
    if fm > 0.0:
        em = fm * Tm
    h2[i, k] = h[i, k] + fp + fm
    E2[i, k] = E[i, k] + ep + em


@wp.kernel
def k_thermal(h: wp.array2d(dtype=float), E: wp.array2d(dtype=float), dep: wp.array2d(dtype=float),
              inflow: wp.array(dtype=float), inflow_T: float, Twall_top: float, z_top: float,
              dz: float, dt: float, alpha: float, Tsol: float, Tliq: float, Tamb: float, h_pin: float):
    i, k = wp.tid()
    nz = h.shape[1]
    hh = h[i, k]
    e = E[i, k]
    if k == nz - 1:
        add = inflow[i] * dt / dz
        hh = hh + add
        e = e + add * inflow_T
    if hh <= 1e-10:
        h[i, k] = 0.0
        E[i, k] = 0.0
        return
    T = e / hh
    z = (float(k) + 0.5) * dz
    Tw = Tamb + (Twall_top - Tamb) * wp.exp(-(z_top - z) / 0.010)
    tau = 0.55 * hh * hh / alpha + 0.03
    # latent heat buffers cooling in the mushy zone
    lat = 1.0
    if T < Tliq:
        lat = 0.35
    T = Tw + (T - Tw) * wp.exp(-dt * lat / tau)
    # freezing: liquid becomes deposit
    if T < Tliq:
        fr = wp.min(1.0, (1.6 * (Tliq - T) + 0.3) * dt)
        if T < Tsol:
            fr = wp.min(1.0, fr * 4.0)
        frozen = hh * fr
        # a thin pinned film freezes outright once cold
        if hh < h_pin * 1.2 and T < Tliq - 1.0:
            frozen = hh
        dep[i, k] = dep[i, k] + frozen
        hh = hh - frozen
    h[i, k] = hh
    E[i, k] = hh * T


@wp.kernel
def k_absmax(a: wp.array2d(dtype=float), out: wp.array(dtype=float)):
    i, k = wp.tid()
    wp.atomic_max(out, 0, a[i, k])


def periodic_noise(n, rng, octaves=((3, 1.0), (7, 0.5), (15, 0.25), (31, 0.12))):
    th = np.arange(n) * 2 * np.pi / n
    out = np.zeros(n)
    for f, a in octaves:
        out += a * np.sin(f * th + rng.uniform(0, 2 * np.pi))
    return out / sum(a for _, a in octaves)


class WaxSim:
    def __init__(self, preset: C.WaxPreset, geom: C.CandleGeometry, device=None, seed=7):
        self.p = preset
        self.g = geom
        self.device = device or wp.get_preferred_device()
        self.rng = np.random.default_rng(seed)
        n, nz = preset.ntheta, preset.nz
        self.n, self.nz = n, nz
        self.R = geom.radius
        self.dtheta = 2 * np.pi / n
        self.dxl = self.R * self.dtheta
        # film grid covers the side wall up to the start of the rim rounding
        self.z_top = geom.rim_height - 0.0015
        self.dz = self.z_top / nz
        self.h_pin = 0.07e-3
        self.theta = np.arange(n) * self.dtheta
        d = self.device
        self.h = wp.zeros((n, nz), dtype=float, device=d)
        self.E = wp.zeros((n, nz), dtype=float, device=d)
        self.h2 = wp.zeros((n, nz), dtype=float, device=d)
        self.E2 = wp.zeros((n, nz), dtype=float, device=d)
        self.q = wp.zeros((n, nz), dtype=float, device=d)
        self.qe = wp.zeros((n, nz), dtype=float, device=d)
        self.dep = wp.zeros((n, nz), dtype=float, device=d)
        self.inflow = wp.zeros(n, dtype=float, device=d)
        self.table = wp.zeros(n, dtype=float, device=d)       # wax that reached the table (m^2 per unit length)
        self.cmax = wp.zeros(1, dtype=float, device=d)
        # rim crest profile
        rim = geom.rim_height + 0.0007 * periodic_noise(n, self.rng)
        dth = np.angle(np.exp(1j * (self.theta - geom.notch_azimuth)))
        self.notch_shape = np.exp(-(dth / geom.notch_width) ** 2)
        rim -= geom.notch_depth * self.notch_shape
        self.rim = rim
        self.L = geom.pool_level
        self.pool_area = np.pi * geom.pool_radius ** 2
        self.wet = np.zeros(n)
        self.t = 0.0
        self.spilled = 0.0

    # ------------------------------------------------------------------ #
    def film_step(self, dt, inflow, inflow_T=T_POOL, wall_top=50.0):
        d = self.device
        n, nz = self.n, self.nz
        self.inflow.assign(inflow.astype(np.float32))
        wp.launch(k_thermal, dim=(n, nz), inputs=[self.h, self.E, self.dep, self.inflow, inflow_T, wall_top,
                                                  self.z_top, self.dz, dt, ALPHA, T_SOL, T_LIQ, T_AMB, self.h_pin],
                  device=d)
        self.cmax.zero_()
        wp.launch(k_flux, dim=(n, nz), inputs=[self.h, self.E, self.q, self.qe, self.h_pin, dt, self.dz, self.cmax],
                  device=d)
        wp.launch(k_table, dim=n, inputs=[self.q, self.table, dt], device=d)
        wp.launch(k_update, dim=(n, nz), inputs=[self.h, self.E, self.q, self.qe, self.h2, self.E2, dt, self.dz],
                  device=d)
        wp.launch(k_lateral, dim=(n, nz), inputs=[self.h2, self.E2, self.h, self.E, 1.2e-6, dt, self.dxl, self.h_pin],
                  device=d)
        return float(self.cmax.numpy()[0])

    def stable_dt(self, c, dt_max):
        return min(dt_max, 0.45 * self.dz / max(c, 1e-6), 0.2 * self.dxl ** 2 / 1.2e-6)

    # ------------------------------------------------------------------ #
    def history(self, events):
        """Old spills that froze long ago: run them all, to completion."""
        rng = self.rng
        n = self.n
        specs = []
        for e in range(events):
            th = rng.uniform(0, 2 * np.pi)
            # keep the hero notch area mostly clean so the new drip reads clearly
            if abs(np.angle(np.exp(1j * (th - self.g.notch_azimuth)))) < 0.35:
                th += 0.8
            width = rng.uniform(0.022, 0.07)
            vol = rng.uniform(0.04e-6, 0.30e-6)      # m^3 (0.04-0.3 ml)
            dur = rng.uniform(0.4, 1.4)
            t0 = rng.uniform(0.0, 1.5)
            temp = rng.uniform(66, 82)
            specs.append((th, width, vol, dur, t0, temp))
        t = 0.0
        c = 0.1
        T_end = 9.0
        print(f"[wax] history: {events} old spills", flush=True)
        while t < T_end:
            dt = self.stable_dt(c, 2e-3)
            inflow = np.zeros(n)
            for th, width, vol, dur, t0, temp in specs:
                if t0 <= t < t0 + dur:
                    dth = np.angle(np.exp(1j * (self.theta - th)))
                    prof = np.exp(-(dth / width) ** 2)
                    rate = vol / dur                                     # m^3/s
                    inflow += prof * rate / (prof.sum() * self.dxl)      # m^2/s per unit length
            c = self.film_step(dt, inflow, inflow_T=74.0, wall_top=40.0)
            t += dt
        # anything still liquid freezes in place
        dep = self.dep.numpy() + self.h.numpy()
        self.dep.assign(dep)
        self.h.zero_()
        self.E.zero_()
        print(f"[wax] history done: max deposit {dep.max()*1e3:.2f} mm", flush=True)

    # ------------------------------------------------------------------ #
    def pool_step(self, dt, t, lean_xy):
        """Advance the melt pool and return the inflow (m^2/s per unit rim length)."""
        g = self.g
        f = TL.frame_at(t)
        # melting of the rim: heated more on the side the flame leans to
        lx, ly = lean_xy
        lean_mag = math.hypot(lx, ly)
        heat_dir = np.cos(self.theta - math.atan2(ly, lx)) if lean_mag > 1e-5 else np.zeros(self.n)
        heat = 1.0 + np.clip(heat_dir, 0, None) * min(lean_mag / 0.002, 3.0)
        # directed breach: the hot side of the rim softens and slumps around BREACH_F
        ramp = TL._smooth((f - (TL.BREACH_F - 30)) / 40.0)
        slump = 0.0019 * ramp * self.notch_shape ** 1.5 * (0.75 + 0.25 * np.minimum(heat, 4.0) / 4.0)
        rim_now = self.rim0 - slump
        # erosion of the spout by the hot overflow
        rim_now -= self.erosion
        # pool level: slow rise from melting (stronger while the flame is pushed down)
        self.L += dt * 0.00004 * (1.0 + 0.5 * min(lean_mag / 0.002, 2.0))
        head = np.maximum(self.L - rim_now, 0.0)
        qw = 0.55 * (2.0 / 3.0) * math.sqrt(2 * G) * head ** 1.5 * 0.25     # viscous weir, m^2/s
        out_vol = (qw * self.R * self.dtheta).sum() * dt
        self.L -= out_vol / self.pool_area
        self.spilled += out_vol
        self.erosion += dt * 60.0 * qw
        self.erosion = np.minimum(self.erosion, 0.0015 * self.notch_shape + 0.0002)
        self.wet = np.maximum(self.wet * math.exp(-dt / 1.5), np.clip(qw / 2e-6, 0, 1))
        self.rim_now = rim_now
        return qw

    def snapshot(self):
        h = self.h.numpy()
        E = self.E.numpy()
        T = np.where(h > 1e-9, E / np.maximum(h, 1e-12), T_AMB)
        return dict(dep=self.dep.numpy(), h=h, T=T, table=self.table.numpy())


def _lean_for_frame(flame_dir, frame, cache={}):
    if flame_dir is None:
        return (0.0, 0.0)
    p = os.path.join(flame_dir, f"stats_{frame:04d}.json")
    if not os.path.exists(p):
        return (0.0, 0.0)
    if p not in cache:
        with open(p) as f:
            s = json.load(f)
        c = s.get("centroid")
        cache[p] = (c[0], c[1]) if c else (0.0, 0.0)
    return cache[p]


def run(out_dir, quality="preview", flame_dir=None, device=None, max_frames=None):
    _, preset, _ = C.get_preset(quality)
    geom = C.CandleGeometry()
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "DONE")):
        print("[wax] already done", flush=True)
        return
    wp.init()
    sim = WaxSim(preset, geom, device=device)
    print(f"[wax] device={sim.device} film grid {sim.n}x{sim.nz}  dz={sim.dz*1e3:.3f} mm", flush=True)
    t0w = time.time()
    hist_path = os.path.join(out_dir, "history.npz")
    if os.path.exists(hist_path):
        hz = np.load(hist_path)
        sim.dep.assign(hz["dep"])
        sim.table.assign(hz["table"])
    else:
        sim.history(preset.history_events)
        np.savez_compressed(hist_path, dep=sim.dep.numpy(), table=sim.table.numpy())
    sim.rim0 = sim.rim.copy()
    sim.erosion = np.zeros(sim.n)
    sim.rim_now = sim.rim.copy()

    times = TL.frame_times()
    n_frames = len(times) if max_frames is None else min(max_frames, len(times))
    t = TL.WARMUP - 0.05
    c = 0.05
    log = []
    for f in range(n_frames):
        target = times[f]
        while t < target - 1e-9:
            dt = min(sim.stable_dt(c, 2e-3), target - t)
            lean = _lean_for_frame(flame_dir, max(0, int(TL.frame_at(t))))
            qw = sim.pool_step(dt, t, lean)
            c = sim.film_step(dt, qw)
            t += dt
        snap = sim.snapshot()
        # leading edge of the newest drip, for the camera to follow
        hot = (snap["h"] > 0.12e-3) & (snap["T"] > T_SOL)
        front = None
        if hot.any():
            ii, kk = np.nonzero(hot)
            j = np.argmin(kk)
            front = [float(sim.theta[ii[j]]), float((kk[j] + 0.5) * sim.dz)]
        info = dict(frame=f, t=t, L=sim.L, spilled_ml=sim.spilled * 1e6, front=front,
                    max_h_mm=float(snap["h"].max() * 1e3))
        log.append(info)
        np.savez_compressed(os.path.join(out_dir, f"wax_{f:04d}.npz"),
                            dep=snap["dep"].astype(np.float16), h=snap["h"].astype(np.float16),
                            T=snap["T"].astype(np.float16), rim=sim.rim_now.astype(np.float32),
                            wet=sim.wet.astype(np.float16), L=np.array([sim.L]),
                            table=snap["table"].astype(np.float32),
                            dz=np.array([sim.dz]), z_top=np.array([sim.z_top]))
        if f % 48 == 0 or f == n_frames - 1:
            print(f"[wax] frame {f}/{n_frames} t={t:.3f}s L={sim.L*1e3:.2f}mm spilled={sim.spilled*1e6:.3f} ml "
                  f"front={front} wall={(time.time()-t0w):.0f}s", flush=True)
    with open(os.path.join(out_dir, "wax_log.json"), "w") as fh:
        json.dump(log, fh)
    with open(os.path.join(out_dir, "DONE"), "w") as fh:
        fh.write("ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", default="preview")
    ap.add_argument("--flame", default=None, help="flame output dir (for flame lean statistics)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.out, a.quality, a.flame, a.device, a.max_frames)


if __name__ == "__main__":
    main()
