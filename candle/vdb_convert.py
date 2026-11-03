"""Stage 3: raw flame fields -> render-ready OpenVDB volumes + light proxies.

For every frame this computes, from temperature / soot / reaction rate / smoke:
  emission (vec3)  soot incandescence (Planck x 1/lambda, see spectral.py) plus
                   CH*/C2* chemiluminescence, radiance per metre, linear Rec.709
  soot (float)     absorbing soot density
  smoke (float)    white scattering paraffin smoke density
and a small JSON with three point-light proxies (positions and RGB power) that
reproduce the flame's illumination: Cycles cannot importance-sample emissive
volumes, so lighting the wax directly from the volume would be hopelessly
noisy.  A point light of power P has intensity P/(4 pi), and a volume with
emission e per metre has intensity integral(e dV) (both verified in Cycles 5.2),
so P = 4 pi integral(e dV).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import spectral
from .look import LOOK
from .vdb_writer import write_vdb

_TEMPS, _LUT = spectral.soot_lut()
_CH = spectral.chemiluminescence_rgb()


def convert_frame(args):
    src, dst_vdb, dst_json, look = args
    z = np.load(src)
    T = z["T"].astype(np.float32)
    soot = z["soot"].astype(np.float32)
    smoke = z["smoke"].astype(np.float32)
    react = z["react"].astype(np.float32)
    lo = z["lo"]
    dx = float(z["dx"][0])
    origin = z["origin"]
    em = np.empty(T.shape + (3,), np.float32)
    for c in range(3):
        em[..., c] = (soot * np.interp(T, _TEMPS, _LUT[:, c]) * look["soot_gain"]
                      + np.maximum(react, 0) * _CH[c] * look["blue_gain"])
    em[em < 0.05] = 0.0
    # Exposure guard: a transient (breath gust, coarse-grid soot burst) must not
    # flood the shot. Cap the frame's total emitted power at a few times the
    # steady flame, scaling the volume and its light proxies together.
    cap = look.get("max_power_ratio", 2.0) * (look.get("flame_power") or 0.0)
    if cap > 0:
        pw = float((em @ np.array([0.2126, 0.7152, 0.0722], np.float32)).sum()) * dx ** 3 * 4 * math.pi
        if pw > cap:
            em *= cap / pw
    trans = origin + (lo + 0.5) * dx
    write_vdb(dst_vdb, {"emission": em, "soot": soot, "smoke": smoke}, dx, trans, threshold=1e-4)

    # ---- light proxies --------------------------------------------------------
    dv = dx ** 3
    lum = em @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    total = float(lum.sum())
    lights = []
    if total > 1e-12:
        nx, ny, nz = lum.shape
        xs = trans[0] + np.arange(nx) * dx
        ys = trans[1] + np.arange(ny) * dx
        zs = trans[2] + np.arange(nz) * dx
        zprof = lum.sum(axis=(0, 1))
        cdf = np.cumsum(zprof) / zprof.sum()
        edges = [0, int(np.searchsorted(cdf, 1 / 3)) + 1, int(np.searchsorted(cdf, 2 / 3)) + 1, nz]
        for b0, b1 in zip(edges[:-1], edges[1:]):
            if b1 <= b0:
                continue
            L = lum[:, :, b0:b1]
            tl = float(L.sum())
            if tl <= 0:
                continue
            px = float((L.sum(axis=(1, 2)) * xs).sum() / tl)
            py = float((L.sum(axis=(0, 2)) * ys).sum() / tl)
            pz = float((L.sum(axis=(0, 1)) * zs[b0:b1]).sum() / tl)
            rgb = em[:, :, b0:b1].reshape(-1, 3).sum(0) * dv * 4 * math.pi
            # spatial spread (for the light radius)
            sz = float(np.sqrt(max((L.sum(axis=(0, 1)) * (zs[b0:b1] - pz) ** 2).sum() / tl, 0)))
            lights.append(dict(pos=[px, py, pz], rgb=[float(v) for v in rgb], spread=sz))
    with open(dst_json, "w") as f:
        json.dump(dict(lights=lights, total_lum_power=total * dv * 4 * math.pi), f)
    return os.path.basename(dst_vdb)


def normalised_look(flame_dir, look):
    """Scale the flame gains so the steady flame emits LOOK['flame_power'] watts.

    Soot loading (hence brightness) depends somewhat on grid resolution; this
    keeps exposure identical between the preview and the 4K run.
    """
    look = dict(look)
    lums = []
    for f in range(0, 70):
        p = os.path.join(flame_dir, f"stats_{f:04d}.json")
        if os.path.exists(p):
            with open(p) as fh:
                v = json.load(fh).get("lum_total", 0.0)
            if v > 0:
                lums.append(v)
    if lums and look.get("flame_power"):
        ref = float(np.median(lums))
        gain = look["flame_power"] / (4 * math.pi * ref)
        ratio = gain / look["soot_gain"]
        # a few-x correction is expected between grids; anything beyond that
        # means the soot statistic is unreliable, so do not trust it blindly
        ratio = min(max(ratio, 0.33), 3.0)
        gain = look["soot_gain"] * ratio
        look["soot_gain"] = gain   # blue CH* light is a separate emitter: left as is
        print(f"[vdb] exposure normalisation: soot_gain={gain:.0f} (x{ratio:.2f})", flush=True)
    return look


def run(flame_dir, out_dir, workers=None, look=None):
    look = normalised_look(flame_dir, look or LOOK)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "look_used.json"), "w") as fh:
        json.dump(look, fh, indent=1)
    srcs = sorted(glob.glob(os.path.join(flame_dir, "fields_*.npz")))
    jobs = []
    for s in srcs:
        f = int(os.path.basename(s)[7:11])
        dv = os.path.join(out_dir, f"flame_{f:04d}.vdb")
        dj = os.path.join(out_dir, f"lights_{f:04d}.json")
        if os.path.exists(dv) and os.path.exists(dj) and os.path.getmtime(dv) >= os.path.getmtime(s):
            continue
        jobs.append((s, dv, dj, look))
    print(f"[vdb] {len(jobs)} frames to convert ({len(srcs) - len(jobs)} up to date)", flush=True)
    if not jobs:
        return
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1, 16))
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for name in ex.map(convert_frame, jobs, chunksize=1):
            done += 1
            if done % 32 == 0 or done == len(jobs):
                print(f"[vdb] {done}/{len(jobs)}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.flame, a.out, a.workers)


if __name__ == "__main__":
    main()
