"""Quick-look renderer for exported flame fields (no Blender needed).

Integrates emission (soot blackbody + CH* blue) and smoke along the view axis,
applies a simple filmic curve and writes PNGs / a contact sheet.  Useful for
tuning the flame model and for sanity checks on the pod:

    python -m candle.peek work/flame --frames 0 100 200 290 320 --out sheet.png
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from . import spectral
from .look import LOOK


def load_frame(d, f):
    z = np.load(os.path.join(d, f"fields_{f:04d}.npz"))
    shape = tuple(z["shape"])
    lo = z["lo"]

    def full(name, fill=0.0):
        a = np.full(shape, fill, np.float32)
        c = z[name].astype(np.float32)
        a[lo[0]:lo[0] + c.shape[0], lo[1]:lo[1] + c.shape[1], lo[2]:lo[2] + c.shape[2]] = c
        return a
    return dict(T=full("T", 293.0), soot=full("soot"), smoke=full("smoke"), react=full("react"),
                dx=float(z["dx"][0]))


def emission_rgb(fr, soot_gain=1.0, blue_gain=1.0):
    temps, lut = spectral.soot_lut()
    ch = spectral.chemiluminescence_rgb()
    T = fr["T"]
    e = np.empty(T.shape + (3,), np.float32)
    for c in range(3):
        e[..., c] = fr["soot"] * np.interp(T, temps, lut[:, c]) * soot_gain + fr["react"] * ch[c] * blue_gain
    return e


def view(fr, axis=1, soot_gain=None, blue_gain=None, exposure=1.0, smoke_light=0.08):
    """Front view (looking along +y), simple emission-absorption integration."""
    e = emission_rgb(fr, LOOK["soot_gain"] if soot_gain is None else soot_gain,
                     LOOK["blue_gain"] if blue_gain is None else blue_gain)
    dx = fr["dx"]
    img = e.sum(axis=axis) * dx                          # radiance seen by a camera
    sm = fr["smoke"].sum(axis=axis) * dx * 1000.0
    img = img + smoke_light * sm[..., None] * np.array([0.9, 0.92, 1.0])
    img = np.transpose(img, (1, 0, 2))[::-1]            # z up
    x = img * exposure
    out = x / (1.0 + x)                                  # Reinhard
    return (np.clip(out, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--out", default="peek.png")
    ap.add_argument("--exposure", type=float, default=0.05)
    ap.add_argument("--blue", type=float, default=1.0)
    ap.add_argument("--crop", type=float, default=0.55, help="fraction of domain height shown")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.dir, "fields_*.npz")))
    if not a.frames:
        avail = [int(os.path.basename(f)[7:11]) for f in files]
        a.frames = avail[:: max(1, len(avail) // 8)]
    tiles = []
    for f in a.frames:
        fr = load_frame(a.dir, f)
        im = view(fr, exposure=a.exposure, blue_gain=LOOK["blue_gain"] * a.blue)
        h = im.shape[0]
        im = im[int(h * (1 - a.crop)):]
        tiles.append(im)
    sheet = np.concatenate([np.pad(t, ((0, 0), (0, 2), (0, 0))) for t in tiles], axis=1)
    import imageio.v3 as iio
    scale = max(1, 600 // sheet.shape[0])
    sheet = sheet.repeat(scale, 0).repeat(scale, 1)
    iio.imwrite(a.out, sheet)
    print("wrote", a.out, sheet.shape)


if __name__ == "__main__":
    main()
