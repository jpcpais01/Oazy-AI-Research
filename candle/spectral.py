"""Spectral emission of a candle flame, integrated to linear Rec.709 RGB.

Two emitters make the colour of a real candle flame:

* Soot (incandescent carbon particles, the yellow/orange body).  Small soot
  particles absorb, and hence emit, with an emissivity roughly proportional to
  1/lambda (Rayleigh regime), times the Planck spectrum B(lambda, T).
* Chemiluminescence of excited CH* (431 nm) and C2* Swan bands (473, 516,
  563 nm) in the thin reaction zone, the blue skirt at the base.

We integrate both against the CIE 1931 2-degree colour matching functions
(Wyman/Sloan/Shirley 2013 multi-lobe fit) and convert to linear Rec.709, the
scene-linear space Blender/Cycles works in.
"""
from __future__ import annotations

import numpy as np

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23

XYZ_TO_REC709 = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_xyz(lam_nm):
    x = (1.056 * _g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * _g(lam_nm, 442.0, 16.0, 26.7)
         - 0.065 * _g(lam_nm, 501.1, 20.4, 26.2))
    y = 0.821 * _g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _g(lam_nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


LAM = np.linspace(380.0, 780.0, 401)
CMF = cie_xyz(LAM)


def planck(lam_nm, T):
    lam = lam_nm * 1e-9
    a = 2.0 * H * C * C / lam ** 5
    with np.errstate(over="ignore"):
        return a / np.expm1(H * C / (lam * KB * T))


def spectrum_to_rgb(spec):
    xyz = (spec[:, None] * CMF).sum(0) * (LAM[1] - LAM[0])
    return XYZ_TO_REC709 @ xyz


def soot_lut(t_min=300.0, t_max=3000.0, n=1024, t_ref=1800.0):
    """RGB radiance of soot vs temperature, normalised so luminance(t_ref) = 1.

    Returns (temps, rgb[n,3]).  Negative RGB (out of Rec.709 gamut) is clipped.
    """
    temps = np.linspace(t_min, t_max, n)
    rgb = np.zeros((n, 3))
    for i, T in enumerate(temps):
        spec = planck(LAM, T) * (550.0 / LAM)   # emissivity ~ 1/lambda
        rgb[i] = spectrum_to_rgb(spec)
    ref_spec = planck(LAM, t_ref) * (550.0 / LAM)
    ref = spectrum_to_rgb(ref_spec)
    lum_ref = 0.2126 * ref[0] + 0.7152 * ref[1] + 0.0722 * ref[2]
    rgb = np.clip(rgb / lum_ref, 0.0, None)
    return temps, rgb.astype(np.float32)


def chemiluminescence_rgb():
    """Colour of the blue reaction zone, luminance normalised to 1."""
    lines = [(431.4, 1.0, 2.0), (473.7, 0.35, 3.0), (516.5, 0.55, 3.0), (563.5, 0.15, 3.0),
             (390.0, 0.25, 4.0)]
    spec = np.zeros_like(LAM)
    for mu, w, s in lines:
        spec += w * np.exp(-0.5 * ((LAM - mu) / s) ** 2)
    rgb = np.clip(spectrum_to_rgb(spec), 0.0, None)
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (rgb / lum).astype(np.float32)


def blackbody_rgb(T, norm=True):
    """Plain blackbody colour (for the wick ember shader), luminance normalised."""
    rgb = np.clip(spectrum_to_rgb(planck(LAM, T)), 0, None)
    if norm:
        rgb = rgb / max(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 1e-30)
    return rgb


def blackbody_luminance(T):
    rgb = spectrum_to_rgb(planck(LAM, T))
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


if __name__ == "__main__":
    t, lut = soot_lut()
    for T in (1000, 1200, 1400, 1600, 1800, 2000):
        i = np.searchsorted(t, T)
        print(T, np.round(lut[i], 4))
    print("CH*", chemiluminescence_rgb())
