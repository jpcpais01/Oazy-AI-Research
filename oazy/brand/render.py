"""Physically-inspired glass renderer for the Oazy logo shapes.

Each shape becomes a height field (glass thickness). Per pixel we take the surface normal,
refract a straight-down view ray through the glass separately for many wavelengths
(each with its own index of refraction, Cauchy dispersion), and look up a dark studio
environment with a few hard-edged light panels. Because every wavelength bends by a
different amount, light-panel edges seen through curved glass split into crisp spectral
fringes: real dispersion, not painted gradients. Fresnel reflection adds white highlights.
"""
import sys, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1600
PX = S / 120.0  # px per unit


def load(k):
    a = np.asarray(Image.open(f"{HERE}/masks/{k}.png").convert("L"), dtype=np.float32) / 255.0
    return a


def dist(mask):
    return ndi.distance_transform_edt(mask > 0.5) / PX  # in units


def bevel(mask, w, pillow=0.0, sig=3.0):
    d = ndi.gaussian_filter(dist(mask), 2.5) * (mask > 0.5)
    t = np.clip(d / w, 0, 1)
    h = w * np.sqrt(1 - (1 - t) ** 2)
    if pillow:
        h = h + pillow * w * ndi.gaussian_filter((mask > 0.5).astype(np.float32), 1.4 * w * PX)
    return ndi.gaussian_filter(h, sig) * (mask > 0.02)


def smooth_step(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def box(x, y, x0, x1, y0, y1, e=0.012):
    return smooth_step(x, x0 - e, x0 + e) * (1 - smooth_step(x, x1 - e, x1 + e)) * \
        smooth_step(y, y0 - e, y0 + e) * (1 - smooth_step(y, y1 - e, y1 + e))


def arc(r, th, r0, r1, t0, t1, e=0.012, et=0.45):
    a = smooth_step(r, r0 - e, r0 + e) * (1 - smooth_step(r, r1 - e, r1 + e))
    if t0 is None:
        return a
    d = np.angle(np.exp(1j * (th - (t0 + t1) / 2)))
    half = (t1 - t0) / 2
    return a * (1 - smooth_step(np.abs(d), half - et, half + et))


def env(x, y):
    """Dark studio with hard-edged curved light panels. (x,y) = lateral direction."""
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    v = 0.006 + 0.05 * np.clip(r, 0, 1) ** 2
    v = v + 1.15 * arc(r, th, 0.20, 0.27, None, None)                       # ring light
    v = v + 1.9 * arc(r, th, 0.46, 0.86, np.radians(180), np.radians(280))  # key, upper left
    v = v + 1.1 * arc(r, th, 0.52, 0.92, np.radians(0), np.radians(90))    # rim, lower right
    v = v + 0.7 * arc(r, th, 0.36, 0.42, np.radians(-120), np.radians(-60)) # top kicker
    return v


def wl_rgb(l):
    """Approximate colour of a wavelength (nm) in linear RGB (Bruton-style)."""
    r = g = b = 0.0
    if 380 <= l < 440: r, b = -(l - 440) / 60, 1
    elif l < 490: g, b = (l - 440) / 50, 1
    elif l < 510: g, b = 1, -(l - 510) / 20
    elif l < 580: r, g = (l - 510) / 70, 1
    elif l < 645: r, g = 1, -(l - 645) / 65
    elif l <= 700: r = 1
    return np.array([r, g, b]) ** 2.2


def render(h, alpha, bend=1.6, A=1.43, B=0.065, tint=(1, 1, 1)):
    gy, gx = np.gradient(h, 1 / PX)
    nx, ny, nz = -bend * gx, -bend * gy, np.ones_like(h)
    inv = 1 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    cosi = nz
    lams = np.linspace(405, 695, 14)
    W = np.stack([wl_rgb(l) for l in lams])
    W = W / W.sum(0, keepdims=True)
    col = np.zeros(h.shape + (3,), np.float32)
    for l, w in zip(lams, W):
        n = A + B / (l / 1000) ** 2
        eta = 1 / n
        k = 1 - eta * eta * (1 - cosi * cosi)
        k = np.maximum(k, 0)
        f = eta * cosi - np.sqrt(k)
        tx, ty = f * nx, f * ny            # incident (0,0,-1) contributes nothing laterally
        ox, oy = n * tx, n * ty            # exit through the flat back face (Snell)
        m = np.sqrt(ox * ox + oy * oy)
        s = np.where(m > 0.995, 0.995 / np.maximum(m, 1e-6), 1)
        e = env(ox * s, oy * s)
        col += e[..., None] * w[None, None, :]
    F = 0.04 + 0.96 * (1 - np.clip(cosi, 0, 1)) ** 5
    rx, ry = 2 * nz * nx, 2 * nz * ny
    refl = env(-rx, -ry)
    col = col * (1 - F[..., None]) * np.array(tint)[None, None, :] + (F * refl * 1.25)[..., None]
    col = 1 - np.exp(-1.5 * col)                       # soft filmic shoulder
    col = col * alpha[..., None]
    srgb = np.where(col <= 0.0031308, 12.92 * col, 1.055 * np.power(np.clip(col, 0, 1), 1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1)


def save(img, name, out):
    im = Image.fromarray((img * 255).astype(np.uint8))
    im = im.resize((S // 2, S // 2), Image.LANCZOS)
    im.save(f"{out}/{name}.png")


def main(out):
    os.makedirs(out, exist_ok=True)
    m = {k[:-4]: load(k[:-4]) for k in os.listdir(f"{HERE}/masks")}

    h = bevel(m['liquid-o'], 8.5, pillow=0.25)
    save(render(h, m['liquid-o']), '18-liquid-o', out)

    hd = bevel(m['drop'], 12, pillow=0.35)
    hr = bevel(m['ripple'], 1.1, sig=1.5)
    a = np.maximum(m['drop'], m['ripple'])
    save(render(np.maximum(hd, hr), a), '19-oasis-drop', out)

    hl = bevel(m['loop'], 5.0)
    save(render(hl, m['loop']), '20-glass-loop', out)

    h = bevel(m['lensL'], 13, pillow=0.4) + bevel(m['lensR'], 13, pillow=0.4)
    a = np.maximum(m['lensL'], m['lensR'])
    save(render(h, a, bend=1.3), '21-twin-lens', out)

    hs = bevel(m['sphere'], 40, sig=1.0)
    liq = ndi.gaussian_filter(m['liquid'] * m['sphere'], 0.9 * PX) * 2.6
    save(render(hs + liq, m['sphere'], bend=0.75), '22-horizon', out)

    h = bevel(m['pebble'], 13, pillow=0.12)
    save(render(h, m['pebble']), '23-pebble', out)


if __name__ == '__main__':
    main(sys.argv[1])
