"""Export the Oazy mark (the glass drop, no ripples) at every size the apps use.
Run from any folder after `node oazy/brand/masks.js`; files land in the current folder."""
import numpy as np, sys, os
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R

drop = R.load('drop')
img = R.render(R.bevel(drop, 12, pillow=0.35), drop)
TIGHT = (21, 8, 99, 86)      # drop fills ~95% of the square: logo and favicon
PADDED = (14, 0, 106, 92)    # breathing room for the home-screen icon


def save(box, size, path, black=False):
    x0, y0, x1, y1 = [int(round(v * R.PX)) for v in box]
    c = img[y0:y1, x0:x1]
    if black:
        im = Image.fromarray((c * 255).astype(np.uint8))
    else:  # glass is additive light: alpha = brightest channel, colour un-premultiplied
        a = c.max(-1, keepdims=True)
        rgb = np.where(a > 1e-4, c / np.maximum(a, 1e-4), 0)
        im = Image.fromarray((np.concatenate([rgb, a], -1) * 255).round().astype(np.uint8), 'RGBA')
    im.resize((size, size), Image.LANCZOS).save(path)


save(TIGHT, 512, 'logo-512.png')
save(TIGHT, 256, 'logo-256.png')
save(TIGHT, 64, 'favicon-64.png')
save(PADDED, 180, 'apple-touch-icon.png', black=True)
