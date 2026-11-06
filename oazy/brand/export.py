import numpy as np, sys
from PIL import Image
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R
m = {k: R.load(k) for k in ['drop', 'ripple']}
hd = R.bevel(m['drop'], 12, pillow=0.35); hr = R.bevel(m['ripple'], 1.1, sig=1.5)
full = R.render(np.maximum(hd, hr), np.maximum(m['drop'], m['ripple']))
dropo = R.render(hd, m['drop'])
def rgba(img, box, size, path, black=False):
    px = R.PX; x0, y0, x1, y1 = [int(round(v * px)) for v in box]
    c = img[y0:y1, x0:x1]
    if black:
        im = Image.fromarray((c * 255).astype(np.uint8))
    else:
        a = c.max(-1, keepdims=True); rgb = np.where(a > 1e-4, c / np.maximum(a, 1e-4), 0)
        im = Image.fromarray((np.concatenate([rgb, a], -1) * 255).round().astype(np.uint8), 'RGBA')
    im.resize((size, size), Image.LANCZOS).save(path)
rgba(full, (8, 4, 112, 108), 512, 'logo-512.png')
rgba(full, (8, 4, 112, 108), 256, 'logo-256.png')
rgba(dropo, (18, 5, 102, 89), 256, 'drop-256.png')
rgba(dropo, (18, 5, 102, 89), 64, 'favicon-64.png')
rgba(dropo, (12, -1 + 1, 108, 96), 180, 'apple-touch-icon.png', black=True)
