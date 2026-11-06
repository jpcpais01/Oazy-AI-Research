# Oazy mark

The logo is a glass drop, rendered with real refraction.
Each shape is turned into a glass height field; for every pixel a view ray is
refracted through the glass once per wavelength (each with its own index of
refraction), then looks up a dark studio with hard-edged light panels. The colour
fringes are dispersion, not painted gradients.

Regenerate (needs Node + Playwright, Python with numpy, scipy, Pillow):

    node oazy/brand/masks.js          # shape masks -> oazy/brand/masks/
    cd /tmp && python3 /path/to/oazy/brand/export.py   # logo-512/256, drop-256, favicon-64, apple-touch-icon

Where the files go: `oazy/site/logo.png`, `favicon.png`, `apple-touch-icon.png`;
`oazy/web/icon.svg` and `oazy/web/favicon.svg` wrap the PNG
so the dashboard and chat apps can keep serving SVG.
