// Render fill-only mask SVGs (white on black) for the refraction renderer.
const { chromium } = require('playwright');
const fs = require('fs');
const S = 1600;
const loop = 'M60 60 C48 41 20 39 20 60 C20 81 48 79 60 60 C72 41 100 39 100 60 C100 81 72 79 60 60';
const masks = {
  'liquid-o': `<path fill-rule="evenodd" fill="#fff" d="M60 18 A42 42 0 1 1 59.99 18 Z M62 34 A21.5 21.5 0 1 0 62.01 34 Z"/>`,
  'drop': `<path fill="#fff" d="M60 10 C66 22 84 42 84 60 A24 24 0 0 1 36 60 C36 42 54 22 60 10 Z"/>`,
  'ripple': `<g fill="none" stroke="#fff"><ellipse cx="60" cy="100" rx="44" ry="8.5" stroke-width="1.8"/><ellipse cx="60" cy="100" rx="29" ry="5.4" stroke-width="2.2"/><ellipse cx="60" cy="100" rx="14" ry="2.6" stroke-width="2.4"/></g>`,
  'loop': `<g transform="translate(60 60) scale(1.12) translate(-60 -60)"><path d="${loop}" fill="none" stroke="#fff" stroke-width="10" stroke-linecap="round"/></g>`,
  'bead': `<circle cx="${60 + 40 * 1.12}" cy="60" r="5.2" fill="#fff"/>`,
  'lensL': `<circle cx="45" cy="60" r="30" fill="#fff"/>`,
  'lensR': `<circle cx="75" cy="60" r="30" fill="#fff"/>`,
  'sphere': `<circle cx="60" cy="60" r="40" fill="#fff"/>`,
  'liquid': `<path fill="#fff" d="M10 64 C28 58.5 44 69.5 60 64 C76 58.5 92 69.5 110 64 L110 120 L10 120 Z"/>`,
  'pebble': `<rect x="17" y="17" width="86" height="86" rx="30" fill="#fff"/>`,
};
(async () => {
  fs.mkdirSync(__dirname + '/masks', { recursive: true });
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: S, height: S } });
  for (const [k, body] of Object.entries(masks)) {
    await p.setContent(`<html><body style="margin:0;background:#000"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="${S}" height="${S}">${body}</svg></body></html>`);
    await p.screenshot({ path: `${__dirname}/masks/${k}.png` });
  }
  await b.close();
})();
