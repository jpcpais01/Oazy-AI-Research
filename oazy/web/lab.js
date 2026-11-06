// The training lab's look, the landing page's (oazy/site): the hero's light streaks seen through liquid glass behind
// the page head, a faint photon field under the rest of the page with every panel made of clear glass that bends
// them (so a panel's edge is drawn by refracted light, not by a border), the launcher-style bar with sliding tabs,
// and the slim white liquid-glass scrollbar. Decoration only: without WebGL a plain CSS look stays, and nothing here
// touches the data.
"use strict";

const LAB = (() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------------------ the bar: solid once scrolled, sliding pill */
  function bar() {
    const el = document.getElementById("bar");
    const onScroll = () => el.classList.toggle("solid", scrollY > 12);
    addEventListener("scroll", onScroll, { passive: true }); onScroll();

    const tabs = document.getElementById("tabs"), pill = document.getElementById("tabpill");
    if (!tabs) return;
    const links = [...tabs.querySelectorAll("a[data-sec]")];
    const move = (a) => {
      if (!a) { pill.style.opacity = 0; return; }
      pill.style.opacity = 1;
      pill.style.width = a.offsetWidth + "px";
      pill.style.transform = `translateX(${a.offsetLeft}px)`;
      links.forEach((l) => l.classList.toggle("on", l === a));
    };
    let cur = null;
    const spy = () => {
      let best = null;
      for (const a of links) {
        const s = document.getElementById(a.dataset.sec);
        if (!s || !s.offsetParent) continue;                   // a hidden section (e.g. no Memtrix run)
        if (s.getBoundingClientRect().top < innerHeight * 0.35) best = a;
      }
      if (best !== cur) { cur = best; move(best); }
    };
    addEventListener("scroll", spy, { passive: true });
    addEventListener("resize", () => { cur = null; spy(); });
    setInterval(spy, 1000);                                     // sections appear as the data arrives
    spy();
  }

  /* ------------------------------------------------------------------ the scrollbar: a sliver of white liquid glass */
  function rail() {
    if (matchMedia("(pointer: coarse)").matches) return;         // phones keep their own
    const rail = document.createElement("div"), thumb = document.createElement("div");
    rail.className = "rail"; rail.setAttribute("aria-hidden", "true"); thumb.className = "thumb";
    rail.appendChild(thumb); document.body.appendChild(rail);
    document.documentElement.classList.add("has-rail");
    let hideT = null, dragging = false, dragY = 0, dragTop = 0;
    const sizes = () => {
      const view = innerHeight, full = document.documentElement.scrollHeight, track = rail.clientHeight;
      return { track, th: Math.max(36, track * view / full), max: full - view };
    };
    const place = () => {
      const z = sizes();
      thumb.style.height = z.th + "px";
      thumb.style.transform = `translateY(${z.max > 0 ? (scrollY / z.max) * (z.track - z.th) : 0}px)`;
      rail.style.display = z.max > 4 ? "" : "none";
    };
    const wake = () => {
      rail.classList.add("on"); clearTimeout(hideT);
      hideT = setTimeout(() => { if (!dragging) rail.classList.remove("on"); }, 1100);
    };
    addEventListener("scroll", () => { place(); wake(); }, { passive: true });
    addEventListener("resize", place);
    new ResizeObserver(place).observe(document.body);           // the page grows as the data arrives
    thumb.addEventListener("pointerdown", (e) => {
      dragging = true; dragY = e.clientY; dragTop = scrollY;
      rail.classList.add("drag"); thumb.setPointerCapture(e.pointerId); e.preventDefault(); e.stopPropagation();
    });
    thumb.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const z = sizes();
      scrollTo({ top: dragTop + (e.clientY - dragY) * z.max / Math.max(1, z.track - z.th), behavior: "instant" });
    });
    const end = () => { dragging = false; rail.classList.remove("drag"); wake(); };
    thumb.addEventListener("pointerup", end); thumb.addEventListener("pointercancel", end);
    rail.addEventListener("pointerdown", (e) => {
      if (e.target !== rail) return;
      const z = sizes(), r = rail.getBoundingClientRect();
      const f = (e.clientY - r.top - z.th / 2) / Math.max(1, z.track - z.th);
      scrollTo({ top: Math.max(0, Math.min(1, f)) * z.max, behavior: "smooth" });
    });
    place();
  }

  /* ------------------------------------------------------------------ the head: light streaks through liquid glass */
  let pace = 1;          // 1: models training (the light flows), ~0.15: nothing training (almost still)
  function band() {
    const canvas = document.getElementById("warp");
    if (!canvas) return;
    const opts = { antialias: false, alpha: false, depth: false, stencil: false, premultipliedAlpha: false, powerPreference: "low-power" };
    let gl = canvas.getContext("webgl2", opts);
    const v2 = !!gl;
    if (!gl) {
      gl = canvas.getContext("webgl", opts);
      if (gl && !gl.getExtension("OES_standard_derivatives")) gl = null;
    }
    if (!gl) { canvas.remove(); return; }                         // the CSS glow behind it stays

    let texType = gl.UNSIGNED_BYTE, texIFmt = gl.RGBA, hdr = false;
    if (v2 && gl.getExtension("EXT_color_buffer_float")) { hdr = true; texType = gl.HALF_FLOAT; texIFmt = gl.RGBA16F; }
    else if (!v2) {
      const hf = gl.getExtension("OES_texture_half_float");
      if (hf && gl.getExtension("EXT_color_buffer_half_float")) { hdr = true; texType = hf.HALF_FLOAT_OES; }
    }
    const SCALE = hdr ? 1.0 : 0.45;

    const HEAD_V = v2 ? "#version 300 es\n#define ATTR in\n#define VARY out\n" : "#define ATTR attribute\n#define VARY varying\n";
    const HEAD_F = v2 ? "#version 300 es\nprecision highp float;\n#define VARY in\n#define TEX texture\n#define OUT o\nout vec4 o;\n"
                      : "#extension GL_OES_standard_derivatives : enable\nprecision highp float;\n#define VARY varying\n#define TEX texture2D\n#define OUT gl_FragColor\n";
    const VERT = HEAD_V + "ATTR vec2 p; VARY vec2 vUv; void main(){ vUv = p * 0.5 + 0.5; gl_Position = vec4(p, 0., 1.); }";

    // 1. the streak field (the landing page's tunnel: wisps, comets, swells, drifting colour), its waist behind the title
    const FIELD = HEAD_F + `
VARY vec2 vUv;
uniform vec2  uRes;      // target size in pixels
uniform float uTime;
uniform vec2  uMouse;    // -1..1, eased
uniform float uCenter;   // waist height, 0..1 from the bottom
uniform float uPx;       // target pixels per CSS pixel
uniform float uScale;
uniform float uWaistX;  // waist across, 0..1 from the left

float h11(float n) { return fract(sin(n * 91.3458) * 47453.5453); }

// smooth 1D value noise along a lane, evolving slowly in time (2D lattice)
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = h11(i.x + i.y * 57.0), b = h11(i.x + 1.0 + i.y * 57.0);
  float c = h11(i.x + (i.y + 1.0) * 57.0), d = h11(i.x + 1.0 + (i.y + 1.0) * 57.0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

vec3 tint(float r) {
  // pale lilac and ice blue carry the scene; violet, magenta, cyan, ember and mint add life
  if (r < 0.30) return vec3(0.84, 0.82, 1.00);   // lilac white
  if (r < 0.50) return vec3(0.52, 0.62, 1.00);   // ice blue
  if (r < 0.66) return vec3(0.62, 0.46, 1.00);   // violet
  if (r < 0.76) return vec3(0.42, 0.86, 1.00);   // cyan
  if (r < 0.85) return vec3(1.00, 0.46, 0.80);   // magenta
  if (r < 0.92) return vec3(1.00, 0.64, 0.46);   // ember
  return vec3(0.52, 1.00, 0.76);                 // mint
}

// a smooth, looping gradient of the tunnel's colours
vec3 field(float h) {
  h = fract(h) * 6.0;
  vec3 c0 = vec3(0.84, 0.80, 1.00);   // lilac white
  vec3 c1 = vec3(0.60, 0.46, 1.00);   // violet
  vec3 c2 = vec3(0.44, 0.60, 1.00);   // electric blue
  vec3 c3 = vec3(0.44, 0.86, 1.00);   // cyan
  vec3 c4 = vec3(0.86, 0.70, 1.00);   // pale orchid
  vec3 c5 = vec3(1.00, 0.52, 0.78);   // magenta rose
  float t = smoothstep(0.0, 1.0, fract(h));
  if (h < 1.0) return mix(c0, c1, t);
  if (h < 2.0) return mix(c1, c2, t);
  if (h < 3.0) return mix(c2, c3, t);
  if (h < 4.0) return mix(c3, c4, t);
  if (h < 5.0) return mix(c4, c5, t);
  return mix(c5, c0, t);
}

vec3 layer(vec2 p, float aspect, float id, float N, float waist, float flare, float gain, float spd) {
  float depth = id / 3.0;
  float xn = p.x / (0.5 * aspect) - uMouse.x * 0.04 * (0.3 + depth);
  float ax = sqrt(xn * xn + 0.0016);                // a soft |x|: no crease where the two halves meet
  float y = p.y + uMouse.y * 0.02 * (0.3 + depth) + 0.010 * sin(xn * 1.7 + uTime * 0.11 + id * 1.7) * ax;
  float w = waist + flare * ax * ax;                 // half-height of the bundle here
  float s = y / w;                                   // which lane we are in
  float env = exp(-s * s * 1.35) * (0.55 + 0.45 * exp(-s * s * 5.0));
  if (env < 0.003) return vec3(0.0);

  float lane = floor(s * N);
  float r1 = h11(lane * 1.37 + id * 17.31);
  float r2 = h11(lane * 2.11 + id * 5.71 + 3.1);
  float r3 = h11(lane * 0.73 + id * 9.13 + 7.7);
  float r4 = h11(lane * 4.31 + id * 1.97 + 1.3);

  float lanePx = w * uRes.y / N;
  float sc = (lane + 0.5 + (r4 - 0.5) * 0.7) / N;
  float dpx = abs(s - sc) * w * uRes.y;
  // hair-thin: about 0.8 CSS px, a touch wider close to us
  float t = uPx * (0.45 + 0.5 * ax) * (0.7 + 0.6 * r3);
  float tt = max(t, 0.6);
  float prof = exp(-dpx * dpx / (tt * tt)) * min(1.0, t / 0.6);

  // perspective along the tunnel: log distance, so streaks are tiny far away
  // (in the middle) and long close up (at the edges), and speed up outwards
  float u = log(ax + 0.035);
  // each lane has its own pace, and that pace itself ebbs and flows smoothly
  float sp = spd * mix(0.5, 1.6, r1);
  float wf = mix(0.07, 0.19, r2);
  float travel = uTime * sp + 0.7 * sp / wf * sin(uTime * wf + r4 * 6.283);
  float r5 = h11(lane * 7.77 + id * 3.33 + 5.9);
  float lit = smoothstep(0.22, 0.42, r5);

  // 1. wisps: soft segments that drift outwards and slowly change shape
  float freq = mix(1.6, 4.2, r2);
  float evolve = uTime * mix(0.06, 0.16, r3);
  float n = vnoise(vec2(u * freq - travel + r3 * 40.0, evolve + r4 * 31.0));
  float n2 = vnoise(vec2(u * freq * 3.1 - travel * 1.3 + r2 * 23.0, evolve * 1.7 + 7.0));
  float wisp = smoothstep(0.55, 0.95, n) * smoothstep(0.15, 0.6, n2);
  wisp = wisp * wisp * (3.0 - 2.0 * wisp);

  // 2. comets: now and then a brighter head with a long tail glides out
  float cf = mix(0.7, 1.6, r4);
  float ph = u * cf - travel * 1.6 + r2 * 9.0;
  float f = fract(ph);
  float duty = mix(0.4, 0.85, r3);
  float aa = min(fwidth(ph) * 1.2 + 0.004, 0.25);
  float comet = pow(smoothstep(0.0, duty, f), 2.2) * (1.0 - smoothstep(duty - aa * 2.0, duty + aa, f));
  comet *= step(0.74, h11(lane * 3.9 + id * 11.1 + floor(ph) * 7.31));   // not every lap, not every lane
  float head = smoothstep(duty * 0.82, duty, f) * comet;

  // 3. swells: soft, local waves of extra light rolling through parts of the tunnel
  float swell = 0.8 + 0.55 * pow(0.5 + 0.5 * sin(u * 1.8 - uTime * 0.7 + s * 1.3 + id * 1.9), 5.0);

  float streak = (wisp * 0.9 + comet * 1.05) * lit * swell;
  float bright = 1.35 * (0.3 + 0.7 * r1 * r1) * (r2 > 0.955 ? 2.2 : 1.0);   // a few hero streaks

  float xg = mix(0.32, 1.0, smoothstep(0.0, 0.9, ax));                     // brighter close up
  // colour comes from slowly drifting regions, so neighbouring streaks share a hue
  float hue = vnoise(vec2(u * 0.75 + uTime * 0.03 + id * 3.1, s * 1.5 - uTime * 0.022 + id * 7.7));
  hue = fract(hue * 1.35 + 0.35 * vnoise(vec2(s * 0.6 + 19.0, uTime * 0.017 + u * 0.3)));
  vec3 col = mix(field(hue), tint(fract(r3 * 7.13 + r1)), 0.15);
  // colour blobs: soft, random patches of stronger colour that ride along with the flowing light
  float flow = uTime * spd * 0.9;
  vec2 bq = vec2(u * 1.1 - flow + id * 5.3, s * 1.9 + id * 2.1);
  float bm = vnoise(bq + vec2(0.0, uTime * 0.05)) * 0.65 + vnoise(bq * 2.3 + vec2(17.0, -uTime * 0.08)) * 0.35;
  bm = smoothstep(0.48, 0.80, bm);
  float bh = vnoise(vec2(u * 0.35 - flow * 0.5 + 41.0, s * 0.7 + id * 9.0 + uTime * 0.02));
  vec3 blobCol = bh < 0.25 ? mix(vec3(1.00, 0.42, 0.40), vec3(1.00, 0.36, 0.78), bh / 0.25)          // coral -> magenta
               : bh < 0.50 ? mix(vec3(1.00, 0.36, 0.78), vec3(0.42, 0.40, 1.00), (bh - 0.25) / 0.25)  // magenta -> blue violet
               : bh < 0.75 ? mix(vec3(0.42, 0.40, 1.00), vec3(0.25, 0.90, 1.00), (bh - 0.50) / 0.25)  // -> cyan
               :             mix(vec3(0.25, 0.90, 1.00), vec3(0.40, 1.00, 0.62), (bh - 0.75) / 0.25); // -> mint
  col = mix(col, blobCol, bm * 0.85);
  // light disperses as it comes closer: cooler far away, slightly warmer at the edges
  col *= mix(vec3(0.86, 0.82, 1.12), vec3(1.08, 0.97, 0.95), smoothstep(0.15, 1.0, ax));
  col = mix(col, vec3(1.0, 0.97, 1.0), head * 0.75);                       // white-hot heads
  vec3 line = col * prof * streak * bright;

  // where lanes pack tighter than a few pixels, show their average light (no shimmer)
  vec3 haze = vec3(0.70, 0.66, 1.00) * 0.045 * min(1.0, 1.6 * tt / max(lanePx, 0.001));
  float k = smoothstep(1.3, 3.6, lanePx);
  return mix(haze, line, k) * env * xg * gain;
}

void main() {
  vec2 frag = vUv * uRes;
  float aspect = uRes.x / uRes.y;
  vec2 p = vec2((frag.x - uWaistX * uRes.x) / uRes.y, frag.y / uRes.y - uCenter);
  float sq = clamp(aspect, 0.55, 1.0);
  p.y *= mix(1.7, 1.0, (sq - 0.55) / 0.45);        // keep the hourglass on tall phones

  vec3 c = vec3(0.0);
  c += layer(p, aspect, 0.0, 170.0, 0.150, 0.16, 0.85, 0.075);
  c += layer(p, aspect, 1.0, 120.0, 0.170, 0.22, 1.00, 0.10);
  c += layer(p, aspect, 2.0, 105.0, 0.195, 0.29, 1.15, 0.135);
  c += layer(p, aspect, 3.0,  58.0, 0.225, 0.38, 1.35, 0.18);
  // pools of light: big soft regions that glow brighter and others that fall dim,
  // drifting outwards with the flow (shared by all layers so they read as one light)
  float xn = p.x / (0.5 * aspect), ax = sqrt(xn * xn + 0.0016);
  // each side has its own pattern; they cross-fade softly through the middle so there is no seam
  vec2 rq = vec2(log(ax + 0.035) * 0.6 - uTime * 0.045, p.y / (0.19 + 0.28 * ax * ax) * 0.8);
  vec2 rr = rq + vec2(37.0, 0.0);
  float gl = vnoise(rq + vec2(0.0, uTime * 0.03)) * 0.6 + vnoise(rq * 2.1 + vec2(-uTime * 0.05, 11.0)) * 0.4;
  float gr = vnoise(rr + vec2(0.0, uTime * 0.03)) * 0.6 + vnoise(rr * 2.1 + vec2(-uTime * 0.05, 11.0)) * 0.4;
  float g = mix(gl, gr, smoothstep(-0.12, 0.12, xn));
  c *= 0.2 + 2.0 * smoothstep(0.3, 0.85, g);
  OUT = vec4(c * uScale, 1.0);
}`;
    // 2. bloom blur (separable, 9 taps folded into 5 fetches)
    const BLUR = HEAD_F + `
VARY vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uDir;       // one texel step in the blur direction
void main() {
  vec3 c = TEX(uTex, vUv).rgb * 0.2270270270;
  c += (TEX(uTex, vUv + uDir * 1.3846153846).rgb + TEX(uTex, vUv - uDir * 1.3846153846).rgb) * 0.3162162162;
  c += (TEX(uTex, vUv + uDir * 3.2307692308).rgb + TEX(uTex, vUv - uDir * 3.2307692308).rgb) * 0.0702702703;
  OUT = vec4(c, 1.0);
}`;
    // 3. final: seen through a sheet of liquid glass with slow crossing ripples (bending splits the colour), glow, grain
    const FINAL = HEAD_F + `
VARY vec2 vUv;
uniform sampler2D uScene, uB1, uB2;
uniform vec2  uRes;
uniform float uTime, uIntro, uFade, uCenter, uScale, uWaistX;
float h21(vec2 p) { p = fract(p * vec2(233.34, 851.73)); p += dot(p, p + 23.45); return fract(p.x * p.y); }
// one travelling ripple on the glass: height, slope (x, y) and curvature
vec4 rip(vec2 q, vec2 k, float w, float ph, float a) {
  float th = dot(k, q) + w * uTime + ph;
  float sn = sin(th), cs = cos(th);
  return vec4(a * sn, a * cs * k, -a * sn * dot(k, k));
}
void main() {
  vec2 uv = vUv;
  vec2 d = uv - vec2(uWaistX, uCenter);
  float aspect = uRes.x / uRes.y;

  // we look at the light through a sheet of liquid glass. Its surface is a few slow, crossing
  // ripples. Where the surface tilts, the light
  // behind it is bent; where it curves, the light is gathered or spread, so it brightens and
  // dims like caustics; and blue bends a little more than red, so the bending splits colour.
  vec2 q = d * vec2(aspect, 1.0);
  q += 0.12 * vec2(sin(q.y * 2.1 + uTime * 0.13), sin(q.x * 1.6 - uTime * 0.11));
  vec4 s = rip(q, vec2(3.1, 1.4), 0.21, 0.0, 0.0016)
         + rip(q, vec2(-2.2, 3.6), -0.17, 1.7, 0.0013)
         + rip(q, vec2(5.3, -2.6), 0.29, 4.1, 0.0006)
         + rip(q, vec2(-1.3, -6.4), 0.23, 2.3, 0.0004);
  vec2 off = s.yz * vec2(1.0 / aspect, 1.0);
  // lens aberration grows with how hard the light is bent, plus a gentle split at the edges
  vec2 ca = d * vec2(0.010, 0.022) * (0.55 + 1.6 * dot(d, d)) + off * 0.16;
  vec2 o1 = off * 1.5, o2 = off * 2.0;
  vec3 sc = vec3(TEX(uScene, uv + off * 0.9 + ca).r, TEX(uScene, uv + off).g, TEX(uScene, uv + off * 1.1 - ca).b);
  vec3 b1 = vec3(TEX(uB1, uv + o1 * 0.9 + ca * 1.5).r, TEX(uB1, uv + o1).g, TEX(uB1, uv + o1 * 1.1 - ca * 1.5).b);
  vec3 b2 = vec3(TEX(uB2, uv + o2 * 0.9 + ca * 2.2).r, TEX(uB2, uv + o2).g, TEX(uB2, uv + o2 * 1.1 - ca * 2.2).b);
  float caustic = clamp(1.0 - s.w * 10.0, 0.5, 2.0);
  vec3 c = (sc * caustic + b1 * 0.42 + b2 * 0.6) / uScale;

  // a faint violet breath behind the waist
  float ax = abs(d.x) * 2.0;
  c += vec3(0.20, 0.16, 0.55) * 0.05 * exp(-pow(d.y / (0.10 + 0.25 * ax * ax), 2.0));

  c *= uIntro * uFade * 0.85;                          // a calmer cut than the landing page's hero
  c = vec3(1.0) - exp(-c * 2.1);                       // soft shoulder, no hard clipping
  float vig = smoothstep(1.25, 0.25, length(d * vec2(1.0, 1.6)));
  c *= mix(0.55, 1.0, vig);
  c *= smoothstep(0.0, 0.42, uv.y);                    // melt into the page below
  c += vec3(0.0196, 0.0196, 0.0275);                   // the page black (#050507)
  float g = h21(gl_FragCoord.xy + fract(uTime * 7.0) * 113.0) - 0.5;
  c += g * 0.018 * (0.3 + c);                          // film grain
  c += (h21(gl_FragCoord.yx + 3.7) - 0.5) / 255.0;     // dither, kills banding
  OUT = vec4(c, 1.0);
}`;

    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : (console.warn(gl.getShaderInfoLog(s)), null); };
    const vs = sh(gl.VERTEX_SHADER, VERT);
    const program = (src, names) => {
      const fs = sh(gl.FRAGMENT_SHADER, src); if (!vs || !fs) return null;
      const pr = gl.createProgram(); gl.attachShader(pr, vs); gl.attachShader(pr, fs); gl.bindAttribLocation(pr, 0, "p"); gl.linkProgram(pr);
      if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) return null;
      const u = {}; names.forEach((n) => (u[n] = gl.getUniformLocation(pr, n))); return { pr, u };
    };
    const P1 = program(FIELD, ["uRes", "uTime", "uCenter", "uPx", "uScale", "uWaistX"]);
    const PB = program(BLUR, ["uTex", "uDir"]);
    const PF = program(FINAL, ["uScene", "uB1", "uB2", "uRes", "uTime", "uIntro", "uFade", "uCenter", "uScale", "uWaistX"]);
    if (!P1 || !PB || !PF) { canvas.remove(); return; }
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const target = (w, h) => {
      const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
      gl.texImage2D(gl.TEXTURE_2D, 0, texIFmt, w, h, 0, gl.RGBA, texType, null);
      const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      return { tex, fb, w, h };
    };
    const host = canvas.parentElement, title = host.querySelector("h1");
    let W = 0, H = 0, px = 1, quality = 1, T = {}, waistX = 0.5, center = 0.5;
    function resize() {
      const r = canvas.getBoundingClientRect();
      px = Math.min(devicePixelRatio || 1, 1.5) * quality;
      const nW = Math.max(8, Math.round(r.width * px)), nH = Math.max(8, Math.round(r.height * px));
      // the waist sits right behind the page title: towards its end on wide screens, centred on narrow ones
      const tr = title ? title.getBoundingClientRect() : null;
      waistX = r.width > 900 ? 0.6 : 0.5;
      center = tr && tr.height ? 1 - ((tr.top + tr.height * 0.5) - r.top) / r.height : 0.55;
      if (nW === W && nH === H) return;
      W = nW; H = nH; canvas.width = W; canvas.height = H;
      Object.values(T).forEach((t) => { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); });
      const q = (n, d) => Math.max(4, Math.round(n / d));
      T = { scene: target(W, H), a1: target(q(W, 4), q(H, 4)), b1: target(q(W, 4), q(H, 4)), a2: target(q(W, 8), q(H, 8)), b2: target(q(W, 8), q(H, 8)) };
    }
    const pass = (prog, out, w, h) => { gl.useProgram(prog.pr); gl.bindFramebuffer(gl.FRAMEBUFFER, out ? out.fb : null); gl.viewport(0, 0, w, h); };
    const bind = (unit, tex, loc) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(loc, unit); };
    const blur = (src, tmp, dst) => {
      pass(PB, tmp, tmp.w, tmp.h); bind(0, src.tex, PB.u.uTex); gl.uniform2f(PB.u.uDir, 1 / src.w, 0); gl.drawArrays(gl.TRIANGLES, 0, 3);
      pass(PB, dst, dst.w, dst.h); bind(0, tmp.tex, PB.u.uTex); gl.uniform2f(PB.u.uDir, 0, 1 / tmp.h); gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    let simT = 20, intro = 0, fade = 1;
    function draw() {
      pass(P1, T.scene, W, H);
      gl.uniform2f(P1.u.uRes, W, H); gl.uniform1f(P1.u.uTime, simT); gl.uniform1f(P1.u.uCenter, center);
      gl.uniform1f(P1.u.uPx, px); gl.uniform1f(P1.u.uScale, SCALE); gl.uniform1f(P1.u.uWaistX, waistX);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      blur(T.scene, T.a1, T.b1); blur(T.b1, T.a2, T.b2); blur(T.b2, T.a2, T.b2);
      pass(PF, null, W, H);
      bind(0, T.scene.tex, PF.u.uScene); bind(1, T.b1.tex, PF.u.uB1); bind(2, T.b2.tex, PF.u.uB2);
      gl.uniform2f(PF.u.uRes, W, H); gl.uniform1f(PF.u.uTime, simT); gl.uniform1f(PF.u.uIntro, intro); gl.uniform1f(PF.u.uFade, fade);
      gl.uniform1f(PF.u.uCenter, center); gl.uniform1f(PF.u.uScale, SCALE); gl.uniform1f(PF.u.uWaistX, waistX);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    window.__oazyWarp = { render(t) { simT = t; intro = 1; fade = 1; draw(); } };     // lets a test render any moment
    resize();
    if (reduce) { intro = 1; draw(); addEventListener("resize", () => { resize(); draw(); }); return; }

    // ~30 frames a second is plenty for a slow drift, and keeps the GPU (and the charts) relaxed
    let visible = true, running = false, last = 0, start = 0, shown = 0, frames = 0, acc = 0, sp = pace;
    function frame(now) {
      if (!visible || document.hidden) { running = false; return; }
      requestAnimationFrame(frame);
      if (now - shown < 31) return;
      const dt = Math.min(0.08, (now - (last || now)) / 1000); last = now; shown = now;
      if (!start) start = now;
      const k = Math.min(1, (now - start) / 2200);
      intro = 1 - Math.pow(1 - k, 3);
      sp += (pace - sp) * Math.min(1, dt * 0.8);                   // ease between "training" and "idle"
      simT += dt * sp * (1 + 4 * Math.pow(1 - k, 2.4));            // rush in, then drift
      const r = host.getBoundingClientRect();
      fade = Math.max(0.2, Math.min(1, 1 + r.top / (r.height * 0.8)));
      draw();
      if (k >= 1) {                                                // a slow GPU gets a lower resolution
        frames++; acc += dt;
        if (frames === 30) {
          if (acc / frames > 0.045 && quality > 0.5) { quality *= 0.8; resize(); }
          frames = 0; acc = 0;
        }
      }
    }
    const go = () => { if (!running && visible && !document.hidden) { running = true; last = 0; requestAnimationFrame(frame); } };
    addEventListener("resize", resize);
    new ResizeObserver(resize).observe(host);
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; go(); }).observe(host);
    document.addEventListener("visibilitychange", go);
    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); running = false; });
    go();
  }

  /* ------------------------------------------------------------------ below the head: a photon field under clear glass */
  // A faint photon field (the page as a detector catching the hero's light: tiny points that flash as photons land,
  // more where the light pools), and every panel (.card) a pane of clear glass over them: flat and calm in the middle
  // for reading, a thick rounded rim that bends the light behind it (blue more than red, so its edge splits into
  // colour) and catches the light from the top left. The same shader as the landing page.
  function land() {
    const cv = document.getElementById("land");
    const plain = () => { if (cv) cv.remove(); document.documentElement.classList.add("noland"); };
    if (!cv) return plain();
    const gl = cv.getContext("webgl", { antialias: false, alpha: false, premultipliedAlpha: false, powerPreference: "low-power" });
    if (!gl || !gl.getExtension("OES_standard_derivatives")) return plain();
    const vs = "attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}";
    const fs = `#extension GL_OES_standard_derivatives : enable
precision highp float;
uniform vec2 uRes; uniform float uT, uTop, uDpr, uPa, uScroll, uCw; uniform vec2 uP; uniform vec4 uB[12]; uniform float uBR[12]; uniform int uBN; 
float h21(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
float fbm(vec2 p){ float s = 0., a = .55; for (int i = 0; i < 3; i++) { s += a * vn(p); p = p * 1.9 + vec2(1.7, 9.2); a *= .42; } return s; }
float ln(float v, float w){ float d = abs(fract(v + .5) - .5); return 1. - smoothstep(.45 * w, 1.45 * w, d); }
// below the hero: a photon field. The page is a detector slowly catching the hero's light: a fine,
// slightly irregular grid of tiny points, each one flashing now and then as a single photon lands
// and fading after, more often where the light is stronger (soft pools of the hero's colours that
// drift slowly, brighter towards the margins and under the pointer). Each point splits faintly into
// colour like the hero's lens, and where many land the detector keeps a soft glow, like an exposure.
vec3 field(float h){
  h = fract(h) * 4.;
  vec3 a = vec3(.80, .76, 1.), b = vec3(.58, .46, 1.), c = vec3(.45, .82, 1.), d = vec3(1., .58, .84);
  float t = smoothstep(0., 1., fract(h));
  if (h < 1.) return mix(a, b, t); if (h < 2.) return mix(b, c, t); if (h < 3.) return mix(c, d, t); return mix(d, a, t);
}
vec2 h22(vec2 p){ return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453); }
vec3 land(vec2 css, float wIn, out float wOut){
  wOut = 1.;
  float vw = uRes.x / uDpr, t = uT;
  vec2 pc = vec2(css.x, css.y + uScroll * .5);                    // a little depth when scrolling
  vec2 q = pc / 620.;
  // how much light falls here
  float L = smoothstep(.2, .9, vn(q + vec2(t * .01, -t * .008))) * .8 + .2 * vn(q * 3. + 5.);
  float calm = mix(.35, 1., smoothstep(uCw - 80., uCw + 160., abs(css.x - vw * .5)));
  vec2 dp = (css - uP) / 150.; L += .7 * uPa * exp(-dot(dp, dp));
  L *= calm;
  vec3 tint = field(vn(q * .8 + vec2(-t * .006, t * .004)) * 1.4);
  // the detector's points
  const float G = 9.;
  vec2 cell = floor(pc / G), f = pc - (cell + .5) * G;
  vec2 hh = h22(cell);
  float rate = .05 + .25 * hh.y;
  float ph = fract(t * rate + hh.x * 7.);
  // a photon lands: a quick flash, then a slow fade
  float on = step(hh.x, .25 + .75 * L) * smoothstep(0., .04, ph) * exp(-ph * 6.);
  vec2 d = f - (h22(cell + 9.) - .5) * 4.;
  vec3 dot3 = exp(-vec3(dot(d + vec2(.55, 0.), d + vec2(.55, 0.)), dot(d, d), dot(d - vec2(.55, 0.), d - vec2(.55, 0.))) / .5);
  vec3 c = dot3 * on * mix(tint, vec3(1.), .35 * on) * (.45 + 1.4 * L);
  c += tint * L * L * .022;                                          // the exposure
  return c;
}
float sdBox(vec2 p, vec2 b, float r){ vec2 q = abs(p) - b + r; return length(max(q, 0.)) + min(max(q.x, q.y), 0.) - r; }
// distance to the nearest glass pane (negative inside); near the pointer the pane's skin swells towards it
float panes(vec2 css){
  float sd = 1e5;
  for (int i = 0; i < 12; i++) {
    if (i >= uBN) break;
    vec4 b = uB[i]; vec2 c = b.xy + b.zw * .5, hb = b.zw * .5;
    float d = sdBox(css - c, hb, uBR[i]);
    float dp = sdBox(uP - c, hb, uBR[i]);
    vec2 e = (css - uP) / 70.;
    d -= 14. * uPa * exp(-dp * dp / 3200.) * exp(-dot(e, e));
    sd = min(sd, d);
  }
  return sd;
}
void main(){
  vec2 css = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y) / uDpr;
  float w0; vec3 c = land(css, -1., w0);
  float sd = panes(css);
  if (sd < 1.5) {
    // glass: a flat clear middle and a thick rounded rim. The rim bends the lines behind it
    // outwards (more for blue than red, so edges split into colour) and catches the light
    float e = 1.5;
    vec2 n = normalize(vec2(panes(css + vec2(e, 0.)) - panes(css - vec2(e, 0.)), panes(css + vec2(0., e)) - panes(css - vec2(0., e))) + 1e-5);
    float rim = smoothstep(-26., 0., min(sd, 0.));            // 0 in the middle, 1 at the edge
    float bend = pow(rim, 2.4) * 34.;
    vec3 g = c;
    float wr = w0 * (1. + bend * .04), wx;
    if (bend > .05) g = vec3(land(css + n * bend * .92, wr, wx).r, land(css + n * bend, wr, wx).g, land(css + n * bend * 1.1, wr, wx).b);
    g *= mix(.4, 1.25, pow(rim, 1.6));                         // calm middle for reading, denser light at the rim
    float lit = .55 + .45 * dot(n, normalize(vec2(-.55, -.85))); // light from the top left
    g += vec3(.92, .94, 1.) * (pow(rim, 9.) * .22 * lit + pow(rim, 3.) * .028);
    g += vec3(.006, .006, .01);
    c = mix(c, g, smoothstep(.75, -.75, sd));
  }
  float y = css.y * uDpr;
  c *= smoothstep(uTop + 40. * uDpr, uTop + 380. * uDpr, y);              // fades in below the hero
  c += (h21(gl_FragCoord.xy + fract(uT)) - .5) / 255.;
  gl_FragColor = vec4(c + vec3(.0196, .0196, .0275), 1.);
}`;
    const sh = (t, src) => { const o = gl.createShader(t); gl.shaderSource(o, src); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(o); return o; };
    let pr;
    try { pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(pr); gl.useProgram(pr); }
    catch (e) { console.warn(e); return plain(); }
    document.documentElement.classList.add("land");
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const la = gl.getAttribLocation(pr, "a"); gl.enableVertexAttribArray(la); gl.vertexAttribPointer(la, 2, gl.FLOAT, false, 0, 0);
    const U = (n) => gl.getUniformLocation(pr, n);
    const uRes = U("uRes"), uT = U("uT"), uTop = U("uTop"), uDpr = U("uDpr"), uB = U("uB"), uBR = U("uBR"), uBN = U("uBN"), uScroll = U("uScroll"), uCw = U("uCw");
    gl.uniform2f(U("uP"), -999, -999); gl.uniform1f(U("uPa"), 0);   // no pointer swell: the lab stays still under the mouse
    const head = document.querySelector(".lab-head"), wrap = document.querySelector(".wrap");
    const N = 12, B = new Float32Array(N * 4), BR = new Float32Array(N), radius = new WeakMap();
    let dpr = 1;
    const size = () => { dpr = Math.min(devicePixelRatio || 1, 1.5); cv.width = Math.round(innerWidth * dpr); cv.height = Math.round(innerHeight * dpr); gl.viewport(0, 0, cv.width, cv.height); };
    function draw(t) {
      gl.uniform2f(uRes, cv.width, cv.height); gl.uniform1f(uT, t); gl.uniform1f(uDpr, dpr);
      gl.uniform1f(uTop, Math.max(0, (head ? head.getBoundingClientRect().bottom : 0) - 60) * dpr);
      // the panes on screen right now (the page re-draws its cards as data arrives, so look them up every frame)
      let n = 0;
      for (const el of document.querySelectorAll(".wrap .card")) {
        if (n >= N) break;
        const r = el.getBoundingClientRect();
        if (!r.width || r.bottom < -60 || r.top > innerHeight + 60) continue;
        if (!radius.has(el)) radius.set(el, parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0);
        B.set([r.left, r.top, r.width, r.height], n * 4); BR[n] = radius.get(el); n++;
      }
      gl.uniform4fv(uB, B); gl.uniform1fv(uBR, BR); gl.uniform1i(uBN, n);
      gl.uniform1f(uScroll, scrollY);
      const wr = wrap ? wrap.getBoundingClientRect() : null;
      gl.uniform1f(uCw, wr ? Math.min(wr.width * 0.5 - 16, innerWidth * 0.5 - 20) : 560);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    window.__oazyLand = { render: (t) => draw(t) };
    // every frame while the page moves (so the glass stays glued to its panel), ~30 a second when it is still
    let run = false, shown = 0, moved = 0;
    const t0 = performance.now();
    function frame(now) {
      if (!run) return;
      requestAnimationFrame(frame);
      if (now - moved > 250 && now - shown < 31) return;
      shown = now;
      draw(reduce ? 20 : 20 + (now - t0) / 1000);
    }
    const wanted = () => !document.hidden && (!head || head.getBoundingClientRect().bottom < innerHeight * 1.2);
    function check() {
      const w = wanted();
      if (w && !run) { run = true; requestAnimationFrame(frame); } else if (!w) run = false;
    }
    size(); draw(20);
    addEventListener("resize", () => { size(); moved = performance.now(); draw(20); check(); });
    addEventListener("scroll", () => { moved = performance.now(); check(); }, { passive: true });
    document.addEventListener("visibilitychange", check);
    cv.addEventListener("webglcontextlost", (e) => { e.preventDefault(); run = false; plain(); });
    check();
  }

  bar(); rail(); band(); land();
  return { setPace(p) { pace = p; } };
})();
