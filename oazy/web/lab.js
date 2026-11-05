// The training lab's look, borrowed from the landing page (oazy/site): a thin band of light streaks behind the
// page head (WebGL, a calmer cut of the landing page's tunnel), plus the launcher-style bar with sliding tabs.
// It is decoration only: if WebGL is missing, a CSS glow stays, and nothing here touches the data.
"use strict";

const LAB = (() => {
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

  /* ------------------------------------------------------------------ the light band */
  let pace = 1;          // 1: models training (streaks drift), ~0.15: nothing training (almost still)
  function band() {
    const canvas = document.getElementById("warp");
    if (!canvas) return;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const opts = { antialias: false, alpha: false, depth: false, stencil: false, premultipliedAlpha: false, powerPreference: "low-power" };
    let gl = canvas.getContext("webgl2", opts);
    const v2 = !!gl;
    if (!gl) gl = canvas.getContext("webgl", opts);
    if (!gl) { canvas.remove(); return; }

    let texType = gl.UNSIGNED_BYTE, texIFmt = gl.RGBA, hdr = false;
    if (v2 && gl.getExtension("EXT_color_buffer_float")) { hdr = true; texType = gl.HALF_FLOAT; texIFmt = gl.RGBA16F; }
    const SCALE = hdr ? 1.0 : 0.45;

    const HEAD_V = v2 ? "#version 300 es\n#define ATTR in\n#define VARY out\n" : "#define ATTR attribute\n#define VARY varying\n";
    const HEAD_F = v2 ? "#version 300 es\nprecision highp float;\n#define VARY in\n#define TEX texture\n#define OUT o\nout vec4 o;\n"
                      : "precision highp float;\n#define VARY varying\n#define TEX texture2D\n#define OUT gl_FragColor\n";
    const VERT = HEAD_V + "ATTR vec2 p; VARY vec2 vUv; void main(){ vUv = p * 0.5 + 0.5; gl_Position = vec4(p, 0., 1.); }";

    // the streak field: lanes of hair-thin light running outwards from a waist, in 3 depth layers
    const FIELD = HEAD_F + `
VARY vec2 vUv;
uniform vec2 uRes; uniform float uTime, uCenter, uPx, uScale, uWaistX;
float h11(float n) { return fract(sin(n * 91.3458) * 47453.5453); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  float a = h11(i.x + i.y * 57.0), b = h11(i.x + 1.0 + i.y * 57.0);
  float c = h11(i.x + (i.y + 1.0) * 57.0), d = h11(i.x + 1.0 + (i.y + 1.0) * 57.0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
vec3 tint(float r) {
  return r < 0.40 ? vec3(0.80, 0.78, 1.00) : r < 0.64 ? vec3(0.62, 0.68, 1.00) : r < 0.80 ? vec3(0.66, 0.56, 1.00)
       : r < 0.90 ? vec3(0.60, 0.88, 1.00) : r < 0.95 ? vec3(1.00, 0.62, 0.66) : vec3(0.62, 1.00, 0.80);
}
vec3 layer(vec2 p, float aspect, float id, float N, float waist, float flare, float gain, float spd) {
  float xn = p.x / (0.5 * aspect);
  float ax = abs(xn);
  float y = p.y + 0.008 * sin(xn * 1.7 + uTime * 0.11 + id * 1.7) * ax;
  float w = waist + flare * ax * ax;
  float s = y / w;
  float env = exp(-s * s * 1.35) * (0.55 + 0.45 * exp(-s * s * 5.0));
  if (env < 0.003) return vec3(0.0);
  float lane = floor(s * N);
  float r1 = h11(lane * 1.37 + id * 17.31), r2 = h11(lane * 2.11 + id * 5.71 + 3.1);
  float r3 = h11(lane * 0.73 + id * 9.13 + 7.7), r4 = h11(lane * 4.31 + id * 1.97 + 1.3);
  float lanePx = w * uRes.y / N;
  float sc = (lane + 0.5 + (r4 - 0.5) * 0.7) / N;
  float dpx = abs(s - sc) * w * uRes.y;
  float t = uPx * (0.55 + 0.55 * ax) * (0.7 + 0.6 * r3);
  float tt = max(t, 0.75);
  float prof = exp(-dpx * dpx / (tt * tt)) * min(1.0, t / 0.75);
  float u = log(ax + 0.035);
  float freq = mix(1.8, 4.6, r2);
  float evolve = uTime * mix(0.05, 0.16, r1);
  float n = vnoise(vec2(u * freq - uTime * spd * mix(0.6, 1.4, r1) + r3 * 40.0, evolve + r4 * 31.0));
  float n2 = vnoise(vec2(u * freq * 3.1 - uTime * spd * 1.7 + r2 * 23.0, evolve * 1.6 + 7.0));
  float r5 = h11(lane * 7.77 + id * 3.33 + 5.9);
  float streak = smoothstep(0.56, 0.95, n) * smoothstep(0.15, 0.6, n2) * smoothstep(0.25, 0.45, r5);
  streak = streak * streak * (3.0 - 2.0 * streak);
  float bright = 1.35 * (0.25 + 0.75 * r1 * r1) * (r2 > 0.955 ? 2.6 : 1.0);
  float xg = mix(0.32, 1.0, smoothstep(0.0, 0.9, ax));
  vec3 line = tint(fract(r3 * 7.13 + r1)) * prof * streak * bright;
  vec3 haze = vec3(0.70, 0.66, 1.00) * 0.045 * min(1.0, 1.6 * tt / max(lanePx, 0.001));
  return mix(haze, line, smoothstep(1.3, 3.6, lanePx)) * env * xg * gain;
}
void main() {
  vec2 frag = vUv * uRes;
  float aspect = uRes.x / uRes.y;
  vec2 p = vec2((frag.x - uWaistX * uRes.x) / uRes.y, frag.y / uRes.y - uCenter);
  vec3 c = vec3(0.0);
  c += layer(p, aspect, 0.0, 120.0, 0.16, 0.10, 0.85, 0.050);
  c += layer(p, aspect, 1.0,  90.0, 0.19, 0.15, 1.00, 0.080);
  c += layer(p, aspect, 2.0,  52.0, 0.23, 0.22, 1.25, 0.130);
  OUT = vec4(c * uScale, 1.0);
}`;
    const BLUR = HEAD_F + `
VARY vec2 vUv; uniform sampler2D uTex; uniform vec2 uDir;
void main() {
  vec3 c = TEX(uTex, vUv).rgb * 0.2270270270;
  c += (TEX(uTex, vUv + uDir * 1.3846153846).rgb + TEX(uTex, vUv - uDir * 1.3846153846).rgb) * 0.3162162162;
  c += (TEX(uTex, vUv + uDir * 3.2307692308).rgb + TEX(uTex, vUv - uDir * 3.2307692308).rgb) * 0.0702702703;
  OUT = vec4(c, 1.0);
}`;
    // final: soft colour fringing, bloom, tone, vignette, grain, and a fade into the page black at the bottom
    const FINAL = HEAD_F + `
VARY vec2 vUv;
uniform sampler2D uScene, uB1; uniform float uTime, uIntro, uFade, uCenter, uScale, uWaistX;
float h21(vec2 p) { p = fract(p * vec2(233.34, 851.73)); p += dot(p, p + 23.45); return fract(p.x * p.y); }
void main() {
  vec2 uv = vUv, d = uv - vec2(uWaistX, uCenter);
  vec2 ca = vec2(d.x * 0.008, d.y * 0.003);
  vec3 sc = vec3(TEX(uScene, uv + ca).r, TEX(uScene, uv).g, TEX(uScene, uv - ca).b);
  vec3 c = (sc + TEX(uB1, uv).rgb * 0.9) / uScale;
  float ax = abs(d.x) * 2.0;
  c += vec3(0.20, 0.16, 0.55) * 0.05 * exp(-pow(d.y / (0.12 + 0.2 * ax * ax), 2.0));
  c *= uIntro * uFade * 0.8;
  c = vec3(1.0) - exp(-c * 2.0);
  c *= mix(0.45, 1.0, smoothstep(1.2, 0.2, length(d * vec2(0.9, 1.4))));
  c *= smoothstep(0.0, 0.42, uv.y);                       // melt into the page below
  c += vec3(0.0196, 0.0196, 0.0275);                      // the page black (#050507)
  c += (h21(gl_FragCoord.xy + fract(uTime * 7.0) * 113.0) - 0.5) * 0.018 * (0.3 + c);
  c += (h21(gl_FragCoord.yx + 3.7) - 0.5) / 255.0;
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
    const PF = program(FINAL, ["uScene", "uB1", "uTime", "uIntro", "uFade", "uCenter", "uScale", "uWaistX"]);
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
    const host = canvas.parentElement;
    let W = 0, H = 0, px = 1, quality = 1, T = {}, waistX = 0.5, center = 0.5;
    function resize() {
      const r = canvas.getBoundingClientRect();
      px = Math.min(devicePixelRatio || 1, 1.25) * quality;
      const nW = Math.max(8, Math.round(r.width * px)), nH = Math.max(8, Math.round(r.height * px));
      // the waist sits behind the page title on wide screens, centred on narrow ones
      waistX = r.width > 900 ? 0.62 : 0.5;
      center = r.width > 900 ? 0.52 : 0.58;
      if (nW === W && nH === H) return;
      W = nW; H = nH; canvas.width = W; canvas.height = H;
      Object.values(T).forEach((t) => { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); });
      const q = (n) => Math.max(4, Math.round(n / 4));
      T = { scene: target(W, H), a: target(q(W), q(H)), b: target(q(W), q(H)) };
    }
    const pass = (prog, out, w, h) => { gl.useProgram(prog.pr); gl.bindFramebuffer(gl.FRAMEBUFFER, out ? out.fb : null); gl.viewport(0, 0, w, h); };
    const bind = (unit, tex, loc) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(loc, unit); };
    let simT = 24, intro = 0, fade = 1;
    function draw() {
      pass(P1, T.scene, W, H);
      gl.uniform2f(P1.u.uRes, W, H); gl.uniform1f(P1.u.uTime, simT); gl.uniform1f(P1.u.uCenter, center);
      gl.uniform1f(P1.u.uPx, px); gl.uniform1f(P1.u.uScale, SCALE); gl.uniform1f(P1.u.uWaistX, waistX);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      pass(PB, T.a, T.a.w, T.a.h); bind(0, T.scene.tex, PB.u.uTex); gl.uniform2f(PB.u.uDir, 1 / W, 0); gl.drawArrays(gl.TRIANGLES, 0, 3);
      pass(PB, T.b, T.b.w, T.b.h); bind(0, T.a.tex, PB.u.uTex); gl.uniform2f(PB.u.uDir, 0, 1 / T.a.h); gl.drawArrays(gl.TRIANGLES, 0, 3);
      pass(PF, null, W, H);
      bind(0, T.scene.tex, PF.u.uScene); bind(1, T.b.tex, PF.u.uB1);
      gl.uniform1f(PF.u.uTime, simT); gl.uniform1f(PF.u.uIntro, intro); gl.uniform1f(PF.u.uFade, fade);
      gl.uniform1f(PF.u.uCenter, center); gl.uniform1f(PF.u.uScale, SCALE); gl.uniform1f(PF.u.uWaistX, waistX);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
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
      const k = Math.min(1, (now - start) / 2000);
      intro = 1 - Math.pow(1 - k, 3);
      sp += (pace - sp) * Math.min(1, dt * 0.8);                   // ease between "training" and "idle"
      simT += dt * sp * (1 + 6 * Math.pow(1 - k, 2.4));            // rush in, then drift
      const r = host.getBoundingClientRect();
      fade = Math.max(0.2, Math.min(1, 1 + r.top / (r.height * 0.8)));
      draw();
      if (k >= 1) {                                                // a slow GPU gets a lower resolution
        frames++; acc += dt;
        if (frames === 30) {
          if (acc / frames > 0.045 && quality > 0.45) { quality *= 0.8; resize(); }
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

  bar(); band();
  return { setPace(p) { pace = p; } };
})();
