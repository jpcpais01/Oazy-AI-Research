/* The stage behind the app: the lab dashboard's photon field (oazy/web/lab.js, land()), the page as a detector slowly
   catching light, tiny points that flash as single photons land, and every .glass panel a pane of clear glass drawn by
   the light it bends (a calm middle for reading, a thick rim that splits into colour and catches the light from the
   top left). The same shader, tuned for a phone: rendered at the screen's full pixel density (sharp points on a phone's 3x screen), ~30 frames a second, slower
   while the model is writing (the processor is busy), stopped when the app is in the background. */
window.Land = (() => {
  const cv = document.getElementById("land");
  const plain = () => { if (cv) cv.remove(); document.documentElement.classList.add("noland"); return { busy() {}, visible() {}, kick() {} }; };
  if (!cv) return plain();
  const gl = cv.getContext("webgl", { antialias: false, alpha: false, premultipliedAlpha: false, powerPreference: "low-power" });
  if (!gl || !gl.getExtension("OES_standard_derivatives")) return plain();
  const vs = "attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}";
  const fs = `#extension GL_OES_standard_derivatives : enable
precision highp float;
uniform vec2 uRes; uniform float uT, uDpr, uScroll, uGlow; uniform vec3 uAcc; uniform vec4 uB[10]; uniform float uBR[10]; uniform int uBN;
float h21(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
vec3 field(float h){
  h = fract(h) * 4.;
  vec3 a = vec3(.80, .76, 1.), b = vec3(.58, .46, 1.), c = vec3(.45, .82, 1.), d = vec3(1., .58, .84);
  float t = smoothstep(0., 1., fract(h));
  if (h < 1.) return mix(a, b, t); if (h < 2.) return mix(b, c, t); if (h < 3.) return mix(c, d, t); return mix(d, a, t);
}
vec2 h22(vec2 p){ return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453); }
vec3 land(vec2 css){
  float vh = uRes.y / uDpr, t = uT;
  vec2 pc = vec2(css.x, css.y + uScroll * .35);
  vec2 q = pc / 520.;
  float L = smoothstep(.2, .9, vn(q + vec2(t * .01, -t * .008))) * .8 + .2 * vn(q * 3. + 5.);
  // a soft pool of the model's own colour rising from the bottom while it writes
  float pool = uGlow * smoothstep(vh * .45, vh * 1.05, css.y);
  L = L * (.75 + .25 * smoothstep(0., 140., css.y)) + .55 * pool;
  vec3 tint = mix(field(vn(q * .8 + vec2(-t * .006, t * .004)) * 1.4), uAcc, .55 * pool);
  const float G = 9.;
  vec2 cell = floor(pc / G), f = pc - (cell + .5) * G;
  vec2 hh = h22(cell);
  float rate = .05 + .25 * hh.y;
  float ph = fract(t * rate + hh.x * 7.);
  float on = step(hh.x, .25 + .75 * L) * smoothstep(0., .04, ph) * exp(-ph * 6.);
  vec2 d = f - (h22(cell + 9.) - .5) * 4.;
  vec3 dot3 = exp(-vec3(dot(d + vec2(.55, 0.), d + vec2(.55, 0.)), dot(d, d), dot(d - vec2(.55, 0.), d - vec2(.55, 0.))) / .5);
  vec3 c = dot3 * on * mix(tint, vec3(1.), .35 * on) * (.45 + 1.4 * L);
  c += tint * L * L * .022;
  return c;
}
float sdBox(vec2 p, vec2 b, float r){ vec2 q = abs(p) - b + r; return length(max(q, 0.)) + min(max(q.x, q.y), 0.) - r; }
float panes(vec2 css){
  float sd = 1e5;
  for (int i = 0; i < 10; i++) {
    if (i >= uBN) break;
    vec4 b = uB[i];
    sd = min(sd, sdBox(css - (b.xy + b.zw * .5), b.zw * .5, uBR[i]));
  }
  return sd;
}
void main(){
  vec2 css = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y) / uDpr;
  vec3 c = land(css);
  float sd = panes(css);
  if (sd < 1.5) {
    float e = 1.5;
    vec2 n = normalize(vec2(panes(css + vec2(e, 0.)) - panes(css - vec2(e, 0.)), panes(css + vec2(0., e)) - panes(css - vec2(0., e))) + 1e-5);
    float rim = smoothstep(-22., 0., min(sd, 0.));
    float bend = pow(rim, 2.4) * 30.;
    vec3 g = c;
    if (bend > .05) g = vec3(land(css + n * bend * .92).r, land(css + n * bend).g, land(css + n * bend * 1.1).b);
    g *= mix(.38, 1.25, pow(rim, 1.6));
    float lit = .55 + .45 * dot(n, normalize(vec2(-.55, -.85)));
    g += vec3(.92, .94, 1.) * (pow(rim, 9.) * .22 * lit + pow(rim, 3.) * .028);
    g += vec3(.006, .006, .01);
    c = mix(c, g, smoothstep(.75, -.75, sd));
  }
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
  const uRes = U("uRes"), uT = U("uT"), uDpr = U("uDpr"), uB = U("uB"), uBR = U("uBR"), uBN = U("uBN"), uScroll = U("uScroll"), uGlow = U("uGlow"), uAcc = U("uAcc");
  const N = 10, B = new Float32Array(N * 4), BR = new Float32Array(N), radius = new WeakMap();
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let dpr = 1, glow = 0, glowTo = 0, acc = [1, .54, .47], busy = false, shown = true;
  const size = () => { dpr = Math.min(devicePixelRatio || 1, 3); cv.width = Math.round(innerWidth * dpr); cv.height = Math.round(innerHeight * dpr); gl.viewport(0, 0, cv.width, cv.height); };
  const scroller = () => document.getElementById("scroll");
  function draw(t) {
    glow += (glowTo - glow) * 0.06;
    gl.uniform2f(uRes, cv.width, cv.height); gl.uniform1f(uT, t); gl.uniform1f(uDpr, dpr);
    gl.uniform1f(uGlow, glow); gl.uniform3f(uAcc, acc[0], acc[1], acc[2]);
    let n = 0;
    for (const el of document.querySelectorAll(".glass")) {
      if (n >= N) break;
      const r = el.getBoundingClientRect();
      if (!r.width || r.bottom < -40 || r.top > innerHeight + 40 || el.closest("[hidden]")) continue;
      if (getComputedStyle(el).visibility === "hidden") continue;
      if (!radius.has(el)) radius.set(el, parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0);
      B.set([r.left, r.top, r.width, r.height], n * 4); BR[n] = radius.get(el); n++;
    }
    gl.uniform4fv(uB, B); gl.uniform1fv(uBR, BR); gl.uniform1i(uBN, n);
    const s = scroller();
    gl.uniform1f(uScroll, s ? s.scrollTop : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  let run = false, last = 0, moved = 0;
  const t0 = performance.now();
  function frame(now) {
    if (!run) return;
    requestAnimationFrame(frame);
    const gap = now - moved < 300 ? 0 : busy ? 80 : 33;     // glued to moving panels; calmer while the model writes
    if (now - last < gap) return;
    last = now;
    draw(reduce ? 20 : 20 + (now - t0) / 1000);
  }
  function check() {
    const want = shown && !document.hidden;
    if (want && !run) { run = true; requestAnimationFrame(frame); } else if (!want) run = false;
  }
  size(); draw(20);
  addEventListener("resize", () => { size(); moved = performance.now(); draw(20); });
  document.addEventListener("scroll", () => { moved = performance.now(); }, { passive: true, capture: true });
  document.addEventListener("visibilitychange", check);
  cv.addEventListener("webglcontextlost", (e) => { e.preventDefault(); run = false; plain(); });
  check();
  const hex = (h) => { const m = /^#?([0-9a-f]{6})$/i.exec(h || ""); if (!m) return null; const v = parseInt(m[1], 16); return [(v >> 16) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]; };
  return {
    busy(b) { busy = b; glowTo = b ? 1 : 0; },
    accent(h) { acc = hex(h) || acc; },
    visible(v) { shown = v; check(); },
    kick() { moved = performance.now(); },
  };
})();
