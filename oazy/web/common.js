// Shared drawing helpers for the Oazy dashboard and chat app (no libraries).
"use strict";

const UI = (() => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
  const font = (size, weight = 400) => `${weight} ${size}px ${FONT}`;

  function fitCanvas(cv, h) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    if (h != null) cv.style.height = h + "px";
    const H = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(H * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h: H };
  }

  function b64bytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    const mult = [1, 2, 2.5, 5, 10].find((m) => span / (step0 * m) <= n) || 10;
    const step = step0 * mult;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  function fmtTokens(v) {
    if (v >= 1e9) return (v / 1e9).toFixed(v % 1e9 ? 1 : 0) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(v % 1e6 ? 1 : 0) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(0) + "k";
    return String(Math.round(v));
  }

  // "#rrggbb" or "rgb(r,g,b)" -> "rgba(r,g,b,a)"
  function alpha(color, a) {
    if (color.startsWith("#")) {
      const n = parseInt(color.length === 4 ? color.replace(/#(.)(.)(.)/, "$1$1$2$2$3$3") : color.slice(1), 16);
      return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
    }
    const m = color.match(/[\d.]+/g);
    return m ? `rgba(${m[0]},${m[1]},${m[2]},${a})` : color;
  }

  function placeTip(tip, x, y, boxW, boxH) {
    tip.style.display = "block";
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = x + 16;
    if (left + tw > boxW - 2) left = x - 16 - tw;
    if (left < 2) left = 2;
    let top = y;
    if (boxH != null && top + th > boxH) top = Math.max(2, boxH - th);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  // Line chart with a crosshair tooltip.
  // series: [{name, color, dash, width, alpha, area, dots, noTip, endLabel, points: [[x,y],...]}]
  function lineChart(cv, tip, opt) {
    const state = { opt, hoverX: null, hoverY: 0 };
    function draw() {
      const { ctx, w, h } = fitCanvas(cv);
      const o = state.opt;
      ctx.clearRect(0, 0, w, h);
      const hasEnd = o.series.some((s) => s.endLabel && s.points.length);
      const pad = { l: 44, r: hasEnd ? 58 : 12, t: 12, b: 26 };
      const all = o.series.flatMap((s) => s.points);
      if (!all.length) {
        ctx.fillStyle = css("--muted");
        ctx.font = font(13);
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        if (o.empty !== "") ctx.fillText(o.empty || "Waiting for data…", w / 2, h / 2);
        tip.style.display = "none";
        return;
      }
      let x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
      if (x1 === x0) x1 = x0 + 1;
      const ys = all.map((p) => p[1]).filter((v) => isFinite(v));
      let y0 = o.yMin != null ? o.yMin : Math.min(...ys), y1 = o.yMax != null ? o.yMax : Math.max(...ys);
      if (o.yClip) y1 = Math.min(y1, o.yClip(ys));
      if (y1 === y0) { y1 += 1; y0 -= o.yMin != null ? 0 : 1; }
      const padY = (y1 - y0) * 0.08;
      if (o.yMin == null) y0 -= padY;
      if (o.yMax == null) y1 += padY;
      const X = (v) => pad.l + ((v - x0) / (x1 - x0)) * (w - pad.l - pad.r);
      const Y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (h - pad.t - pad.b);
      const bottom = h - pad.b;
      // grid + axes
      ctx.font = font(11);
      ctx.lineWidth = 1;
      const yt = niceTicks(y0, y1, Math.max(2, Math.min(5, Math.floor((h - 40) / 40))));
      for (const v of yt) {
        const y = Math.round(Y(v)) + 0.5;
        if (y < pad.t - 1 || y > bottom + 1) continue;
        ctx.strokeStyle = css("--grid");
        ctx.setLineDash([2, 4]);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = css("--muted");
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText((o.yFmt || ((x) => x))(v), pad.l - 8, y);
      }
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = css("--muted");
      let lastLab = null;
      for (const v of niceTicks(x0, x1, Math.max(2, Math.floor((w - 100) / 90)))) {
        const lab = (o.xFmt || ((x) => x))(v);
        if (lab === lastLab) continue;
        lastLab = lab;
        ctx.fillText(lab, X(v), bottom + 8);
      }
      ctx.strokeStyle = css("--axis");
      ctx.beginPath(); ctx.moveTo(pad.l, bottom + 0.5); ctx.lineTo(w - pad.r, bottom + 0.5); ctx.stroke();
      // lines
      ctx.save();
      ctx.beginPath(); ctx.rect(pad.l, pad.t - 4, w - pad.l - pad.r, bottom - pad.t + 6); ctx.clip();
      for (const s of o.series) {
        if (!s.points.length) continue;
        const pts = s.points.filter((p) => isFinite(p[1]));
        if (s.area && pts.length > 1) {
          const g = ctx.createLinearGradient(0, pad.t, 0, bottom);
          g.addColorStop(0, alpha(s.color, 0.22));
          g.addColorStop(1, alpha(s.color, 0));
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.moveTo(X(pts[0][0]), bottom);
          pts.forEach((p) => ctx.lineTo(X(p[0]), Y(p[1])));
          ctx.lineTo(X(pts[pts.length - 1][0]), bottom);
          ctx.closePath(); ctx.fill();
        }
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = s.alpha || 1;
        ctx.lineWidth = s.width || 2;
        ctx.setLineDash(s.dash || []);
        ctx.lineJoin = "round"; ctx.lineCap = "round";
        ctx.beginPath();
        pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
        if ((s.width || 2) >= 2 && !s.dash) { ctx.shadowColor = alpha(s.color, 0.55); ctx.shadowBlur = 12; }   // a soft neon glow
        ctx.stroke();
        ctx.shadowBlur = 0; ctx.shadowColor = "transparent";
        ctx.setLineDash([]);
        if (s.dots) {
          for (const p of pts) {
            ctx.fillStyle = css("--solid") || css("--surface"); ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.6, 0, 7); ctx.fill();
            ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.4, 0, 7); ctx.fill();
          }
        }
        ctx.globalAlpha = 1;
      }
      ctx.restore();
      // direct labels at the end of lines
      const labels = [];
      for (const s of o.series) {
        if (!s.endLabel || !s.points.length) continue;
        const p = s.points[s.points.length - 1];
        labels.push({ y: Y(p[1]), x: X(p[0]), text: s.endLabel === true ? (o.endFmt || o.tipFmt || o.yFmt || ((x) => x))(p[1]) : s.endLabel, color: s.color });
      }
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 15) labels[i].y = labels[i - 1].y + 15;
      for (const L of labels) {
        ctx.fillStyle = css("--solid"); ctx.beginPath(); ctx.arc(L.x, L.y, 5, 0, 7); ctx.fill();
        ctx.fillStyle = L.color; ctx.beginPath(); ctx.arc(L.x, L.y, 3.5, 0, 7); ctx.fill();
        ctx.font = font(11.5, 600); ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(L.text, w - pad.r + 8, Math.max(pad.t, Math.min(bottom, L.y)));
      }
      // crosshair
      if (state.hoverX != null && state.hoverX >= pad.l && state.hoverX <= w - pad.r) {
        const xv = x0 + ((state.hoverX - pad.l) / (w - pad.l - pad.r)) * (x1 - x0);
        const g = ctx.createLinearGradient(0, pad.t, 0, bottom);
        g.addColorStop(0, alpha(css("--text"), 0)); g.addColorStop(0.5, alpha(css("--text"), 0.35)); g.addColorStop(1, alpha(css("--text"), 0.1));
        ctx.strokeStyle = g;
        ctx.beginPath(); ctx.moveTo(state.hoverX + 0.5, pad.t); ctx.lineTo(state.hoverX + 0.5, bottom); ctx.stroke();
        const rows = [];
        for (const s of o.series) {
          if (!s.points.length || s.noTip) continue;
          let best = s.points[0];
          for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
          ctx.fillStyle = css("--solid");
          ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 6, 0, 7); ctx.fill();
          ctx.strokeStyle = s.color; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 4, 0, 7); ctx.stroke();
          const f = s.tipFmt || o.tipFmt || o.yFmt || ((x) => x);
          rows.push(`<div class="tiprow"><span class="sw" style="background:${s.color};opacity:${s.alpha && s.alpha < 1 ? 0.5 : 1}"></span>${s.name}<b>${f(best[1])}</b></div>`);
        }
        const note = o.tipNote ? o.tipNote(xv) : "";
        tip.innerHTML = `<div class="tiphead">${(o.xTip || o.xFmt || ((x) => x))(xv)}</div>` + rows.join("") + (note ? `<div class="tipnote">${note}</div>` : "");
        placeTip(tip, state.hoverX, 6, w);
      } else tip.style.display = "none";
    }
    cv.addEventListener("mousemove", (e) => { state.hoverX = e.offsetX; draw(); });
    cv.addEventListener("mouseleave", () => { state.hoverX = null; draw(); });
    return { draw, set(opt) { state.opt = opt; draw(); } };
  }

  // Colour ramp: deep sea -> blue -> aqua -> white foam
  const HEAT = [[10, 22, 44], [22, 64, 150], [38, 140, 240], [90, 220, 255], [228, 252, 255]];
  function heatRGB(t) {
    t = Math.max(0, Math.min(1, t));
    const f = t * (HEAT.length - 1), i = Math.min(HEAT.length - 2, Math.floor(f)), u = f - i;
    const a = HEAT[i], b = HEAT[i + 1];
    return a.map((v, k) => Math.round(v + (b[k] - v) * u));
  }
  const heat = (t) => `rgb(${heatRGB(t).join(",")})`;
  const heatCSS = () => `linear-gradient(90deg, ${HEAT.map((c) => `rgb(${c.join(",")})`).join(", ")})`;

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // Keeps the filled part of a range slider in sync (the CSS reads --p).
  function syncRange(el) {
    const lo = +el.min || 0, hi = +el.max || 100;
    el.style.setProperty("--p", (hi > lo ? ((+el.value - lo) / (hi - lo)) * 100 : 0) + "%");
  }
  function watchRanges(root = document) {
    root.querySelectorAll("input[type=range]").forEach((el) => {
      syncRange(el);
      el.addEventListener("input", () => syncRange(el));
    });
  }

  // A word piece as shown on the sheet: newlines and spaces made visible.
  const pieceLabel = (s) => s.replace(/\n/g, "⏎").trim() || "␣";

  // Loss in plain words: on average, the real next word was as hard to guess as picking from N equally likely words.
  const choices = (loss) => Math.exp(loss);
  const fmtChoices = (loss) => { const n = choices(loss); return n < 10 ? n.toFixed(1) : n < 1000 ? Math.round(n).toString() : fmtTokens(n); };

  return { css, font, fitCanvas, b64bytes, lineChart, fmtTokens, heat, heatRGB, heatCSS, escapeHtml, niceTicks, alpha,
           placeTip, syncRange, watchRanges, pieceLabel, choices, fmtChoices };
})();
