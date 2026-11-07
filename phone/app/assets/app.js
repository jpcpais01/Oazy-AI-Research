/* The Oazy phone app's page. It talks to the phone through window.OazyNative (Bridge.java); events come back through
   oazy.event({...}). Opened in a desktop browser (no OazyNative), it runs on a stand-in so the design can be checked. */
(() => {
  const $ = (s) => document.querySelector(s);
  const N = window.OazyNative || mock();
  const store = {
    get(k, d) { try { const v = localStorage.getItem("oazy." + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("oazy." + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };
  const device = JSON.parse(N.device() || "{}");

  // ------------------------------------------------------------------ settings (each explained in plain words)
  const SETTINGS = [
    { k: "temp", grp: "main", label: "Creativity", min: 0, max: 1.5, step: 0.05, def: 0.7, fmt: (v) => v.toFixed(2),
      help: "Temperature. 0 always picks the likeliest word." },
    { k: "maxNew", grp: "main", label: "Answer length", min: 32, max: 1024, step: 16, def: 320, fmt: (v) => v + " tok",
      help: "The most it writes in one go. It often stops sooner." },
    { k: "rep", grp: "main", label: "Avoid repeating", min: 1, max: 1.5, step: 0.01, def: 1.1, fmt: (v) => v.toFixed(2),
      help: "Fewer loops. 1 = off." },
    { k: "topK", grp: "more", label: "Top-k", min: 1, max: 100, step: 1, def: 40, fmt: (v) => v,
      help: "Only the k likeliest next words are considered." },
    { k: "topP", grp: "more", label: "Top-p", min: 0.5, max: 1, step: 0.01, def: 0.95, fmt: (v) => v.toFixed(2),
      help: "Of those, keep the likeliest that cover this share of the chance." },
    { k: "threads", grp: "more", label: "Processor cores", min: 1, max: Math.max(2, device.cores || 8), step: 1, def: device.fast || 4, fmt: (v) => v,
      help: `This phone has ${device.cores || "?"}, ${device.fast || "?"} of them fast. Applies when a model opens.` },
    { k: "ctx", grp: "more", label: "Conversation memory", min: 512, max: 8192, step: 256, def: 2048, fmt: (v) => v.toLocaleString("en") + " tok",
      help: "How much of the chat it keeps in view (trained on 2,048). Applies when a model opens." },
  ];
  const S = Object.fromEntries(SETTINGS.map((s) => [s.k, store.get("set." + s.k, s.def)]));

  // ------------------------------------------------------------------ state
  let models = [], info = null, busy = false, mode = store.get("mode", "chat"), cur = null, loadingFile = null, started = false;
  const fmtN = (n) => n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + "B" : n >= 1e6 ? Math.round(n / 1e6).toLocaleString("en") + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : String(n);
  const fmtB = (b) => b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : (b / 1e6).toFixed(0) + " MB";
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  function toast(t, ms = 2600) {
    const el = $("#toast");
    el.textContent = t;
    el.hidden = false;
    clearTimeout(toast.t);
    toast.t = setTimeout(() => { el.hidden = true; }, ms);
  }

  // ------------------------------------------------------------------ models
  function describe(m) {
    const kind = m.kind || "";
    const moe = +m.n_expert > 0;
    const layers = m.layers || "";
    const mem = (layers.match(/M/g) || []).length, att = (layers.match(/A/g) || []).length;
    const parts = [];
    if (moe) parts.push(`${m.n_expert} experts, ${m.top_k} per word`);
    parts.push(mem ? `${mem} memory + ${att} attention layers` : `${att} attention layers`);
    return { kind, moe, line: parts.join(" · ") };
  }

  function refreshModels() {
    models = JSON.parse(N.models() || "[]");
    const list = $("#modelList");
    list.innerHTML = "";
    if (!models.length) list.innerHTML = `<p class="empty">No models on this phone yet.</p>`;
    for (const m of models) {
      const b = document.createElement("button");
      const on = info && info.file === m.file;
      b.className = "mrow" + (on ? " on" : "");
      b.style.setProperty("--m", m.accent || "#8f7bff");
      const tok = +m.tokens_trained ? ` · ${fmtN(+m.tokens_trained)} tokens` : "";
      b.innerHTML = `<i class="sw"></i><div><b>${esc(m.label || m.name)}</b><small>${(m.quant || "").toUpperCase()} · ${fmtB(+m.bytes)} · ${fmtN(+m.params)} params${tok}${on ? " · OPEN" : ""}</small></div>
        <span class="del" role="button" aria-label="Delete"><svg viewBox="0 0 16 16"><path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg></span>`;
      b.onclick = (e) => {
        if (e.target.closest(".del")) {
          if (confirmDelete(m)) { N.deleteModel(m.file); if (on) { info = null; paint(); } refreshModels(); }
          return;
        }
        closeSheets();
        open(m.file);
      };
      list.appendChild(b);
    }
    const free = device.free ? `${fmtB(device.free)} RAM free` : "";
    $("#freeSpace").textContent = free;
    paint();
  }

  function confirmDelete(m) {
    const k = "del." + m.file;
    if (confirmDelete.armed === k) { confirmDelete.armed = null; return true; }
    confirmDelete.armed = k;
    toast(`Tap the bin again to delete ${m.label || m.name}`);
    setTimeout(() => { if (confirmDelete.armed === k) confirmDelete.armed = null; }, 3000);
    return false;
  }

  function open(file) {
    const m = models.find((x) => x.file === file);
    if (!m) return;
    if (device.ram && +m.bytes > device.ram * 0.6) toast(`${m.label || m.name} needs ${fmtB(+m.bytes)}; this phone has ${fmtB(device.ram)} of memory. It may be slow or close.`, 5000);
    info = null;
    loadingFile = file;
    cur = null;
    $("#thread").innerHTML = "";
    started = false;
    store.set("last", file);
    N.load(file, S.threads, S.ctx);
    paint();
  }

  // ------------------------------------------------------------------ painting
  function paint() {
    const m = info ? models.find((x) => x.file === info.file) || info : loadingFile ? models.find((x) => x.file === loadingFile) : null;
    const acc = (m && m.accent) || "#8f7bff";
    document.documentElement.style.setProperty("--acc", acc);
    Land.accent(acc);
    document.body.classList.toggle("ready", !!info);
    document.body.classList.toggle("busy", busy);
    $("#pillName").textContent = m ? (m.label || m.name) : models.length ? "Pick a model" : "No model";
    $("#emptyCard").hidden = models.length > 0;
    $("#modelCard").hidden = !m;
    $("#welcome").hidden = started;
    $("#newChat").hidden = !started;
    if (m) {
      const d = describe(m);
      $("#mcName").textContent = m.label || m.name;
      $("#mcKind").textContent = `NEXT-WORD MODEL · ${(m.quant || "").toUpperCase()}`;
      $("#mcLine").textContent = d.line;
      $("#mcParams").textContent = fmtN(+m.params);
      const act = +m.active && +m.active < +m.params ? `${fmtN(+m.params)}` : fmtN(+m.params);
      $("#mcParams").textContent = act;
      $("#mcParams").nextElementSibling.textContent = d.moe ? `parameters, ${fmtN(+m.active)} per word` : "parameters";
      $("#mcTokens").textContent = +m.tokens_trained ? fmtN(+m.tokens_trained) : "—";
      const sp = store.get("speed." + (m.file || ""), null);
      $("#mcSpeed").textContent = sp ? sp.toFixed(1) : "—";
      $("#mcState").textContent = info ? "ready" : loadingFile ? "opening…" : "";
    }
    $("#send").disabled = !info && !busy;
    chips();
    facts();
  }

  const IDEAS = {
    chat: ["What is a black hole, simply?", "Give me three tips to sleep better", "Explain photosynthesis to a child",
      "Write a haiku about the sea", "Why is the sky blue?", "How do airplanes stay in the air?", "What is a good breakfast before a run?",
      "Explain gravity in two sentences", "Give me a fun fact about octopuses", "How do I make a simple tomato sauce?",
      "What causes the seasons?", "Write a short poem about autumn", "How can I focus better when studying?",
      "What is the difference between weather and climate?", "Tell me a short story about a lost robot", "Why do cats purr?",
      "How does the heart pump blood?", "Give me three ideas for a rainy day", "What is DNA, simply?", "How do vaccines work?",
      "Explain what a computer program is", "What makes a good friend?", "Why do leaves change colour?",
      "How do I start running as a beginner?", "Write a limerick about a cat", "What is the Moon made of?",
      "How do rainbows form?", "Give me tips to save money each month", "What is the tallest mountain in the world?",
      "Explain how a volcano erupts", "Suggest a name for a small café by the sea", "How do bees make honey?",
      "Why do we dream?", "Write a thank-you note to a teacher", "What is electricity?", "How can I sleep on a long flight?",
      "Describe Lisbon in three sentences", "What are the planets in our solar system?", "How do I keep a plant alive indoors?",
      "Explain the water cycle simply", "Give me a riddle", "What is a healthy lunch idea?", "How does the internet work, simply?"],
    write: ["Once upon a time, in a village by the sea,", "The most surprising thing about the ocean is", "Lisbon is a city where",
      "The old lighthouse keeper opened the door and", "Scientists have recently discovered that", "The best way to learn a new language is",
      "On the first day of spring,", "The robot looked at the stars and", "In the year 2100, cities will", "My grandmother always said that",
      "The secret to a perfect cup of coffee is", "Deep in the forest there was a small house where", "The history of the bicycle begins",
      "When the train stopped in the middle of nowhere,", "A good morning routine starts with", "The library was empty except for",
      "Volcanoes are formed when", "The little fox had never seen snow before, so", "Every great invention starts with",
      "The storm arrived just after midnight and", "Honeybees are important because", "The map led them to a beach where"],
  };
  const PICKS = { mode: null, on: false, list: [] };
  const draw = (pool, taken) => { const free = pool.filter((t) => !taken.includes(t)); return free[Math.floor(Math.random() * free.length)]; };
  function chips() {
    const box = $("#chips");
    if (!info) { box.innerHTML = ""; PICKS.list.forEach((c) => clearTimeout(c.t)); PICKS.list = []; PICKS.on = false; return; }
    if (PICKS.on && PICKS.mode === mode) return;           // keep the four on screen; each swaps on its own timer
    PICKS.list.forEach((c) => clearTimeout(c.t));
    box.innerHTML = "";
    PICKS.mode = mode; PICKS.on = true; PICKS.list = [];
    const pool = IDEAS[mode];
    for (let i = 0; i < Math.min(4, pool.length); i++) {
      const b = document.createElement("button");
      b.className = "chip";
      const c = { b, text: draw(pool, PICKS.list.map((x) => x.text)) };
      b.textContent = c.text;
      b.onclick = () => { $("#input").value = c.text; grow(); send(); };
      const swap = () => {
        c.t = setTimeout(() => {
          if (!b.isConnected) return;
          b.classList.add("fade");
          setTimeout(() => {
            c.text = draw(pool, PICKS.list.map((x) => x.text));
            b.textContent = c.text;
            b.classList.remove("fade");
            swap();
          }, 350);
        }, 15000 + Math.random() * 15000);
      };
      swap();
      PICKS.list.push(c);
      box.appendChild(b);
    }
  }

  function facts() {
    const f = $("#facts");
    const rows = [["Phone", device.model], ["Android", device.android], ["Memory", device.ram ? fmtB(device.ram) : "?"],
      ["Cores", device.cores ? `${device.cores} (${device.fast} fast)` : "?"]];
    if (info) rows.push(["Engine", `${info.kernel}, ${info.threads} threads`], ["Model", `${info.name} · ${info.quant} · ${info.layers} layers`],
      ["Opened in", `${(+info.load).toFixed(2)} s`]);
    rows.push(["Models folder", device.dir || "?"]);
    f.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(String(v || "?"))}</dd>`).join("");
  }

  // ------------------------------------------------------------------ the conversation
  const thread = $("#thread"), scroller = $("#scroll");
  let stick = true;
  scroller.addEventListener("scroll", () => { stick = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80; }, { passive: true });
  const down = (force) => { if (stick || force) scroller.scrollTop = scroller.scrollHeight; };

  function render(el, raw) {
    // plain text, with ``` code blocks kept apart
    const parts = raw.split("```");
    el.innerHTML = parts.map((p, i) => i % 2 ? `<pre>${esc(p.replace(/^[a-z]*\n/, ""))}</pre>` : esc(p)).join("");
  }

  function send() {
    const inp = $("#input");
    if (busy) { N.stop(); return; }
    const text = inp.value.trim();
    if (!info) { openSheet("models"); return; }
    if (!text && mode === "chat") return;
    inp.value = "";
    grow();
    started = true;
    paint();
    if (mode === "chat") {
      const u = document.createElement("div");
      u.className = "you";
      u.textContent = text;
      thread.appendChild(u);
    }
    const a = document.createElement("div");
    a.className = "ai thinking";
    a.innerHTML = `<div class="who"><i></i>${esc(info.label || info.name)}${mode === "write" ? " · writing on" : ""}</div><div class="text"></div><div class="meta"></div>`;
    thread.appendChild(a);
    cur = { el: a, text: "", seed: mode === "write" ? text : "", t0: performance.now() };
    if (cur.seed) a.querySelector(".text").innerHTML = `<span class="seed">${esc(cur.seed)}</span>`;
    busy = true;
    Land.busy(true);
    paint();
    down(true);
    if (!N.ask(mode, text, S.maxNew, S.temp, S.topK, S.topP, S.rep)) { done(null); note("The model is not open. Pick it again in Models.", true); }
  }

  function piece(t) {
    if (!cur) return;
    cur.el.classList.remove("thinking");
    cur.text += t;
    const out = mode === "chat" || !cur.seed ? cur.text.replace(/^\s+/, "") : cur.text;
    const box = cur.el.querySelector(".text");
    if (cur.seed) { box.innerHTML = `<span class="seed">${esc(cur.seed)}</span>`; const s = document.createElement("span"); render(s, out); box.appendChild(s); }
    else render(box, out);
    const c = document.createElement("span");
    c.className = "caret";
    box.appendChild(c);
    down();
  }

  function done(ev) {
    busy = false;
    Land.busy(false);
    if (cur) {
      const box = cur.el.querySelector(".text");
      box.querySelector(".caret")?.remove();
      cur.el.classList.remove("thinking");
      cur.text = cur.text.replace(/\s+$/, "");
      if (!cur.text && !cur.seed) box.innerHTML = `<span class="seed">(no answer: the model handed the turn straight back)</span>`;
      if (ev) {
        const meta = cur.el.querySelector(".meta");
        const parts = [`<b>${ev.words}</b> tokens`];
        if (ev.speed) parts.push(`<b>${ev.speed.toFixed(1)}</b> tok/s`);
        if (ev.read) parts.push(`read ${ev.read} at ${Math.round(ev.readSpeed)}/s`);
        parts.push(`memory ${Math.round(100 * ev.used / ev.ctx)}%`);
        meta.innerHTML = parts.join(" · ");
        if (ev.words >= 8 && ev.speed && info) store.set("speed." + info.file, ev.speed);
      }
    }
    cur = null;
    paint();
    down();
  }

  function note(t, err) {
    const n = document.createElement("div");
    n.className = "note" + (err ? " err" : "");
    n.textContent = t;
    thread.appendChild(n);
    down(true);
  }

  // ------------------------------------------------------------------ events from the phone
  window.oazy = {
    event(e) {
      switch (e.type) {
        case "ready":
          info = e;
          loadingFile = null;
          const m = models.find((x) => x.file === e.file);
          if (m) info.label = m.label || m.name;
          paint();
          refreshModels();
          break;
        case "tok": piece(e.text); break;
        case "end": done(e); break;
        case "note": note(e.text); break;
        case "selftest": {
          const out = $("#checkOut");
          const corr = +((/correlation ([\d.]+)/.exec(e.text) || [])[1] || 0);
          const good = /tokenizer (\d+)\/\1\b/.test(e.text) && corr >= 0.98;
          out.textContent = (good ? "Passed. " : "Something differs. ") + e.text;
          out.className = "hint " + (good ? "ok" : "bad");
          $("#checkBtn").disabled = false;
          break;
        }
        case "error":
          if (busy) done(null);
          note(e.text, true);
          loadingFile = null;
          paint();
          break;
        case "exit":
          if (busy) done(null);
          if (info || loadingFile) note(`The engine stopped (code ${e.code}). This phone may not have enough free memory for this model; try a 4-bit copy or close other apps.`, true);
          info = null;
          loadingFile = null;
          paint();
          break;
        case "transfer": transfer(e); break;
        case "link":
          openSheet("models");
          $("#url").value = e.url;
          toast("Link received: tap Download");
          break;
      }
    },
    back() {
      if (!$("#veil").hidden) { closeSheets(); return true; }
      return false;
    },
    visible(v) { Land.visible(v); },
  };

  function transfer(e) {
    const box = $("#transfer");
    if (e.state === "progress") {
      box.hidden = false;
      const pct = e.total > 0 ? e.done / e.total : 0;
      $("#tName").textContent = `${e.what === "import" ? "Copying" : "Downloading"} ${e.file}`;
      $("#tPct").textContent = e.total > 0 ? `${fmtB(e.done)} of ${fmtB(e.total)}` : fmtB(e.done);
      $("#tBar").style.width = (100 * pct).toFixed(1) + "%";
      $("#tDot").style.left = (100 * pct).toFixed(1) + "%";
    } else {
      box.hidden = true;
      if (e.state === "done") {
        toast(`${e.file} is on the phone`);
        refreshModels();
        if (!info) open(e.file);
      } else if (e.state === "error") toast(e.message, 6000);
      else if (e.state === "cancelled") toast("Stopped");
    }
  }

  // ------------------------------------------------------------------ sheets
  function openSheet(id) {
    closeSheets(true);
    $("#veil").hidden = false;
    $("#" + id).hidden = false;
    if (id === "models") refreshModels();
    if (id === "settings") placeSlide();
    Land.kick();
  }
  function closeSheets(quiet) {
    $("#veil").hidden = true;
    $("#models").hidden = true;
    $("#settings").hidden = true;
    if (!quiet) Land.kick();
  }
  $("#veil").onclick = () => closeSheets();
  $("#modelPill").onclick = () => openSheet("models");
  $("#settingsBtn").onclick = () => openSheet("settings");
  document.querySelectorAll("[data-act=pick]").forEach((b) => (b.onclick = () => N.pickModel()));
  document.querySelectorAll("[data-act=models]").forEach((b) => (b.onclick = () => { openSheet("models"); setTimeout(() => $("#url").focus(), 300); }));
  $("#dl").onclick = () => {
    const u = $("#url").value.trim();
    if (!/^https?:\/\//.test(u)) { toast("Paste the link to a .oazy file"); return; }
    store.set("url", u);
    N.download(u);
    transfer({ state: "progress", what: "download", file: u.split("/").pop(), done: 0, total: 0 });
  };
  $("#url").value = store.get("url", "");
  $("#tCancel").onclick = () => N.cancelTransfer();
  $("#newChat").onclick = () => {
    if (busy) N.stop();
    N.reset();
    thread.innerHTML = "";
    started = false;
    PICKS.on = false;                                    // fresh suggestions too
    closeSheets();
    paint();
    $("#scroll").scrollTop = 0;
  };
  $("#checkBtn").onclick = () => {
    if (!info) { toast("Open a model first"); return; }
    $("#checkBtn").disabled = true;
    $("#checkOut").className = "hint";
    $("#checkOut").textContent = "Checking… (this also starts a new conversation)";
    thread.innerHTML = "";
    started = false;
    paint();
    N.selftest();
  };

  // sliders, in two groups: the everyday ones, and "More options"
  const shows = [];
  for (const s of SETTINGS) {
    const w = document.createElement("div");
    w.className = "sl";
    w.innerHTML = `<label><span>${s.label}</span><output></output></label><input type="range" min="${s.min}" max="${s.max}" step="${s.step}" aria-label="${s.label}"><small>${s.help}</small>`;
    const r = w.querySelector("input"), o = w.querySelector("output");
    const show = () => { r.value = S[s.k]; o.textContent = s.fmt(+r.value); r.style.setProperty("--p", (100 * (r.value - s.min) / (s.max - s.min)) + "%"); };
    show();
    shows.push(show);
    r.oninput = () => { S[s.k] = +r.value; store.set("set." + s.k, S[s.k]); show(); };
    $(s.grp === "main" ? "#setMain" : "#setMore").appendChild(w);
  }
  $("#resetSet").onclick = () => {
    for (const s of SETTINGS) { S[s.k] = s.def; store.set("set." + s.k, s.def); }
    shows.forEach((f) => f());
    toast("Back to the defaults");
  };

  // composer
  const inp = $("#input");
  function grow() { inp.style.height = "auto"; inp.style.height = Math.min(inp.scrollHeight, 132) + "px"; }
  inp.addEventListener("input", grow);
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && mode === "chat") { e.preventDefault(); send(); } });
  $("#send").onclick = send;
  function placeSlide() {        // the toggle lives in the Settings sheet: it can only be measured while that is open
    const on = document.querySelector(".modes button.on"), slide = document.querySelector(".modes .slide");
    if (!on.offsetWidth) return;
    if (!slide.style.width) { slide.style.transition = "none"; requestAnimationFrame(() => (slide.style.transition = "")); }
    slide.style.width = on.offsetWidth + "px";
    slide.style.transform = `translateX(${on.offsetLeft - 4}px)`;
  }
  function setMode(m) {
    mode = m;
    store.set("mode", m);
    document.querySelectorAll(".modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
    placeSlide();
    inp.placeholder = m === "chat" ? "Ask anything" : "Start a text and the model carries on";
    inp.setAttribute("enterkeyhint", m === "chat" ? "send" : "enter");
    chips();
  }
  document.querySelectorAll(".modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));

  // start: fonts first (the sliding tab measures text), then the last model used
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => setMode(mode));
  refreshModels();
  const last = store.get("last", null);
  if (last && models.some((m) => m.file === last)) open(last);
  else if (models.length) open(models[0].file);
  paint();

  // ------------------------------------------------------------------ the stand-in (desktop preview only)
  function mock() {
    const M = [
      { file: "Coral_1B-q8.oazy", name: "Coral_1B", label: "Coral_1B", kind: "coral", quant: "q8", bytes: 1172000000, params: 1036800000, active: 211000000,
        tokens_trained: 10000000000, layers: "AAAAAAAAAAAAAAAAAAAA", n_expert: 32, top_k: 2, accent: "#ff8a78" },
      { file: "Bee_v2_0.2B-q8.oazy", name: "Bee_v2_0.2B", label: "Bee_v2_0.2B", kind: "bee2", quant: "q8", bytes: 228000000, params: 203300000, active: 203300000,
        tokens_trained: 2500000000, layers: "MMMAMMMAMMMAMMMAMMMAMMMA", accent: "#ffb347" },
    ];
    const q = new URLSearchParams(location.search);
    const have = q.has("empty") ? [] : M;
    let timer = null;
    const ev = (o) => setTimeout(() => window.oazy.event(o), 0);
    const say = "A black hole is a place in space where gravity pulls so hard that nothing, not even light, can get out. It forms when a very big star runs out of fuel and collapses into a tiny point.\n\nAround it is a border called the event horizon: once something crosses it, it can never come back.";
    return {
      device: () => JSON.stringify({ ram: 12e9, free: 5.2e9, cores: 8, fast: 4, model: "Preview phone", android: "15", dir: "/storage/emulated/0/Android/data/ai.oazy.phone/files/models" }),
      models: () => JSON.stringify(have.map((m) => ({ ...m, loaded: false }))),
      load: (f) => setTimeout(() => ev({ type: "ready", file: f, ...have.find((m) => m.file === f), kernel: "ARM sdot", threads: 4, load: "0.04", layers: 20 }), 300),
      ask: (mode, text) => {
        const words = say.split(/(?=[ \n])/);
        let i = 0;
        timer = setInterval(() => {
          if (i < words.length) ev({ type: "tok", text: words[i++] });
          else { clearInterval(timer); ev({ type: "end", words: 61, speed: 17.8, read: 14, readSpeed: 96, used: 92, ctx: 2048 }); }
        }, 55);
        return true;
      },
      stop: () => { clearInterval(timer); ev({ type: "end", words: 12, speed: 17.1, read: 14, readSpeed: 96, used: 40, ctx: 2048 }); },
      reset() {}, unload() {}, deleteModel: () => true, pickModel() {}, download() {}, cancelTransfer() {},
      selftest: () => setTimeout(() => ev({ type: "selftest", text: "tokenizer 15/15; scores vs PyTorch: correlation 0.9991, top word same, top-10 shared 10/10, max diff 0.081" }), 600),
    };
  }
})();
