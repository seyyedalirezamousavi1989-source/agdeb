// Setup Center («مرکز راه‌اندازی»), Phase 26.
//  * The entry is a button in the right-hand sidebar (#setupChip). It blinks a few times when the app starts and
//    whenever its state changes: green = everything required is installed, orange = something is missing,
//    blue = a download/installation is running.
//  * Clicking it swaps the main area (conversation + composer) for a full-page setup view with a back button.
//  * All styling lives in style.css: the server CSP is `style-src 'self'`, so a <style> element injected from
//    JavaScript is blocked (that was why the old panel looked broken). Only CSSOM calls (el.style.x = …) are used here.
(() => {
  const chip = document.getElementById("setupChip");
  const mainPanel = document.querySelector(".main-panel");
  if (!chip || !mainPanel || document.getElementById("setupView")) return;
  const navSub = document.getElementById("setupNavSub");

  const view = document.createElement("section");
  view.className = "setup-view";
  view.id = "setupView";
  view.hidden = true;
  view.setAttribute("aria-label", "مرکز راه‌اندازی");
  const inner = document.createElement("div");
  inner.className = "sv-inner";
  view.appendChild(inner);
  mainPanel.appendChild(view);

  const state = { status: null, rec: null, jobs: [], loadedAt: 0, timer: null, tab: "req", group: "text", drafts: {}, focusKey: null, failed: false, checking: false };

  // ---------------------------------------------------------------------------------------------- helpers
  const call = async (method, url, body) => {
    const options = { method };
    if (body !== undefined) { options.headers = { "Content-Type": "application/json" }; options.body = JSON.stringify(body); }
    const response = await (window.apiFetch || fetch)(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `خطا (${response.status})`);
    return data;
  };
  const el = (tag, props = {}, ...kids) => {
    const node = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
      if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null) node[k] = v;
    });
    kids.flat().forEach((kid) => { if (kid !== null && kid !== undefined && kid !== false) node.append(kid instanceof Node ? kid : document.createTextNode(String(kid))); });
    return node;
  };
  const ICONS = {
    check: '<path d="m5 12.5 4.2 4.2L19 7"/>',
    alert: '<path d="M12 8.5v4.5M12 16.5v.01"/><path d="M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>',
    minus: '<path d="M6.5 12h11"/>',
    download: '<path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19.5h14"/>',
    play: '<path d="M8 5.5v13l10-6.5-10-6.5Z"/>',
    folder: '<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2.2h7a2 2 0 0 1 2 2v7.8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2Z"/>',
    refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4.5V11h-6.5"/>',
    back: '<path d="M4 12h15M13 6l6 6-6 6"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    bolt: '<path d="M13 2.5 4.5 13.5h6.5L10 21.5 19.5 10h-6.5Z"/>',
    cpu: '<rect x="6.5" y="6.5" width="11" height="11" rx="2"/><path d="M9.5 3v3.5M14.5 3v3.5M9.5 17.5V21M14.5 17.5V21M3 9.5h3.5M3 14.5h3.5M17.5 9.5H21M17.5 14.5H21"/>',
    box: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9Z"/><path d="m4.5 7.8 7.5 4.3 7.5-4.3M12 12.1V21"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3"/>',
    wave: '<path d="M4 12h2M8 8v8M12 5v14M16 8v8M20 12h-2"/>',
    win: '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M3.5 9h17"/>',
    code: '<path d="m8.5 8-4 4 4 4M15.5 8l4 4-4 4M13.5 5.5l-3 13"/>',
    list: '<path d="M8 6.5h11M8 12h11M8 17.5h11M4.5 6.5h.01M4.5 12h.01M4.5 17.5h.01"/>',
    star: '<path d="m12 3.5 2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.8l-5.2 2.8 1-5.8L3.5 9.7l5.9-.9Z"/>',
    info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.01"/>',
  };
  const ico = (name, extra = "") => { const s = el("span", { class: "sv-i " + extra }); s.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`; return s; };
  const COMP_ICON = { ollama: "cpu", audio_env: "mic", edge_tts: "wave", parsvoice: "wave", vc_redist: "win", webview2: "win" };
  const toast = (text) => (typeof showToast === "function" ? showToast(text) : alert(text));
  const gb = (bytes) => (bytes ? (bytes / 1024 ** 3).toFixed(2) + " GB" : "");
  const fa = (n) => Number(n).toLocaleString("fa-IR");
  const LIVE = ["queued", "running", "waiting"];
  const isLive = (j) => LIVE.includes(j.state);

  // ---------------------------------------------------------------------------------------------- state
  const comps = () => state.status?.components || [];
  const essential = () => comps().filter((c) => !c.optional);
  const missingEssential = () => essential().filter((c) => !c.ok);
  const ollamaMissing = () => comps().some((c) => c.id === "ollama" && !c.ok);
  const active = () => state.jobs.some(isLive);
  const overall = () => (!state.status ? "loading" : active() ? "busy" : missingEssential().length ? "need" : "ok");
  const isOpen = () => !view.hidden;

  async function refresh(full = true) {
    try {
      const [status, jobs] = await Promise.all([call("GET", "/api/setup/status"), call("GET", "/api/setup/jobs")]);
      state.status = status; state.jobs = jobs.jobs || []; state.loadedAt = Date.now(); state.failed = false;
      if (full && !state.rec) call("GET", "/api/setup/recommend").then((r) => { state.rec = r; if (isOpen()) paint(); }).catch(() => {});
    } catch (error) { state.failed = !state.status; }
    paintNav();
    if (isOpen()) paint();
    schedule();
  }
  function schedule() {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => refresh(false), active() ? 1500 : isOpen() ? 6000 : 45000);
  }

  // ---------------------------------------------------------------------------------------------- sidebar button
  let blinkTimer = null;
  function blink() {
    chip.classList.remove("blink");
    void chip.offsetWidth;                       // restart the animation
    chip.classList.add("blink");
    clearTimeout(blinkTimer);
    blinkTimer = setTimeout(() => chip.classList.remove("blink"), 3900);
  }
  function paintNav() {
    const s = overall();
    const previous = chip.dataset.state;
    chip.dataset.state = s;
    navSub.textContent = s === "ok" ? "همه‌چیز آماده است" : s === "need" ? `${fa(missingEssential().length)} مورد نیاز به نصب` : s === "busy" ? "در حال دانلود و نصب…" : state.failed ? "بررسی نشد" : "در حال بررسی…";
    if (s !== "loading" && s !== previous) blink();     // first result after start-up, and every later change
  }

  // ---------------------------------------------------------------------------------------------- actions
  async function act(label, work) {
    try { await work(); await refresh(false); }
    catch (error) { toast(`${label}: ${error.message}`); }
  }
  const install = (kind) => act("شروع نصب", () => call("POST", "/api/setup/install", { kind }));
  const goPaths = (key) => { state.tab = "paths"; state.focusKey = key; paint(true); };

  // ---------------------------------------------------------------------------------------------- pieces
  function hero() {
    const s = overall();
    const ess = essential();
    const ready = ess.filter((c) => c.ok).length;
    const pct = state.status ? (ess.length ? Math.round((ready / ess.length) * 100) : 100) : 0;
    const copy = {
      ok: ["همه‌چیز آماده است", "اجزای ضروری نصب و فعال‌اند. هر زمان خواستید مدل یا مسیر تازه‌ای اضافه کنید."],
      need: [`${fa(missingEssential().length)} مورد برای کامل‌شدن راه‌اندازی مانده است`, "از تب «نصب و پیش‌نیازها» هر مورد را با یک کلیک نصب کنید؛ برای شروع فقط Ollama و یک مدل کافی است."],
      busy: ["در حال دانلود و نصب…", "می‌توانید به گفت‌وگو برگردید؛ کار در پس‌زمینه ادامه دارد و با قطع اینترنت یا بستن برنامه از همان‌جا دوباره ادامه می‌یابد."],
      loading: [state.failed ? "بررسی سیستم انجام نشد" : "در حال بررسی سیستم…", state.failed ? "ارتباط با برنامه برقرار نشد؛ «بررسی دوباره» را بزنید." : "چند ثانیه صبر کنید."],
    }[s];
    const ring = el("div", { class: "sv-ring" }, el("span", {}, state.status ? `${fa(ready)}/${fa(ess.length)}` : "…"));
    ring.style.setProperty("--p", String(pct));
    const m = state.rec?.machine;
    const pills = m ? el("div", { class: "sv-pills" }, el("span", { class: "sv-pill" }, ico("cpu"), m.mode === "gpu" ? `${m.gpu_name} · ${m.vram_gb} GB VRAM` : "CPU only"), el("span", { class: "sv-pill" }, `RAM ${m.ram_gb} GB`)) : null;
    const refreshBtn = el("button", { class: "sv-btn", type: "button", onclick: async (e) => {
      const b = e.currentTarget; b.classList.add("spin"); b.disabled = true; state.rec = null;
      await refresh(true); b.classList.remove("spin"); b.disabled = false;
    } }, ico("refresh"), "بررسی دوباره");
    return el("div", { class: "sv-hero " + s }, ring, el("div", { class: "sv-hero-text" }, el("h3", {}, copy[0]), el("p", {}, copy[1]), pills), el("div", { class: "sv-hero-actions" }, refreshBtn));
  }

  function jobCard(j) {
    const names = { ollama: "نصب Ollama", audio_env: "نصب موتور صدا", vc_redist: "نصب Visual C++ Runtime", webview2: "نصب WebView2" };
    const title = names[j.kind] || (j.kind.startsWith("model:") ? "دانلود مدل " + j.kind.slice(6) : j.kind.startsWith("whisper:") ? "دانلود مدل گفتار " + j.kind.slice(8) : j.kind);
    const stateText = { queued: "در صف", running: "در حال انجام", waiting: "منتظر اتصال / تلاش دوباره", done: "انجام شد ✅", error: "ناموفق ❌", cancelled: "لغو شد" }[j.state] || j.state;
    const live = isLive(j);
    const info = j.total ? ` — ${gb(j.bytes)} از ${gb(j.total)}` : "";
    const bar = el("div", { class: "sv-bar" }, el("i"));
    bar.firstChild.style.width = `${Math.max(3, Math.min(100, j.progress || 0))}%`;
    const button = live ? el("button", { class: "sv-btn sm danger", type: "button", onclick: () => act("لغو", () => call("POST", "/api/setup/cancel", { id: j.id })) }, ico("x"), "لغو")
      : j.state === "error" || j.state === "cancelled" ? el("button", { class: "sv-btn sm primary", type: "button", onclick: () => install(j.kind) }, ico("refresh"), "ادامه / تلاش دوباره") : null;
    return el("div", { class: "sv-job " + (j.state === "error" ? "error" : j.state === "done" ? "done" : "") },
      el("div", { class: "sv-job-head" }, el("span", {}, `${title}: ${stateText}${info}`), button),
      j.message ? el("small", {}, j.message) : null, live ? bar : null);
  }

  function steps() {
    const by = (id) => comps().find((c) => c.id === id);
    const hasModel = !!state.rec?.groups?.text?.some((m) => m.installed);
    const list = [
      { t: "Ollama", d: "موتور اجرای مدل‌ها", done: !!by("ollama")?.ok },
      { t: "یک مدل متنی", d: "از تب «مدل‌ها»", done: hasModel },
      { t: "موتور صدا", d: "اختیاری — میکروفون و خواندن پاسخ", done: !!by("audio_env")?.ok },
      { t: "اجزای مایکروسافت", d: "فقط در صورت بروز خطا", done: !!by("vc_redist")?.ok && !!by("webview2")?.ok },
    ];
    return el("div", { class: "sv-steps" }, list.map((s, i) => el("div", { class: "sv-step" + (s.done ? " done" : "") }, el("span", { class: "sv-step-n" }, s.done ? ico("check") : fa(i + 1)), el("div", {}, el("b", {}, s.t), el("small", {}, s.d)))));
  }

  function componentCard(c) {
    const opt = !!c.optional;
    const cls = c.ok ? "ok" : opt ? "opt" : "bad";
    const busy = state.jobs.find((j) => c.install && j.kind === c.install && isLive(j));
    const actions = el("div", { class: "sv-actions" });
    if (c.install) actions.append(el("button", { class: "sv-btn primary", type: "button", disabled: !!busy, onclick: () => install(c.install) }, ico("download"), busy ? "در حال نصب…" : "نصب خودکار"));
    if (c.start) actions.append(el("button", { class: "sv-btn primary", type: "button", onclick: () => act("روشن‌کردن", () => call("POST", "/api/setup/start-ollama", {})) }, ico("play"), "روشن کن"));
    if (c.path_key) actions.append(el("button", { class: "sv-btn", type: "button", onclick: () => goPaths(c.path_key) }, ico("folder"), "قبلاً نصب دارم…"));
    return el("article", { class: "sv-card " + (c.ok || opt ? "" : "bad") },
      el("div", { class: "sv-card-head" }, el("span", { class: "sv-status " + cls }, ico(c.ok ? "check" : opt ? "minus" : "alert")),
        el("div", { class: "sv-card-title" }, el("b", {}, c.label), el("span", { class: "sv-tag " + cls }, c.ok ? "آماده" : opt ? "اختیاری" : "نیاز به نصب"))),
      el("p", { class: "sv-detail" }, c.detail || ""),
      c.needed_for ? el("p", { class: "sv-for" }, ico(COMP_ICON[c.id] || "bolt"), "برای: " + c.needed_for) : null,
      c.hint ? el("p", { class: "sv-hint" }, ico("info"), c.hint) : null,
      c.why ? el("p", { class: "sv-hint" }, ico("info"), c.why) : null,
      actions.childNodes.length ? actions : null);
  }

  function modelCard(m) {
    const job = state.jobs.find((j) => j.kind === "model:" + m.id && isLive(j));
    const fitLabel = { good: "روی GPU", slow: "کندتر (RAM)", too_big: "برای این سیستم بزرگ است" }[m.fit];
    const action = m.installed ? el("span", { class: "sv-ok-text" }, ico("check"), "نصب است")
      : job ? el("span", { class: "sv-size" }, "در حال دانلود…")
        : el("button", { class: "sv-btn sm" + (m.fit === "too_big" ? "" : " primary"), type: "button", onclick: () => {
          if (m.fit === "too_big" && !confirm("این مدل از حافظهٔ سیستم شما بزرگ‌تر است و خیلی کند یا ناپایدار می‌شود. ادامه می‌دهید؟")) return;
          install("model:" + m.id);
        } }, ico("download"), "دانلود");
    return el("article", { class: "sv-card model" + (m.installed ? " installed" : "") },
      el("div", { class: "sv-model-head" }, el("b", {}, m.name), el("span", { class: "sv-badge " + m.fit }, fitLabel), m.recommended ? el("span", { class: "sv-badge rec" }, "پیشنهاد") : null),
      el("p", { class: "sv-detail" }, m.desc || ""),
      el("div", { class: "sv-model-foot" }, el("span", { class: "sv-size" }, m.size), action));
  }

  const PATH_META = {
    models_dir: ["پوشهٔ مدل‌های Ollama", "جایی که Ollama مدل‌ها را ذخیره می‌کند و از آن می‌خواند (شامل manifests و blobs).", "D:\\OllamaModels", "folder"],
    ollama_exe: ["مسیر ollama.exe", "اگر Ollama را در مسیر غیرمعمول نصب کرده‌اید، فایل اجرایی آن را اینجا بدهید.", "C:\\Users\\...\\Ollama\\ollama.exe", "cpu"],
    audio_python: ["مسیر python.exe محیط صدا", "پایتونی که faster-whisper و edge-tts در آن نصب است (برای میکروفون و خواندن پاسخ).", "C:\\...\\python.exe", "mic"],
    parsvoice_dir: ["پوشهٔ مدل ParsVoice", "پوشهٔ ParsVoice-XTTS برای خواندن پاسخ بدون اینترنت.", "D:\\models\\ParsVoice-XTTS", "wave"],
  };
  function pathCard(key) {
    const [label, help, placeholder, icon] = PATH_META[key];
    const custom = (state.status?.paths || {})[key] || "";
    const current = key === "models_dir" ? (custom || state.status?.models_dir || "") : custom;
    const input = el("input", { class: "sv-input", type: "text", spellcheck: false, placeholder, value: state.drafts[key] ?? current });
    input.dataset.key = key;
    input.addEventListener("input", () => { state.drafts[key] = input.value; });
    const save = () => act("ذخیرهٔ مسیر", async () => { await call("POST", "/api/setup/path", { key, value: input.value }); delete state.drafts[key]; state.rec = null; toast("مسیر ذخیره شد."); await refresh(true); });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") save(); });
    return el("article", { class: "sv-card sv-path" },
      el("div", { class: "sv-path-top" }, el("span", { class: "sv-status" }, ico(icon)), el("div", {}, el("b", {}, label), el("small", {}, help))),
      el("div", { class: "sv-path-row" }, input,
        el("button", { class: "sv-btn primary", type: "button", onclick: save }, "ذخیره"),
        el("button", { class: "sv-btn", type: "button", onclick: () => act("بازنشانی", async () => { await call("POST", "/api/setup/path", { key, value: "" }); delete state.drafts[key]; state.rec = null; await refresh(true); }) }, "پیش‌فرض")),
      el("p", { class: "sv-path-cur" }, custom ? "مسیر سفارشی: " + custom : current ? "مسیر فعلی (پیش‌فرض): " + current : "پیش‌فرض برنامه"));
  }

  // ---------------------------------------------------------------------------------------------- tabs
  function tabRequirements() {
    const box = el("div", {});
    box.append(steps(), el("div", { class: "sv-note" }, el("b", {}, "نصب‌کنندهٔ برنامه فقط خودِ LocalChat را نصب می‌کند "), "و چیزی دانلود نمی‌کند؛ بقیهٔ اجزا از همین‌جا با چند کلیک نصب می‌شوند. اینترنت لازم است، ولی اگر قطع شود یا برنامه بسته شود دانلود از همان‌جا ادامه می‌یابد. اگر چیزی را از قبل نصب دارید، «قبلاً نصب دارم» را بزنید و مسیرش را بدهید تا دوباره دانلود نشود."));
    if (!state.status) { box.append(el("div", { class: "sv-skeleton" }, "در حال بررسی…")); return box; }
    const need = comps().filter((c) => !c.optional), opt = comps().filter((c) => c.optional);
    box.append(el("div", { class: "sv-section" }, el("h4", {}, "اجزای ضروری"), el("small", {}, "برای کار کردن گفت‌وگو و صدا")), el("div", { class: "sv-grid" }, need.map(componentCard)));
    box.append(el("div", { class: "sv-section" }, el("h4", {}, "اختیاری"), el("small", {}, "فقط در صورت نیاز")), el("div", { class: "sv-grid" }, opt.map(componentCard)));
    return box;
  }

  function tabModels() {
    const box = el("div", {});
    const rec = state.rec;
    if (!rec) { box.append(el("div", { class: "sv-skeleton" }, "در حال تشخیص سخت‌افزار…")); return box; }
    box.append(el("div", { class: "sv-note" }, rec.note));
    if (rec.starter.length) {
      const todo = rec.starter.filter((e) => !e.installed);
      const all = todo.length > 1 ? el("button", { class: "sv-btn primary", type: "button", onclick: () => todo.reduce((p, e) => p.then(() => call("POST", "/api/setup/install", { kind: "model:" + e.id })), Promise.resolve()).then(() => refresh(false)).catch((e) => toast(e.message)) }, ico("download"), `دانلود همهٔ بستهٔ شروع (${fa(todo.length)} مدل)`) : null;
      box.append(el("div", { class: "sv-starter" },
        el("div", { class: "sv-starter-head" }, el("div", {}, el("h4", {}, "بستهٔ شروع پیشنهادی"), el("p", {}, "به ترتیب: متن، بینایی، جست‌وجوی معنایی — برای شروع کافی است.")), all),
        el("div", { class: "sv-grid" }, rec.starter.map(modelCard))));
    }
    const names = { text: "متنی", vision: "بینایی", audio: "صوتی", embedding: "معنایی (Embedding)" };
    const groups = Object.entries(rec.groups);
    if (!rec.groups[state.group]) state.group = groups[0]?.[0] || "text";
    box.append(el("div", { class: "sv-section" }, el("h4", {}, "همهٔ مدل‌ها"), el("small", {}, "بر اساس کارت گرافیک و RAM شما مرتب شده‌اند")),
      el("div", { class: "sv-seg" }, groups.map(([g, list]) => el("button", { type: "button", class: g === state.group ? "active" : "", onclick: () => { state.group = g; paint(true); } }, `${names[g] || g} (${fa(list.length)})`))),
      el("div", { class: "sv-grid" }, (rec.groups[state.group] || []).map(modelCard)));
    const stt = state.status?.stt_model;
    box.append(el("div", { class: "sv-section" }, el("h4", {}, "گفتار (میکروفون ← متن)")),
      el("div", { class: "sv-grid" }, el("article", { class: "sv-card" },
        el("div", { class: "sv-card-head" }, el("span", { class: "sv-status opt" }, ico("mic")), el("div", { class: "sv-card-title" }, el("b", {}, `مدل گفتار پیشنهادی: ${rec.stt_model}`))),
        el("p", { class: "sv-detail" }, `فعلی: ${stt || "—"} · پس از نصب موتور صدا دانلود می‌شود`),
        el("div", { class: "sv-actions" }, el("button", { class: "sv-btn primary", type: "button", onclick: () => act("دانلود مدل گفتار", () => call("POST", "/api/setup/stt-model", { size: rec.stt_model })) }, ico("download"), "دانلود")))));
    return box;
  }

  function tabPaths() {
    return el("div", {}, el("div", { class: "sv-note" }, "برای مدل‌ها یا ابزارهایی که از قبل روی رایانه دارید، مسیرشان را اینجا بدهید تا دوباره دانلود نشوند. خالی‌گذاشتن یا «پیش‌فرض» به حالت خودکار برمی‌گرداند."),
      el("div", { class: "sv-paths-grid" }, ["models_dir", "ollama_exe", "audio_python", "parsvoice_dir"].map(pathCard)));
  }

  // ---------------------------------------------------------------------------------------------- page
  function paint(force = false) {
    const focused = document.activeElement;
    if (!force && focused && view.contains(focused) && focused.tagName === "INPUT") return;   // never wipe what the user is typing
    const scroll = view.scrollTop;
    const bad = missingEssential().length;
    const tabs = [["req", "نصب و پیش‌نیازها", "list", bad], ["models", "مدل‌ها", "box", 0], ["paths", "مسیرها", "folder", 0]];
    const jobs = state.jobs.filter((j) => isLive(j) || Date.now() / 1000 - (j.updated || 0) < 600).slice(0, 6);
    const body = state.tab === "models" ? tabModels() : state.tab === "paths" ? tabPaths() : tabRequirements();
    inner.replaceChildren(
      el("div", { class: "sv-head" },
        el("button", { class: "sv-back", type: "button", onclick: closeView, title: "بازگشت (Esc)" }, ico("back"), "بازگشت به گفت‌وگو"),
        el("div", { class: "sv-title" }, el("h2", {}, "مرکز راه‌اندازی"), el("p", {}, "نصب پیش‌نیازها، دانلود مدل‌ها و تنظیم مسیرها در یک‌جا"))),
      hero(),
      jobs.length ? el("div", { class: "sv-section" }, el("h4", {}, "دانلودها و نصب‌ها")) : null,
      jobs.length ? el("div", {}, jobs.map(jobCard)) : null,
      el("div", { class: "sv-tabs", role: "tablist" }, tabs.map(([id, label, icon, badge]) =>
        el("button", { type: "button", role: "tab", class: "sv-tab" + (state.tab === id ? " active" : ""), onclick: () => { state.tab = id; paint(true); } }, ico(icon), label, badge ? el("span", { class: "sv-count" }, fa(badge)) : null))),
      body);
    view.scrollTop = scroll;
    if (state.focusKey) {
      const input = inner.querySelector(`input[data-key="${state.focusKey}"]`);
      state.focusKey = null;
      if (input) { input.scrollIntoView({ block: "center", behavior: "smooth" }); input.focus({ preventScroll: true }); }
    }
  }

  function openView(tab) {
    if (tab) state.tab = tab;
    mainPanel.classList.add("setup-open");
    view.hidden = false;
    chip.classList.add("active"); chip.setAttribute("aria-pressed", "true");
    const backdrop = document.getElementById("sidebarBackdrop");
    if (backdrop?.classList.contains("visible")) backdrop.click();          // phones: close the slide-in menu
    view.classList.add("entering");
    setTimeout(() => view.classList.remove("entering"), 400);
    view.scrollTop = 0;
    paint(true);
    refresh(true);
  }
  function closeView() {
    mainPanel.classList.remove("setup-open");
    view.hidden = true;
    chip.classList.remove("active"); chip.setAttribute("aria-pressed", "false");
    schedule();
  }
  chip.addEventListener("click", () => (isOpen() ? closeView() : openView()));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && isOpen() && !document.querySelector(".memory-overlay:not([hidden]), .capture-overlay:not([hidden])")) closeView(); });
  // choosing a chat (or «گفت‌وگوی جدید») from the sidebar leaves the setup page
  document.addEventListener("click", (e) => { if (isOpen() && e.target.closest && e.target.closest("#chatList, #newChat, .sidebar .brand")) closeView(); }, true);

  // ---------------------------------------------------------------------------------------------- warn only when a request needs something missing
  let banner = null;
  function showBanner(text) {
    banner?.remove();
    banner = el("div", { class: "setup-banner" }, el("span", {}, text), el("button", { type: "button", onclick: () => { banner.remove(); banner = null; openView("req"); } }, "باز کردن مرکز راه‌اندازی"),
      el("button", { type: "button", onclick: () => { banner.remove(); banner = null; } }, "✕"));
    document.body.appendChild(banner);
    setTimeout(() => { banner?.remove(); banner = null; }, 15000);
  }
  const NEEDS = [[/\/api\/chat(\?|$)/, "ollama", "Ollama نصب یا روشن نیست؛ بدون آن مدل محلی جواب نمی‌دهد."],
    [/\/api\/models\/pull/, "ollama", "برای دانلود مدل، Ollama باید نصب و روشن باشد."],
    [/\/api\/voice\/transcribe/, "audio_env", "موتور تشخیص گفتار نصب نیست."],
    [/\/api\/voice\/synthesize/, "edge_tts", "موتور صدا نصب نیست."]];
  async function checkNeed(url) {
    const hit = NEEDS.find(([re]) => re.test(url));
    if (!hit) return;
    if (!state.status || Date.now() - state.loadedAt > 20000) await refresh(false);
    const item = comps().find((c) => c.id === hit[1]);
    if (item && !item.ok) showBanner(hit[2]);
  }
  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    try { if (((init && init.method) || "GET").toUpperCase() === "POST") checkNeed(typeof input === "string" ? input : input.url || ""); } catch (_) {}
    return nativeFetch(input, init);
  };

  // first run: if the essential engine is missing, show the guide by itself once
  refresh(true).then(() => {
    if (ollamaMissing() && !localStorage.getItem("setupGuideShown")) {
      try { localStorage.setItem("setupGuideShown", "1"); } catch (_) {}
      openView("req");
    }
  });
})();
