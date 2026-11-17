// Long-running tasks panel: upload a big file, watch chunk progress, pause / resume / cancel, download DOCX/PDF/TXT/MD.
(() => {
  const overlay = document.getElementById("tasksOverlay");
  if (!overlay) return;
  const list = document.getElementById("tasksList");
  const statusLine = document.getElementById("taskStatus");
  const fileInput = document.getElementById("taskFile");
  const startButton = document.getElementById("taskStart");
  let timer = null;

  const STATE_LABEL = {
    queued: "در صف", preparing: "آماده‌سازی", running: "در حال اجرا", paused: "متوقف‌شده", cancelled: "لغوشده",
    completed: "تمام‌شده", partial: "ناقص (بخش‌هایی ناموفق)", failed: "ناموفق", interrupted: "قطع‌شده؛ قابل ادامه",
  };
  const ACTIVE = new Set(["preparing", "running"]);

  function say(text, bad = false) {
    statusLine.textContent = text || "";
    statusLine.style.color = bad ? "#b3261e" : "";
  }
  function fmtTime(seconds) {
    if (seconds == null) return "—";
    const s = Math.round(seconds), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h} ساعت و ${m} دقیقه` : m ? `${m} دقیقه و ${s % 60} ثانیه` : `${s} ثانیه`;
  }
  async function call(method, url, body) {
    const options = { method };
    if (body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    const response = await apiFetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `خطا (${response.status})`);
    return data;
  }
  function button(label, handler, primary = false) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "settings-button" + (primary ? " primary" : "");
    el.textContent = label;
    el.addEventListener("click", handler);
    return el;
  }
  async function act(id, action, payload) {
    if (action === "pause" || action === "cancel") say(action === "pause" ? "درخواست توقف موقت ارسال شد؛ منتظر پایان درخواست جاری مدل…" : "درخواست لغو ارسال شد؛ منتظر پایان درخواست جاری مدل…");
    else say("");
    try { await call("POST", `/api/tasks/${id}/${action}`, payload || {}); } catch (error) { say(error.message, true); }
    refresh();
    if (window.refreshJobBanner) window.refreshJobBanner();
  }
  function notify(text, bad = false) {
    say(text, bad);
    if (typeof window.showToast === "function") window.showToast(text);
    else if (bad) alert(text);
  }
  async function reveal(path) {
    try { await call("POST", "/api/tasks/x/reveal", { path }); } catch (error) { notify(error.message, true); }
  }
  // The app runs in a desktop window that cannot save browser downloads, so the SERVER writes the file into the output folder.
  async function download(id, format, partial = false) {
    say(`در حال ساخت فایل ${format.toUpperCase()}…`);
    try {
      const saved = await call("POST", `/api/tasks/${id}/save`, { format, partial });
      say("");
      statusLine.replaceChildren();
      const msg = document.createElement("span");
      msg.textContent = `ذخیره شد: ${saved.path} `;
      const open = document.createElement("button");
      open.type = "button"; open.className = "settings-button"; open.textContent = "نمایش در پوشه";
      open.addEventListener("click", () => reveal(saved.path));
      statusLine.append(msg, open);
      statusLine.style.color = "";
      if (typeof window.showToast === "function") window.showToast(`ذخیره شد: ${saved.name}`);
      if (outDirInput) outDirInput.value = saved.folder;
    } catch (error) {
      if (/ناقص|کامل نیست/.test(error.message) && !partial && confirm(`${error.message}\n\nنسخهٔ ناقص (با علامت‌گذاری بخش‌های جاافتاده) ساخته شود؟`)) return download(id, format, true);
      notify(error.message || "ساخت فایل ممکن نشد.", true);
    }
  }
  const outDirInput = document.getElementById("taskOutDir");
  async function loadOutDir() {
    try { const data = await call("GET", "/api/tasks/output-dir"); if (outDirInput) outDirInput.value = data.folder; } catch (error) { /* shown on save */ }
  }
  document.getElementById("taskOutSave")?.addEventListener("click", async () => {
    try { const data = await call("POST", "/api/tasks/output-dir", { folder: outDirInput.value }); outDirInput.value = data.folder; notify(`پوشهٔ خروجی: ${data.folder}`); }
    catch (error) { notify(error.message, true); }
  });
  document.getElementById("taskOutOpen")?.addEventListener("click", () => reveal(""));

  function render(tasks) {
    list.replaceChildren();
    if (!tasks.length) {
      const empty = document.createElement("div");
      empty.className = "task-meta";
      empty.textContent = "هنوز کاری ساخته نشده است.";
      list.appendChild(empty);
      return;
    }
    for (const task of tasks) {
      const item = document.createElement("div");
      item.className = "task-item";
      const head = document.createElement("div");
      head.className = "task-head";
      const title = document.createElement("span");
      title.className = "task-title";
      title.textContent = task.title;
      title.title = task.title;
      const state = document.createElement("span");
      state.className = "task-state " + task.status;
      state.textContent = task.stop_requested ? (task.stop_requested === "cancel" ? "در حال لغو…" : "در حال توقف…") : (STATE_LABEL[task.status] || task.status);
      head.append(title, state);
      const bar = document.createElement("div");
      bar.className = "task-bar";
      const fill = document.createElement("span");
      fill.style.width = `${task.percent}%`;
      bar.appendChild(fill);
      const meta = document.createElement("div");
      meta.className = "task-meta";
      meta.textContent = `${task.done} از ${task.total} بخش (${task.percent}٪) · ` +
        `${task.failed ? task.failed + " ناموفق · " : ""}زمان: ${fmtTime(task.elapsed_seconds)}` +
        (task.eta_seconds != null ? ` · باقی‌مانده: ${fmtTime(task.eta_seconds)}` : "");
      item.append(head, bar, meta);
      for (const chunk of (task.chunks || []).filter((entry) => entry.status === "failed")) {
        const detail = document.createElement("div");
        detail.className = "task-error";
        const recovery = chunk.recovery_completed ? ` · ${chunk.recovery_completed} زیر‌بخش بازیابی‌شده حفظ شده` : "";
        detail.textContent = `بخش ${chunk.index} · ${chunk.attempts} تلاش${recovery}: ${chunk.error || "پردازش ناموفق بود."}`;
        item.appendChild(detail);
      }
      const problem = task.error || (task.last_chunk_error && `بخش ${task.last_chunk_error.idx + 1}: ${task.last_chunk_error.error}`);
      if (problem) {
        const err = document.createElement("div");
        err.className = "task-error";
        err.textContent = problem;
        item.appendChild(err);
      }
      const actions = document.createElement("div");
      actions.className = "task-actions";
      if (ACTIVE.has(task.status)) {
        const pauseBtn = button("توقف موقت", () => act(task.id, "pause")), cancelBtn = button("لغو", () => act(task.id, "cancel"));
        if (task.stop_requested) { pauseBtn.disabled = true; cancelBtn.disabled = task.stop_requested === "cancel"; }
        actions.append(pauseBtn, cancelBtn);
      } else {
        if (task.status !== "completed" && task.resumable) actions.append(button("ادامه", () => act(task.id, "resume"), true));
        if (task.failed) actions.append(button("تلاش دوباره برای ناموفق‌ها", () => act(task.id, "retry")));
        if (task.failed) actions.append(button("بازیابی ناموفق‌ها با بخش‌های کوچک‌تر",
          () => act(task.id, "retry_smaller"), true));
        if (task.done > 0) for (const fmt of ["docx", "pdf", "md", "txt"]) actions.append(button(fmt.toUpperCase(), () => download(task.id, fmt)));
        actions.append(button("حذف", () => { if (confirm("این کار و نتایجش حذف شود؟")) act(task.id, "delete"); }));
      }
      item.appendChild(actions);
      list.appendChild(item);
    }
  }

  async function refresh() {
    try {
      const data = await call("GET", "/api/tasks");
      render(data.tasks || []);
      const busy = (data.tasks || []).some((t) => ACTIVE.has(t.status));
      clearTimeout(timer);
      if (!overlay.hidden) timer = setTimeout(refresh, busy ? 1500 : 6000);
    } catch (error) {
      say(error.message, true);
    }
  }

  startButton.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    const model = document.getElementById("modelSelect").value;
    if (!file) return say("ابتدا یک فایل انتخاب کن.", true);
    if (!model) return say("ابتدا یک مدل از بالای صفحه انتخاب کن.", true);
    startButton.disabled = true;
    say("در حال ارسال و تقسیم فایل…");
    const options = {
      kind: document.getElementById("taskKind").value, model, target: document.getElementById("taskTarget").value,
      style: document.getElementById("taskStyle").value.trim(),
      pages: document.getElementById("taskPages").value.trim(),
    };
    try {
      const response = await apiFetch("/api/tasks", {
        method: "POST", body: file,
        headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name),
                   "X-Task-Options": encodeURIComponent(JSON.stringify(options)) },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `خطا (${response.status})`);
      fileInput.value = "";
      say(`کار ساخته شد: ${data.task.total} بخش.`);
      refresh();
      if (window.refreshJobBanner) window.refreshJobBanner();
    } catch (error) {
      say(error.message, true);
    } finally {
      startButton.disabled = false;
    }
  });

  // ---- live banner above the composer: shows that a long job is really running -------------------------
  const banner = document.getElementById("jobBanner");
  const seenActive = new Map();
  let finished = null;
  let bannerTimer = null;

  function setBusy(info) {
    const before = window.localChatBusy || { busy: false };
    window.localChatBusy = { busy: Boolean(info.busy), reason: info.reason || "", kind: info.kind || "" };
    if (before.busy !== window.localChatBusy.busy) window.dispatchEvent(new Event("localchat:busy"));
  }
  function bannerButton(label, handler) {
    const el = document.createElement("button");
    el.type = "button";
    el.textContent = label;
    el.addEventListener("click", handler);
    return el;
  }
  function openPanel() { document.getElementById("tasksButton").click(); }
  function renderBanner(tasks) {
    if (!banner) return;
    const active = tasks.find((t) => ACTIVE.has(t.status));
    banner.replaceChildren();
    banner.className = "job-banner";
    if (active) {
      seenActive.set(active.id, true);
      finished = null;
      const verb = active.kind === "summarize" ? "خلاصه‌سازی" : "ترجمه";
      const line = document.createElement("div");
      line.className = "job-line";
      const title = document.createElement("span");
      title.className = "job-title";
      title.textContent = `در حال ${verb}: ${active.title}`;
      const actions = document.createElement("span");
      actions.className = "job-actions";
      const stopBtn = bannerButton(active.stop_requested ? "در حال توقف…" : "توقف موقت", () => act(active.id, "pause").then(pollBanner));
      stopBtn.disabled = Boolean(active.stop_requested);
      actions.append(stopBtn, bannerButton("جزئیات", openPanel));
      line.append(title, actions);
      const bar = document.createElement("div");
      bar.className = "job-bar" + (active.done === 0 ? " indeterminate" : "");
      const fill = document.createElement("span");
      fill.style.width = `${active.percent}%`;
      bar.appendChild(fill);
      const meta = document.createElement("div");
      const parts = [`${active.done} از ${active.total} بخش تمام شد (${active.percent}٪)`];
      if (active.current_chunk) parts.push(`بخش ${active.current_chunk} در حال پردازش است (${Math.round(active.current_chunk_seconds || 0)} ثانیه)`);
      else parts.push(active.status === "preparing" ? "در حال آماده‌سازی و تقسیم فایل" : "در حال شروع");
      const fit = active.fit;
      if (fit) {
        parts.push(fit.gpu_percent >= 85 ? `مدل ${fit.gpu_percent}٪ روی GPU است` : `⚠ فقط ${fit.gpu_percent}٪ مدل روی GPU است و بقیه روی CPU/RAM — کند؛ مدل کوچک‌تر (مثل qwen3:8b) پیشنهاد می‌شود`);
      }
      const live = active.live;
      if (live) {
        if (live.phase === "thinking") parts.push(`مدل در حال فکر کردن است (${live.tokens} قطعه)؛ ترجمه هنوز شروع نشده`);
        else if (live.phase === "generating") parts.push(`مدل در حال نوشتن ترجمه است: ${live.tokens} قطعه تولید شد` + (live.tokens_per_second ? ` (${live.tokens_per_second} در ثانیه)` : ""));
        else parts.push(`منتظر اولین خروجی مدل (${Math.round(live.call_seconds)} ثانیه) — بارگذاری مدل در GPU/RAM یا خواندن متن ورودی؛ بار اول ممکن است چند دقیقه طول بکشد`);
      }
      if (active.failed) parts.push(`${active.failed} بخش ناموفق`);
      parts.push(`زمان سپری‌شده: ${fmtTime(active.elapsed_seconds)}`);
      parts.push(active.eta_seconds != null ? `باقی‌مانده تقریبی: ${fmtTime(active.eta_seconds)}` : "زمان باقی‌مانده بعد از اولین بخش محاسبه می‌شود");
      meta.textContent = parts.join(" · ");
      banner.append(line, bar, meta);
      banner.hidden = false;
      return;
    }
    for (const id of seenActive.keys()) {
      const task = tasks.find((t) => t.id === id);
      if (task) finished = task;
    }
    seenActive.clear();
    if (finished) {
      const task = tasks.find((t) => t.id === finished.id) || finished;
      banner.classList.add(task.status === "completed" ? "done" : "warn");
      const line = document.createElement("div");
      line.className = "job-line";
      const title = document.createElement("span");
      title.className = "job-title";
      title.textContent = `«${task.title}» — ${STATE_LABEL[task.status] || task.status} (${task.done} از ${task.total} بخش)`;
      const why = task.error || (task.last_chunk_error && `بخش ${task.last_chunk_error.idx + 1}: ${task.last_chunk_error.error}`);
      if (why && task.status !== "completed") title.title = why;
      const actions = document.createElement("span");
      actions.className = "job-actions";
      if (task.done > 0) actions.append(bannerButton("دانلود DOCX", () => download(task.id, "docx")), bannerButton("PDF", () => download(task.id, "pdf")));
      if (task.status !== "completed" && task.resumable) actions.append(bannerButton("ادامه", () => act(task.id, "resume").then(pollBanner)));
      actions.append(bannerButton("کارهای طولانی", openPanel), bannerButton("×", () => { finished = null; banner.hidden = true; }));
      line.append(title, actions);
      banner.appendChild(line);
      if (why && task.status !== "completed") {
        const reason = document.createElement("div");
        reason.className = "job-reason";
        reason.textContent = why;
        banner.appendChild(reason);
      }
      banner.hidden = false;
      return;
    }
    banner.hidden = true;
  }
  async function pollBanner() {
    clearTimeout(bannerTimer);
    let delay = 8000;
    try {
      const data = await call("GET", "/api/tasks");
      const tasks = data.tasks || [];
      renderBanner(tasks);
      setBusy(data.busy || { busy: false });
      if (tasks.some((t) => ACTIVE.has(t.status))) delay = 2000;
      else if (data.busy && data.busy.busy) delay = 3000;
    } catch (error) {
      delay = 15000;                                      // not signed in yet / server restarting: stay quiet
    }
    bannerTimer = setTimeout(pollBanner, delay);
  }
  window.refreshJobBanner = pollBanner;
  pollBanner();

  document.getElementById("tasksButton").addEventListener("click", () => { overlay.hidden = false; say(""); refresh(); loadOutDir(); });
  const close = () => { overlay.hidden = true; clearTimeout(timer); };
  document.getElementById("closeTasks").addEventListener("click", close);
  overlay.addEventListener("click", (event) => { if (event.target === overlay) close(); });
})();
