// Memory / context panel: what Ollama holds right now, why, and a one-click release + automatic-behaviour settings.
(() => {
  const topbar = document.querySelector(".topbar");
  if (!topbar || document.getElementById("runtimeChip")) return;

  const chip = document.createElement("button");
  chip.id = "runtimeChip";
  chip.type = "button";
  chip.className = "runtime-chip";
  chip.title = "وضعیت حافظه و کانتکست مدل";
  chip.textContent = "حافظه: …";
  const panel = document.createElement("div");
  panel.id = "runtimePanel";
  panel.className = "runtime-panel";
  panel.hidden = true;
  const power = document.getElementById("powerButton");
  topbar.insertBefore(chip, power || null);
  document.body.appendChild(panel);

  let timer = null;
  let busy = false;

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
  const fmtLeft = (s) => (s == null ? "—" : s < 90 ? `${s} ثانیه` : `${Math.round(s / 60)} دقیقه`);

  function paintChip(data) {
    const n = data.models.length;
    const gb = data.models.reduce((sum, m) => sum + (m.size_gb || 0), 0);
    const warn = data.findings.some((f) => f.level === "warn");
    chip.textContent = n ? `حافظه: ${n} مدل · ${gb.toFixed(1)}GB` : "حافظه: آزاد";
    chip.classList.toggle("warn", warn);
  }

  function paintPanel(data) {
    panel.textContent = "";
    const title = document.createElement("h3");
    title.textContent = "حافظه و کانتکست مدل";
    panel.appendChild(title);

    if (data.models.length) {
      const table = document.createElement("div");
      table.className = "runtime-models";
      data.models.forEach((m) => {
        const row = document.createElement("div");
        row.className = "runtime-model";
        const gpu = m.gpu_percent == null ? "" : ` · ${m.gpu_percent}٪ روی GPU`;
        const ctx = m.context ? ` · پنجره ${m.context} توکن` : "";
        row.textContent = `${m.name} — ${m.size_gb}GB${gpu}${ctx} · آزادسازی خودکار Ollama تا ${fmtLeft(m.expires_in)}`;
        table.appendChild(row);
      });
      panel.appendChild(table);
    }
    const list = document.createElement("ul");
    list.className = "runtime-findings";
    data.findings.forEach((f) => {
      const li = document.createElement("li");
      li.className = f.level;
      li.textContent = f.text;
      list.appendChild(li);
    });
    panel.appendChild(list);

    if (data.last_release) {
      const note = document.createElement("p");
      note.className = "runtime-note";
      const when = new Date(data.last_release.at * 1000).toLocaleTimeString("fa-IR");
      note.textContent = `آخرین آزادسازی ${data.last_release.reason === "idle" ? "خودکار" : "دستی"}: ${when} — ${data.last_release.models.join("، ")}`;
      panel.appendChild(note);
    }

    const release = document.createElement("button");
    release.type = "button";
    release.className = "settings-button primary";
    release.textContent = "آزادسازی حافظهٔ مدل‌ها";
    release.disabled = busy || !data.models.length || data.busy.chat || data.busy.task;
    release.addEventListener("click", async () => {
      busy = true;
      try { render(await call("POST", "/api/runtime/release", {})); }
      catch (error) { say(error.message, true); }
      finally { busy = false; }
    });
    panel.appendChild(release);

    const hw = document.createElement("div");
    hw.className = "runtime-hw";
    const gpu = data.hardware && data.hardware.gpu;
    const ram = data.hardware && data.hardware.ram;
    const lines = [];
    lines.push(gpu ? `کارت گرافیک: ${gpu.name} — ${gpu.total_gb}GB (آزاد ${gpu.free_gb}GB)` : "کارت گرافیک NVIDIA شناسایی نشد؛ انتخاب کانتکست بر اساس RAM انجام می‌شود.");
    if (ram) lines.push(`RAM: ${ram.total_gb}GB (آزاد ${ram.available_gb}GB)`);
    if (data.window_plan) {
      const p = data.window_plan;
      lines.push(`پنجرهٔ خودکار برای «${p.model}»: ${p.num_ctx} توکن (KV-cache ≈ ${p.kv_mb}MB؛ GPU ${p.gpu_kv_mb ?? "—"}MB / RAM ${p.ram_kv_mb ?? "—"}MB) — ${p.reason}`);
    }
    if (data.hardware_unavailable && data.hardware_unavailable.length) lines.push(`قابل‌اندازه‌گیری نیست: ${data.hardware_unavailable.join("، ")}`);
    (data.contexts || []).forEach((item) => {
      const c = item.context || {}, p = item.plan || {};
      const labels = {verified: "تأییدشده", mismatch: "ناهمسان", unknown: "نامشخص", not_loaded: "بارگذاری‌نشده", unreachable: "Ollama در دسترس نیست"};
      lines.push(`«${item.model}»: اعلام‌شده ${c.declared ?? "؟"} · واقعی (Ollama) ${c.actual ?? "؟"} · پیشنهاد برنامه ${p.target ?? "؟"} (برآورد، ${p.confidence || "؟"}؛ محدودکننده: ${p.limiting || "؟"})`);
    });
    lines.forEach((text) => { const row = document.createElement("div"); row.textContent = text; hw.appendChild(row); });
    (data.contexts || []).forEach((item) => {
      const c = item.context || {}, p = item.plan || {};
      if (!p.target || c.actual === p.target) return;
      const apply = document.createElement("button");
      apply.type = "button";
      apply.textContent = `اعمال پنجرهٔ ${p.target} برای «${item.model}» (مدل دوباره بارگذاری می‌شود)`;
      apply.disabled = busy || data.busy.chat || data.busy.task;
      apply.addEventListener("click", async () => {
        busy = true; apply.disabled = true;
        try {
          const r = await call("POST", "/api/runtime/apply-context", { model: item.model, num_ctx: p.target });
          say(r.applied ? `پنجره ${r.context.actual} تأیید شد.` : `اعمال تأیید نشد: ${r.context.detail || r.context.status}`, !r.applied);
          render(await call("GET", "/api/runtime/status"));
        } catch (error) { say(error.message, true); }
        finally { busy = false; }
      });
      hw.appendChild(apply);
    });
    panel.appendChild(hw);

    const settings = document.createElement("div");
    settings.className = "runtime-settings";
    const ctxLabel = document.createElement("label");
    const ctxSelect = document.createElement("select");
    [0, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144].forEach((n) => {
      const option = document.createElement("option");
      option.value = String(n); option.textContent = n ? String(n) : "خودکار";
      if (n === data.settings.max_context) option.selected = true;
      ctxSelect.appendChild(option);
    });
    ctxSelect.addEventListener("change", async () => {
      try { render(await call("POST", "/api/runtime/settings", { max_context: parseInt(ctxSelect.value, 10) })); }
      catch (error) { say(error.message, true); }
    });
    ctxLabel.append("سقف پنجرهٔ کانتکست (برنامه بزرگ‌ترین مقدار مناسب سخت‌افزار را تا این سقف انتخاب می‌کند): ", ctxSelect);
    settings.appendChild(ctxLabel);
    const auto = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = !!data.settings.auto_resume_tasks;
    box.addEventListener("change", async () => {
      try { render(await call("POST", "/api/runtime/settings", { auto_resume_tasks: box.checked })); }
      catch (error) { say(error.message, true); }
    });
    auto.append(box, " ازسرگیری خودکار ترجمه/خلاصهٔ قطع‌شده بعد از راه‌اندازی مجدد (پیش‌فرض: خاموش)");
    const idle = document.createElement("label");
    const input = document.createElement("input");
    input.type = "number"; input.min = "0"; input.max = "240";
    input.value = data.settings.idle_unload_minutes;
    input.className = "settings-input";
    input.addEventListener("change", async () => {
      try { render(await call("POST", "/api/runtime/settings", { idle_unload_minutes: parseInt(input.value, 10) })); }
      catch (error) { say(error.message, true); }
    });
    idle.append("آزادسازی خودکار بعد از ", input, " دقیقه بی‌کاری (۰ = خاموش)");
    settings.append(auto, idle);
    panel.appendChild(settings);

    const clock = document.createElement("div");
    clock.className = "runtime-hw";
    const c = data.clock || {};
    const age = c.synced ? (c.age_days < 1 ? "امروز" : `${Math.round(c.age_days)} روز پیش`) : "هرگز";
    clock.textContent = `ساعت داخلی برنامه: ${c.now || "—"} · آخرین همگام‌سازی با اینترنت: ${age}` +
      (c.synced ? ` · اختلاف با ساعت ویندوز ${c.offset_seconds} ثانیه` : "");
    const sync = document.createElement("button");
    sync.type = "button";
    sync.className = "settings-button";
    sync.textContent = "همگام‌سازی ساعت (یک بار با اینترنت)";
    sync.addEventListener("click", async () => {
      sync.disabled = true;
      try { await call("POST", "/api/time/sync", {}); render(await call("GET", "/api/runtime/status")); }
      catch (error) { say(error.message, true); sync.disabled = false; }
    });
    clock.appendChild(document.createElement("br"));
    clock.appendChild(sync);
    panel.appendChild(clock);

    const status = document.createElement("p");
    status.id = "runtimeStatus";
    status.className = "runtime-note";
    panel.appendChild(status);
  }
  function say(text, bad) {
    const el = document.getElementById("runtimeStatus");
    if (el) { el.textContent = text; el.style.color = bad ? "#b3261e" : ""; }
  }
  function render(data) { paintChip(data); if (!panel.hidden) paintPanel(data); return data; }

  async function refresh() {
    try { render(await call("GET", "/api/runtime/status")); }
    catch (_) { chip.textContent = "حافظه: ؟"; }
  }
  chip.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    if (!panel.hidden) refresh();
  });
  document.addEventListener("click", (event) => {
    if (!panel.hidden && !panel.contains(event.target) && event.target !== chip) panel.hidden = true;
  });
  refresh();
  timer = setInterval(refresh, 15000);
})();
