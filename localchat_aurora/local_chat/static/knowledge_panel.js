// Private knowledge base panel: add books/articles/notes, switch sources on/off, test what the agent will retrieve.
(() => {
  const overlay = document.getElementById("knowledgeOverlay");
  if (!overlay) return;
  const $ = (id) => document.getElementById(id);
  const list = $("knowledgeList"), results = $("knowledgeResults"), statusLine = $("knowledgeStatus"), embedLine = $("knowledgeEmbedStatus");
  let timer = null;
  let profiles = [];

  function say(text, bad = false) { statusLine.textContent = text || ""; statusLine.style.color = bad ? "#b3261e" : ""; }
  async function call(method, url, body) {
    const options = { method };
    if (body !== undefined) { options.headers = { "Content-Type": "application/json" }; options.body = JSON.stringify(body); }
    const response = await apiFetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `خطا (${response.status})`);
    return data;
  }
  function button(label, handler) {
    const el = document.createElement("button");
    el.type = "button"; el.className = "settings-button"; el.textContent = label;
    el.addEventListener("click", handler);
    return el;
  }
  function fmtChars(n) { return n >= 1e6 ? `${(n / 1e6).toFixed(1)} میلیون نویسه` : `${Math.round(n / 1000)} هزار نویسه`; }

  function renderEmbedding(info, stats) {
    const model = info.model;
    let text;
    if (!model) text = "جست‌وجوی معنایی غیرفعال است (مدل embedding نصب نیست)؛ از «مدیریت مدل‌ها ← معنایی 🔎» مثلاً bge-m3 را دانلود کن. فعلاً جست‌وجوی کلیدواژه‌ای کار می‌کند.";
    else if (info.state === "embedding") text = `در حال ایندکس معنایی با ${model}… ${info.remaining ?? ""} بخش مانده`;
    else if (info.state === "waiting") text = "ایندکس معنایی تا پایان پاسخ چت صبر می‌کند…";
    else if (info.state === "error") text = `ایندکس معنایی ناموفق بود: ${info.error}`;
    else text = stats.chunks ? `جست‌وجوی ترکیبی (کلیدواژه + معنایی) فعال است · مدل: ${model}` : `مدل embedding: ${model}`;
    embedLine.textContent = text;
    embedLine.style.color = info.state === "error" ? "#b3261e" : "";
    const select = $("knowledgeEmbedModel");
    const names = ["", ...new Set([...(info.candidates || []), ...(select.dataset.chosen ? [select.dataset.chosen] : [])])];
    select.replaceChildren(...names.map((n) => { const o = document.createElement("option"); o.value = n; o.textContent = n || "انتخاب خودکار"; return o; }));
    select.value = select.dataset.chosen || "";
  }

  function render(data) {
    $("knowledgeEnabled").checked = data.settings.enabled;
    $("knowledgeEmbedModel").dataset.chosen = data.settings.embed_model || "";
    renderEmbedding(data.embedding || {}, data.stats);
    const collections = new Set(data.sources.map((s) => s.collection).filter(Boolean));
    $("knowledgeCollections").replaceChildren(...[...collections].map((c) => { const o = document.createElement("option"); o.value = c; return o; }));
    const filter = $("knowledgeCollectionFilter");
    const knowledgeSettings = window.localChatKnowledgeSettings?.() || { collections: null };
    const configuredNames = new Set((data.collections || []).map((item) => item.name));
    const selectedNames = knowledgeSettings.collections || [];
    const options = (data.collections || []).map((item) => {
      const option = document.createElement("option");
      option.value = item.name;
      option.textContent = `${item.name || "بدون دسته‌بندی"} · ${item.enabled_chunks} بخش فعال`;
      option.selected = selectedNames.includes(item.name);
      return option;
    });
    for (const name of selectedNames) {
      if (configuredNames.has(name)) continue;
      const option = document.createElement("option");
      option.value = name;
      option.textContent = `${name || "بدون دسته‌بندی"} · موجود نیست`;
      option.selected = true;
      options.push(option);
    }
    filter.replaceChildren(...options);
    window.localChatRenderKnowledgeControls?.();
    profiles = data.profiles || [];
    const profileSelect = $("knowledgeProfile");
    const activeSettings = window.localChatKnowledgeSettings?.() || {};
    profileSelect.replaceChildren(new Option("بدون پروفایل", ""));
    for (const profile of profiles) {
      const option = new Option(`${profile.name}${profile.active ? " · فعال" : ""}`, profile.id);
      option.selected = activeSettings.profile_id === profile.id;
      profileSelect.appendChild(option);
    }
    if (activeSettings.profile_id && !profiles.some((profile) => profile.id === activeSettings.profile_id)) {
      const option = new Option("پروفایل ذخیره‌شده موجود نیست", activeSettings.profile_id);
      option.selected = true;
      profileSelect.appendChild(option);
    }
    renderProfileList();
    list.replaceChildren();
    if (!data.sources.length) {
      const empty = document.createElement("div");
      empty.className = "task-meta"; empty.textContent = "هنوز چیزی اضافه نشده است.";
      list.appendChild(empty);
      return;
    }
    for (const source of data.sources) {
      const item = document.createElement("div");
      item.className = "task-item knowledge-item" + (source.enabled ? "" : " off");
      const head = document.createElement("div"); head.className = "task-head";
      const title = document.createElement("span"); title.className = "task-title"; title.textContent = source.title; title.title = source.filename || source.title;
      const state = document.createElement("span"); state.className = "task-state";
      state.textContent = source.embedded >= source.chunks ? "ایندکس کامل" : `معنایی: ${source.embedded} از ${source.chunks}`;
      head.append(title, state);
      const meta = document.createElement("div"); meta.className = "task-meta";
      meta.textContent = `${source.chunks} بخش · ${fmtChars(source.chars)}${source.collection ? " · دسته: " + source.collection : ""}${source.kind === "note" ? " · یادداشت" : ""}`;
      item.append(head, meta);
      for (const note of source.notes || []) { const n = document.createElement("div"); n.className = "task-error"; n.textContent = note; item.appendChild(n); }
      const actions = document.createElement("div"); actions.className = "task-actions";
      actions.append(
        button(source.enabled ? "غیرفعال‌کردن" : "فعال‌کردن", () => act(`/api/knowledge/${source.id}/toggle`, { enabled: !source.enabled })),
        button("حذف", () => { if (confirm(`«${source.title}» از دانش حذف شود؟`)) act(`/api/knowledge/${source.id}/delete`, {}); }));
      item.appendChild(actions);
      list.appendChild(item);
    }
  }

  function renderProfileList() {
    const container = $("knowledgeProfiles");
    container.replaceChildren();
    for (const profile of profiles) {
      const item = document.createElement("div");
      item.className = "task-item knowledge-item";
      const head = document.createElement("div"); head.className = "task-head";
      const name = document.createElement("span"); name.className = "task-title"; name.textContent = profile.name;
      const state = document.createElement("span"); state.className = "task-state";
      state.textContent = profile.active ? "پروفایل فعال" : profile.enabled ? "آمادهٔ انتخاب" : "غیرفعال";
      head.append(name, state);
      const description = document.createElement("div"); description.className = "task-meta";
      description.textContent = `${profile.description || "بدون توضیح"} · مجموعه‌ها: ${profile.collections.map((value) => value || "بدون دسته‌بندی").join("، ")}`;
      const actions = document.createElement("div"); actions.className = "task-actions";
      actions.append(
        button("ویرایش", () => {
          $("knowledgeProfile").value = profile.id;
          $("knowledgeProfileName").value = profile.name;
          $("knowledgeProfileDescription").value = profile.description;
          $("knowledgeProfilePriority").value = profile.priority;
          const selected = new Set(profile.collections);
          for (const option of $("knowledgeCollectionFilter").options) option.selected = selected.has(option.value);
        }),
        button(profile.enabled ? "غیرفعال‌کردن" : "فعال‌کردن", () => act(
          `/api/knowledge/profiles/${profile.id}/enabled`, { enabled: !profile.enabled })),
        button("حذف", () => {
          if (confirm(`پروفایل «${profile.name}» حذف شود؟ اسناد حذف نمی‌شوند.`))
            act(`/api/knowledge/profiles/${profile.id}/delete`, {});
        }));
      item.append(head, description, actions);
      container.appendChild(item);
    }
  }

  async function refresh() {
    try {
      const data = await call("GET", "/api/knowledge");
      render(data);
      clearTimeout(timer);
      const working = ["embedding", "waiting"].includes((data.embedding || {}).state);
      if (!overlay.hidden) timer = setTimeout(refresh, working ? 2000 : 15000);
    } catch (error) { say(error.message, true); }
  }
  async function act(url, body) {
    try { await call("POST", url, body); say(""); } catch (error) { say(error.message, true); }
    refresh();
  }

  $("knowledgeUpload").addEventListener("click", async () => {
    const input = $("knowledgeFile"), files = [...(input.files || [])];
    if (!files.length) return say("ابتدا یک یا چند فایل انتخاب کن.", true);
    const collection = $("knowledgeCollection").value.trim();
    $("knowledgeUpload").disabled = true;
    const problems = [];
    let added = 0;
    for (const [i, file] of files.entries()) {
      say(`در حال خواندن و ایندکس «${file.name}» (${i + 1} از ${files.length})…`);
      try {
        const response = await apiFetch("/api/knowledge", {
          method: "POST", body: file,
          headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name),
                     "X-Knowledge-Options": encodeURIComponent(JSON.stringify({ collection })) },
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || `خطا (${response.status})`);
        added += 1;
      } catch (error) { problems.push(`${file.name}: ${error.message}`); }
    }
    input.value = "";
    $("knowledgeUpload").disabled = false;
    say(problems.length ? `${added} فایل اضافه شد. ${problems.join(" | ")}` : `${added} فایل به دانش اضافه شد.`, problems.length > 0);
    refresh();
  });
  $("knowledgeNoteAdd").addEventListener("click", async () => {
    try {
      await call("POST", "/api/knowledge/text", { title: $("knowledgeNoteTitle").value, text: $("knowledgeNoteText").value, collection: $("knowledgeCollection").value.trim() });
      $("knowledgeNoteTitle").value = ""; $("knowledgeNoteText").value = "";
      say("یادداشت ذخیره شد.");
    } catch (error) { say(error.message, true); }
    refresh();
  });
  $("knowledgeEnabled").addEventListener("change", (event) => act("/api/knowledge/settings", { enabled: event.target.checked }));
  $("knowledgeEmbedModel").addEventListener("change", (event) => { event.target.dataset.chosen = event.target.value; act("/api/knowledge/settings", { embed_model: event.target.value }); });
  $("knowledgeReindex").addEventListener("click", () => act("/api/knowledge/reindex", {}));
  $("knowledgeProfile").addEventListener("change", async (event) => {
    const profile = profiles.find((item) => item.id === event.target.value);
    if (!profile) {
      window.localChatSetKnowledgeProfile?.(null);
      return;
    }
    try {
      const result = await call("POST", `/api/knowledge/profiles/${profile.id}/activate`, {});
      window.localChatSetKnowledgeProfile?.(result.profile);
      say(`پروفایل «${profile.name}» برای این گفت‌وگو فعال شد.`);
      await refresh();
    } catch (error) { say(error.message, true); }
  });
  $("knowledgeProfileSave").addEventListener("click", async () => {
    const selected = [...$("knowledgeCollectionFilter").selectedOptions].map((option) => option.value);
    if (!selected.length) return say("برای ساخت پروفایل، دست‌کم یک مجموعه را انتخاب کن.", true);
    const currentId = $("knowledgeProfile").value;
    const payload = {
      name: $("knowledgeProfileName").value,
      description: $("knowledgeProfileDescription").value,
      collections: selected,
      priority: Number($("knowledgeProfilePriority").value || 0),
    };
    if (profiles.some((profile) => profile.id === currentId)) payload.id = currentId;
    try {
      const result = await call("POST", "/api/knowledge/profiles", payload);
      const active = await call("POST", `/api/knowledge/profiles/${result.profile.id}/activate`, {});
      window.localChatSetKnowledgeProfile?.(active.profile);
      say("پروفایل دانش ذخیره و برای این گفت‌وگو انتخاب شد.");
      await refresh();
    } catch (error) { say(error.message, true); }
  });
  async function search() {
    const query = $("knowledgeQuery").value.trim();
    if (!query) return;
    results.replaceChildren();
    try {
      const data = await call("POST", "/api/knowledge/search", { query });
      if (!data.results.length) { const none = document.createElement("div"); none.className = "task-meta"; none.textContent = "گزیدهٔ مرتبطی پیدا نشد؛ در چت هم چیزی تزریق نمی‌شود."; results.appendChild(none); return; }
      for (const hit of data.results) {
        const box = document.createElement("div"); box.className = "knowledge-hit";
        const head = document.createElement("b"); head.textContent = hit.title + (hit.heading ? " › " + hit.heading : "");
        const how = document.createElement("small"); how.textContent = `روش: ${hit.via.join(" + ")}`;
        const body = document.createElement("div"); body.textContent = hit.text.length > 420 ? hit.text.slice(0, 420) + " …" : hit.text;
        box.append(head, how, body);
        results.appendChild(box);
      }
    } catch (error) { say(error.message, true); }
  }
  $("knowledgeSearch").addEventListener("click", search);
  $("knowledgeQuery").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); search(); } });

  $("knowledgeButton").addEventListener("click", () => { overlay.hidden = false; say(""); refresh(); });
  const close = () => { overlay.hidden = true; clearTimeout(timer); };
  $("closeKnowledge").addEventListener("click", close);
  overlay.addEventListener("click", (event) => { if (event.target === overlay) close(); });
})();
