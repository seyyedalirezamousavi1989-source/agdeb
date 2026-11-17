/* صفحهٔ مدیریت اتصال گوشی؛ فقط روی خود رایانه کار می‌کند. */
(function () {
  var csrf = "";
  function $(id) { return document.getElementById(id); }
  function say(text, ok) { var m = $("msg"); m.textContent = text || ""; m.className = "remote-error" + (ok ? " remote-ok" : ""); }

  function api(path, body) {
    var run = function () {
      return fetch(path, {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-LocalChat-CSRF": csrf },
        body: JSON.stringify(body || {})
      }).then(function (r) { return r.json().catch(function () { return {}; }).then(function (d) { if (!r.ok) throw new Error(d.error || "ناموفق"); return d; }); });
    };
    if (csrf) return run();
    return fetch("/api/session", { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (d) { csrf = d.csrf; return run(); });
  }

  function fmt(ts) { return ts ? new Date(ts * 1000).toLocaleString("fa-IR") : "—"; }

  function render(data) {
    $("state").textContent = data.enabled ? "روشن" : "خاموش";
    $("state").className = "pill" + (data.enabled ? " on" : "");
    $("offBox").hidden = data.enabled;
    $("onBox").hidden = !data.enabled;
    var urls = $("urls"); urls.textContent = "";
    (data.urls || []).forEach(function (u) { var li = document.createElement("li"); li.textContent = u; urls.appendChild(li); });
    $("noUrl").hidden = (data.urls || []).length > 0;
    $("code").textContent = data.code || "";
    var box = $("devices"); box.textContent = "";
    if (!(data.devices || []).length) { var p = document.createElement("p"); p.textContent = "هنوز دستگاهی جفت نشده است."; box.appendChild(p); }
    (data.devices || []).forEach(function (d) {
      var row = document.createElement("div"); row.className = "row";
      var label = document.createElement("span"); label.textContent = d.name || "دستگاه";
      var small = document.createElement("small"); small.textContent = " آخرین استفاده: " + fmt(d.last_seen);
      label.appendChild(small);
      var btn = document.createElement("button"); btn.type = "button"; btn.textContent = "خارج کردن";
      btn.addEventListener("click", function () { api("/api/remote/revoke", { id: d.id }).then(load).catch(function (e) { say(e.message); }); });
      row.appendChild(label); row.appendChild(btn); box.appendChild(row);
    });
  }

  function load() {
    return fetch("/api/remote/status", { credentials: "same-origin" })
      .then(function (r) { if (!r.ok) throw new Error("این صفحه فقط روی خود رایانه باز می‌شود."); return r.json(); })
      .then(render).catch(function (e) { say(e.message); });
  }

  $("enableBtn").addEventListener("click", function () {
    api("/api/remote/enable", { enabled: true }).then(function () { say("ذخیره شد. برنامه را ببندید و دوباره با start_phone.bat اجرا کنید.", true); load(); }).catch(function (e) { say(e.message); });
  });
  $("disableBtn").addEventListener("click", function () {
    api("/api/remote/enable", { enabled: false }).then(function () { say("ذخیره شد؛ پس از اجرای مجدد برنامه دسترسی گوشی بسته می‌شود.", true); load(); }).catch(function (e) { say(e.message); });
  });
  $("regenBtn").addEventListener("click", function () {
    api("/api/remote/regenerate", { revoke_devices: true }).then(function () { say("کد جدید ساخته شد و دستگاه‌های قبلی خارج شدند.", true); load(); }).catch(function (e) { say(e.message); });
  });
  load();
})();
