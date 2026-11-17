/* ورود با کد جفت‌سازی + ثبت سرویس‌ورکر. روی صفحهٔ آفلاین هم برای دکمهٔ «تلاش دوباره» بارگذاری می‌شود. */
(function () {
  var retry = document.getElementById("retry");
  if (retry) retry.addEventListener("click", function () { location.href = "/"; });

  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(function () { /* HTTP روی شبکهٔ محلی: بدون سرویس‌ورکر هم کار می‌کند */ });
  }

  var form = document.getElementById("loginForm");
  if (!form) return;
  var code = document.getElementById("code");
  var name = document.getElementById("deviceName");
  var error = document.getElementById("error");
  var submit = document.getElementById("submit");

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    error.textContent = "";
    submit.disabled = true;
    fetch("/api/remote/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: code.value, name: name.value })
    }).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (data) { return { ok: response.ok, data: data }; });
    }).then(function (result) {
      if (result.ok) { location.replace("/"); return; }
      error.textContent = result.data.error || "ورود ناموفق بود.";
      submit.disabled = false;
    }).catch(function () {
      error.textContent = "ارتباط با رایانه برقرار نشد.";
      submit.disabled = false;
    });
  });
})();
