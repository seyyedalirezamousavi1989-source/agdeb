/* روی http (شبکهٔ محلی، مثلاً iPhone → http://192.168…) مرورگر چند API را که فقط در «زمینهٔ امن» هست حذف می‌کند.
   بدون این جایگزین‌ها ارسال پیام با خطای crypto.randomUUID بی‌صدا شکست می‌خورد و دکمهٔ «کپی» کار نمی‌کند. */
(function () {
  try {
    if (window.crypto && typeof window.crypto.randomUUID !== "function" && window.crypto.getRandomValues) {
      window.crypto.randomUUID = function () {
        var b = new Uint8Array(16); window.crypto.getRandomValues(b);
        b[6] = (b[6] & 0x0f) | 0x40; b[8] = (b[8] & 0x3f) | 0x80;
        var h = Array.prototype.map.call(b, function (x) { return (x + 256).toString(16).slice(1); }).join("");
        return h.slice(0, 8) + "-" + h.slice(8, 12) + "-" + h.slice(12, 16) + "-" + h.slice(16, 20) + "-" + h.slice(20);
      };
    }
  } catch (_) { /* ignore */ }
  try {
    if (!navigator.clipboard) {
      Object.defineProperty(navigator, "clipboard", { configurable: true, value: {
        writeText: function (text) {
          return new Promise(function (resolve, reject) {
            var area = document.createElement("textarea");
            area.value = String(text); area.setAttribute("readonly", ""); area.className = "clip-shim";
            document.body.appendChild(area); area.select(); area.setSelectionRange(0, area.value.length);
            var ok = false; try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
            document.body.removeChild(area);
            ok ? resolve() : reject(new Error("copy failed"));
          });
        }
      } });
    }
  } catch (_) { /* ignore */ }
})();

/* تم روشن/تیره + جمع‌شدن نوار کناری. این فایل در <head> و بدون defer بارگذاری می‌شود تا هنگام باز شدن صفحه چشمک نزند. */
(function () {
  var KEY = "localchat-theme";
  var SIDEBAR_KEY = "localchat-sidebar-collapsed";
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function stored(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
  function save(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* ignore */ } }

  function resolved() {
    var choice = stored(KEY);
    if (choice === "light" || choice === "dark") return choice;
    return media && media.matches ? "dark" : "light";
  }
  function applyTheme() {
    var theme = resolved();
    root.setAttribute("data-theme", theme);
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#141416" : "#ffffff");
  }
  applyTheme();
  if (media && media.addEventListener) media.addEventListener("change", function () { if (!stored(KEY)) applyTheme(); });

  // PWA: the service worker only works on HTTPS (or localhost); on plain-HTTP LAN it is skipped silently
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    window.addEventListener("load", function () { navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(function () {}); });
  }
  if (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches || navigator.standalone) root.classList.add("pwa-standalone");

  document.addEventListener("DOMContentLoaded", function () {
    applyTheme();
    // «اتصال گوشی» is offered only on the PC itself (the status endpoint is local-only)
    var phoneLink = document.getElementById("phoneLink");
    if (phoneLink) fetch("/api/remote/status", { credentials: "same-origin" }).then(function (r) { if (r.ok) phoneLink.hidden = false; }).catch(function () {});
    var toggle = document.getElementById("themeToggle");
    if (toggle) toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      save(KEY, next);
      applyTheme();
    });

    var shell = document.querySelector(".app-shell");
    var collapse = document.getElementById("collapseSidebar");
    var expand = document.getElementById("expandSidebar");
    if (!shell) return;
    if (stored(SIDEBAR_KEY) === "1") shell.classList.add("sidebar-collapsed");
    function setCollapsed(value) {
      shell.classList.toggle("sidebar-collapsed", value);
      save(SIDEBAR_KEY, value ? "1" : "0");
    }
    if (collapse) collapse.addEventListener("click", function () { setCollapsed(true); });
    if (expand) expand.addEventListener("click", function () { setCollapsed(false); });
  });
})();
