/* Service worker — only the app shell is cached. Chat, models and every /api call always go to the PC.
   The model lives on the PC, so there is nothing useful to do offline except show a clear message.
   __VERSION__ is replaced by the server so a new build refreshes the cache. */
const VERSION = "__VERSION__";
const CACHE = "localchat-shell-" + VERSION;
const SHELL = [
  "/offline.html", "/static/remote.css", "/static/style.css", "/static/theme.js",
  "/static/fonts/Vazirmatn-var.woff2", "/static/icons/icon-192.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => Promise.all(SHELL.map((url) => cache.add(url).catch(() => null))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("localchat-shell-") && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return;                 // never cache or intercept chat / streams

  if (request.mode === "navigate") {                             // page loads: network first, friendly page when the PC is unreachable
    event.respondWith(
      fetch(request).catch(() => caches.match("/offline.html").then((r) => r || Response.error()))
    );
    return;
  }
  if (url.pathname.startsWith("/static/")) {                     // static files: network first (they change per build), cache as fallback
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.ok) { const copy = response.clone(); caches.open(CACHE).then((c) => c.put(request, copy)); }
          return response;
        })
        .catch(() => caches.match(request).then((r) => r || Response.error()))
    );
  }
});
