/* Kasauti service worker — hand-rolled.
 * Cache-first for immutable, content-hashed assets (/_next/static, model binaries handled by the
 * inference worker's own Cache API store), network-first for the dashboard shell so it works offline. */
const SHELL = "kasauti-shell-v1";
self.addEventListener("install", (e) => { self.skipWaiting(); e.waitUntil(caches.open(SHELL).then((c) => c.addAll(["/dashboard", "/icon.svg", "/manifest.webmanifest"]).catch(() => {}))); });
self.addEventListener("activate", (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k.startsWith("kasauti-shell") && k !== SHELL).map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin || u.pathname.startsWith("/api/")) return;
  if (u.pathname.startsWith("/_next/static/")) {
    e.respondWith(caches.open(SHELL).then(async (c) => (await c.match(e.request)) || fetch(e.request).then((r) => { if (r.ok) c.put(e.request, r.clone()); return r; })));
    return;
  }
  if (e.request.mode === "navigate") {
    e.respondWith(fetch(e.request).then((r) => { const cl = r.clone(); caches.open(SHELL).then((c) => c.put(e.request, cl)); return r; }).catch(async () => (await caches.match(e.request)) || caches.match("/dashboard")));
  }
});
