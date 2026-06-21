const CACHE_NAME = 'aperture-cam-v3'; // bump invalidates the poisoned v2 cache

self.addEventListener('install', (e) => self.skipWaiting());

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
    .then((names) => Promise.all(
      names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))
    ))
    .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (!e.request.url.startsWith(self.location.origin)) return; // skip CDNs/ML models
  
  e.respondWith(
    fetch(e.request)
    .then((res) => {
      // Only ever cache real, successful pages — never 403/404/500
      if (res && res.ok && res.status === 200) {
        const copy = res.clone();
        caches.open(CACHE_NAME).then((c) => c.put(e.request, copy));
      }
      return res;
    })
    .catch(async () => {
      const cached = await caches.match(e.request);
      return cached || new Response('Offline — no cached copy.', { status: 503 });
    })
  );
});