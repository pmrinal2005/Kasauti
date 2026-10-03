/** @type {import('next').NextConfig} */
const COI = process.env.NEXT_PUBLIC_LAYA_COI === "1";

/**
 * Cross-origin isolation is OFF by default, deliberately.
 *
 * COOP/COEP would buy multi-threaded WASM for onnxruntime-web (SharedArrayBuffer), but it also
 * blocks every `new Worker(url)` unless the response carries a matching CORP header — and in
 * practice it breaks workers outright in several Chrome builds and in the in-app browsers
 * (WhatsApp / Telegram / Instagram webviews) where a large share of Kasauti's traffic lands.
 * Single-threaded WASM + WebGPU needs no isolation at all, so the default is: no COI, one ORT
 * thread, everything works. Set NEXT_PUBLIC_LAYA_COI=1 to opt in (desktop, standalone browsers).
 */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async headers() {
    const security = [
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
      { key: "Permissions-Policy", value: "microphone=(self), camera=(), geolocation=()" },
      { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
    ];
    const coi = COI
      ? [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ]
      : [];
    return [
      {
        // Static assets, worker chunks and public files. CORP must be present on these or a
        // COI-enabled deployment would block `new Worker(new URL(...))` with
        // net::ERR_BLOCKED_BY_RESPONSE (see the COI note above).
        source: "/:path*",
        headers: [...coi, ...security],
      },
      {
        source: "/api/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "no-referrer" },
          { key: "X-Robots-Tag", value: "noindex" },
        ],
      },
      { source: "/sw.js", headers: [{ key: "Cache-Control", value: "no-cache" }, { key: "Service-Worker-Allowed", value: "/" }] },
    ];
  },
};
export default nextConfig;
