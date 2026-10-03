/**
 * URL helpers that both the main thread and its tests can share.
 *
 * Why this is its own module: a Web Worker resolves a relative `fetch("dev-model/x.onnx")` against
 * its own script URL (`/_next/static/chunks/…`), NOT against the page — so a relative
 * `NEXT_PUBLIC_LAYA_MODEL_BASE=/dev-model` silently 404s every artifact. The main thread therefore
 * resolves the base against `location.href` before posting it into the worker, and this pure
 * function is the whole rule.
 */
export function resolveBase(base: string, pageHref: string): string {
  const b = String(base ?? "").trim().replace(/\/+$/, "");
  if (!b) return "";
  if (/^https?:\/\//i.test(b)) return b;
  // protocol-relative and other schemes are left alone; only path-relative input is resolved
  if (/^[a-z][a-z0-9+.-]*:/i.test(b)) return b;
  return new URL(b.startsWith("/") ? b : `/${b}`, pageHref).href;
}
