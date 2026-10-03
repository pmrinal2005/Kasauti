/**
 * `resolveBase` — the rule that keeps a relative model base from 404-ing inside the Worker.
 *
 * This was a real bug: with NEXT_PUBLIC_LAYA_MODEL_BASE=/dev-model the worker resolved
 * "dev-model/manifest.json" against /_next/static/chunks/… and reported "unpublished" forever.
 * A Worker cannot see the page URL, so the main thread resolves it first — and the rule is pinned
 * here so a future refactor cannot quietly drop it.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { resolveBase } from "../src/lib/url.ts";

const PAGE = "http://localhost:3100/dashboard";

test("absolute model bases pass through untouched (HF is the production path)", () => {
  const hf = "https://huggingface.co/kasauti/laya-multilingual-onnx/resolve/main";
  assert.equal(resolveBase(hf, PAGE), hf);
  assert.equal(resolveBase("https://example.test/models/", PAGE), "https://example.test/models");
  assert.equal(resolveBase("HTTP://EXAMPLE.TEST/a", PAGE), "HTTP://EXAMPLE.TEST/a");
});

test("a relative base resolves against the page, not the worker chunk", () => {
  assert.equal(resolveBase("/dev-model", PAGE), "http://localhost:3100/dev-model");
  assert.equal(resolveBase("dev-model", PAGE), "http://localhost:3100/dev-model");
  assert.equal(resolveBase("dev-model/", PAGE), "http://localhost:3100/dev-model");
  assert.equal(resolveBase("/models/v2/", "https://kasauti.example/dashboard#engine"), "https://kasauti.example/models/v2");
});

test("non-http schemes are not mangled into file paths", () => {
  // a scheme we don't understand is returned verbatim rather than being turned into "/data:…"
  assert.equal(resolveBase("data:application/json,{}", PAGE), "data:application/json,{}");
  // protocol-relative DOES get resolved — against the page's scheme, which is what "//" means
  assert.equal(resolveBase("//cdn.example/models", PAGE), "http://cdn.example/models");
  assert.equal(resolveBase("//cdn.example/models", "https://kasauti.example/dashboard"), "https://cdn.example/models");
});

test("an empty base never throws (unset env var)", () => {
  assert.equal(resolveBase("", PAGE), "");
  assert.equal(resolveBase("   ", PAGE), "");
});

test("the result is always usable as a fetch() prefix", () => {
  for (const raw of ["/dev-model", "dev-model", "https://hf.example/x/"]) {
    const out = resolveBase(raw, PAGE);
    assert.ok(!out.endsWith("/"), `${raw} → ${out} would produce a double slash`);
    assert.match(new URL(`${out}/manifest.json`).protocol, /^https?:$/);
  }
});
