/**
 * Hash routing is the app's only router, and it failed silently once: `hydrate()` kept its own list
 * of valid views, so opening /dashboard#lens in a fresh tab fell back to Overview with no error
 * anywhere. These tests pin the two things that were wrong — the list, and the fallback.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { VIEW_IDS, isView } from "../src/lib/store.ts";

test("every navigable view is registered, including the ones added later", () => {
  for (const v of ["overview", "checker", "lens", "voice", "engine", "registry", "privacy"]) {
    assert.ok(VIEW_IDS.includes(v as never), `${v} is not routable`);
    assert.ok(isView(v), `#${v} would fall back to overview`);
  }
});

test("unknown or hostile hashes fall back to overview rather than rendering nothing", () => {
  for (const bad of ["", "nope", "constructor", "__proto__", "toString", "#lens", " "]) {
    assert.equal(isView(bad), false, `#${bad} must not be treated as a view`);
  }
});

test("the registry is not polluted by prototype keys", () => {
  // hasOwnProperty guards against `#toString` resolving through the prototype chain
  assert.equal(Object.keys(VIEW_IDS).length, VIEW_IDS.length);
  assert.equal(new Set(VIEW_IDS).size, VIEW_IDS.length, "duplicate view ids");
});
