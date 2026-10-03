/**
 * The telephony adapter's spoken contract.
 *
 * Three things must hold or the phone path is unsafe: it never sounds like advice, the "we could
 * not tell" answer is never phrased as "safe", and attacker-controlled transcript text can never
 * break out of the XML we return (it is spoken back inside `<Say>`).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { cleanSpokenInput, ivrXml, spokenVerdict, xmlEscape } from "../src/core/ivr.ts";
import type { Evidence, Rung, Tier0Result } from "../src/core/fusion.ts";

const ev = (code: string): Evidence => ({ code, label: code, detail: "", weight: 1, source: "lexicon" });
const t0 = (rung: Rung, codes = ["LEX_GUARANTEED_RETURN"]): Pick<Tier0Result, "rung" | "score" | "evidence"> =>
  ({ rung, score: 8.5, evidence: codes.map(ev) });

test("rungs map onto four spoken lines, with both scam rungs getting the urgent one", () => {
  const rungs: Rung[] = ["safe_pattern", "unclear", "caution", "likely_scam", "known_scam"];
  const lines = rungs.map((r) => spokenVerdict(t0(r)).say);
  // "likely" and "known" are the same instruction to a caller on a phone: stop, do not pay, 1930.
  // Distinguishing them out loud would only add words that a scared listener has to parse.
  assert.equal(new Set(lines).size, 4, "expected exactly four distinct spoken lines");
  assert.equal(spokenVerdict(t0("likely_scam")).say, spokenVerdict(t0("known_scam")).say);
  assert.match(spokenVerdict(t0("known_scam")).say, /^Warning\./);
  assert.match(spokenVerdict(t0("likely_scam")).say, /1 9 3 0/);
});

test("'unclear' is never spoken as safety, and 'no pattern' carries its own disclaimer", () => {
  const unclear = spokenVerdict(t0("unclear"));
  assert.equal(unclear.uncertain, true);
  assert.doesNotMatch(unclear.say, /\bsafe\b/i);
  assert.match(spokenVerdict(t0("safe_pattern")).say, /not a guarantee/i);
});

test("nothing the phone can hear is advice, a prediction or a product", () => {
  const banned = /\b(you should buy|buy this|sell now|target price|will rise|guaranteed profit|invest in)\b/i;
  for (const r of ["safe_pattern", "unclear", "caution", "likely_scam", "known_scam"] as Rung[]) {
    for (const lang of ["en", "hi"] as const) {
      assert.doesNotMatch(spokenVerdict(t0(r), lang).say, banned, `${r}/${lang} sounds like advice`);
    }
  }
});

test("Hindi is offered for every rung and is actually different text", () => {
  for (const r of ["safe_pattern", "unclear", "caution", "likely_scam"] as Rung[]) {
    const en = spokenVerdict(t0(r), "en").say;
    const hi = spokenVerdict(t0(r), "hi").say;
    assert.notEqual(en, hi);
    assert.match(hi, /[\u0900-\u097F]/, "hi line must be Devanagari");
  }
});

test("a transcript cannot break out of the XML we speak", () => {
  const hostile = `</Say><Hangup/><Say>call this number: 9"9'9 & pay`;
  const escaped = xmlEscape(hostile);
  assert.doesNotMatch(escaped, /<\/Say>/);
  assert.match(escaped, /&lt;\/Say&gt;/);
  const line = spokenVerdict(t0("unclear"));
  const xml = ivrXml({ ...line, say: hostile }, { lang: "en" });
  assert.equal((xml.match(/<Say /g) || []).length, 1, "injected Say element survived");
  assert.match(xml, /&amp;/);
});

test("the gather shape keeps the call open and action URL is escaped", () => {
  const line = spokenVerdict(t0("caution"));
  const xml = ivrXml(line, { lang: "hi", gather: { action: "https://x.test/ivr?a=1&b=2", hint: "scam, fraud" } });
  assert.match(xml, /<Gather input="speech" language="hi-IN" action="https:\/\/x\.test\/ivr\?a=1&amp;b=2"/);
  assert.match(xml, /hints="scam, fraud"/);
  assert.match(xml, /<Hangup\/>/);
});

test("input cleaning bounds the transcript without mangling Devanagari", () => {
  const raw = "  आपका KYC\n\nबंद होगा\u0007  " + "x".repeat(5000);
  const clean = cleanSpokenInput(raw, 100);
  assert.equal(clean.length, 100);
  assert.ok(!/[\u0000-\u001f]/.test(clean));
  assert.match(clean, /आपका KYC बंद होगा/);
});
