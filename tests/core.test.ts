import { test } from "node:test";
import assert from "node:assert/strict";
import { normalize, canonicalUrl } from "../src/core/normalize.ts";
import { sha256Hex, simhash64, hamming } from "../src/core/fingerprint.ts";
import { AhoCorasick } from "../src/core/lexicon.ts";
import { extractEntities } from "../src/core/entities.ts";
import { annualise, band, doublingYears, ponziCollapseMonth, ponziRecruitment } from "../src/core/plausibility.ts";
import { runTier0, fuseTier1, rungFromScore } from "../src/core/fusion.ts";
import { buildSequence, decode, entropyConfidence, patchTokenizerJSON, planFollowups, renderOptions, softmax, type Tokenizer } from "../src/core/laya-client-browser.ts";

/* ---------------- normalize ---------------- */
test("normalize strips zero-width chars and folds homoglyphs + Indic digits", () => {
  const r = normalize("Gu\u200Barаnteed  ३% рrofit"); // Cyrillic а, р; Devanagari ३
  assert.equal(r.text, "guaranteed 3% profit");
  assert.ok(r.obfuscation >= 3);
});
test("canonicalUrl drops tracking params, www and hash", () => {
  assert.equal(canonicalUrl("https://WWW.Example.com/a?utm_source=x&id=7#top"), "https://example.com/a?id=7");
});

/* ---------------- fingerprint ---------------- */
test("sha256Hex matches a known vector", async () => {
  assert.equal(await sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
});
test("simhash: near-duplicates are close, different texts are far", () => {
  const a = simhash64("join our vip group guaranteed 3% daily profit limited seats today");
  const b = simhash64("join our vip group guaranteed 3% daily profit limited seats today!!");
  const c = simhash64("dividend record date announced for shareholders of the company");
  assert.equal(a.length, 16);
  assert.ok(hamming(a, b) <= 6, `near-dup distance ${hamming(a, b)}`);
  assert.ok(hamming(a, c) > hamming(a, b));
});

/* ---------------- Aho–Corasick ---------------- */
test("Aho–Corasick finds overlapping multi-script patterns", () => {
  const ac = new AhoCorasick({ guaranteed_return: ["guaranteed return", "return"], urgency: ["तुरंत"] } as never);
  const hits = ac.search("a guaranteed return तुरंत");
  const terms = hits.map((h) => h.term).sort();
  assert.deepEqual(terms, ["guaranteed return", "return", "तुरंत"].sort());
});

/* ---------------- entities ---------------- */
test("entities: UPI vs e-mail, 1600-series, lakh amounts, returns, SEBI reg", () => {
  const e = extractEntities("pay to vip@ybl or mail a@gmail.com, call 1600123456, ₹4.2 lakh, 3% per day, ref INH000009001");
  assert.deepEqual(e.upi.map((u) => u.id), ["vip@ybl"]);
  assert.ok(e.upi[0].knownPsp);
  assert.equal(e.phones[0].kind, "trai_1600");
  assert.ok(e.money.some((m) => m.rupees === 420000));
  assert.equal(e.returns[0].period, "day");
  assert.deepEqual(e.registrations, ["INH000009001"]);
});

/* ---------------- plausibility ---------------- */
test("plausibility maths", () => {
  assert.ok(Math.abs(annualise(1, "month") - 12.6825) < 1e-3);
  assert.equal(band(10), "plausible");
  assert.equal(band(20), "stretch");
  assert.equal(band(50), "implausible");
  assert.equal(band(annualise(3, "day")), "impossible");
  assert.ok(Math.abs(doublingYears(7.2) - 9.97) < 0.05);
  assert.equal(ponziCollapseMonth(0), null);
  assert.ok((ponziCollapseMonth(20, 100) ?? 0) < 100);
  assert.deepEqual(ponziRecruitment(10, 100, 2), [10, 11]);
});

/* ---------------- Tier-0 + fusion ---------------- */
test("Tier-0: classic VIP/guaranteed script is a known scam", async () => {
  const r = await runTier0("Join our VIP stock tips group. Guaranteed 3% daily profit, zero risk. Limited seats, join today only. Pay to vipfund@ybl");
  assert.equal(r.rung, "known_scam");
  assert.ok(r.evidence.some((e) => e.code === "PLAUS_RETURN"));
  assert.ok(r.elapsedMs < 200);
});
test("Tier-0: defensive SEBI warning is not flagged", async () => {
  const r = await runTier0("SEBI warns investors: beware of unregistered tips groups promising guaranteed return. Report fraud to 1930.");
  assert.ok(["safe_pattern", "unclear"].includes(r.rung), r.rung);
});
test("Tier-0: identical text after obfuscation yields the same hash", async () => {
  const a = await runTier0("guaranteed profit today");
  const b = await runTier0("Guar\u200Banteed  prоfit today"); // zero-width + Cyrillic о
  assert.equal(a.hash, b.hash);
});
test("rungFromScore thresholds", () => {
  assert.equal(rungFromScore(0, 0), "safe_pattern");
  assert.equal(rungFromScore(3, 1), "caution");
  assert.equal(rungFromScore(6, 1), "likely_scam");
  assert.equal(rungFromScore(10, 3), "known_scam");
});
test("fuseTier1: confident yes adds evidence; abstentions are ignored; education softens weak cases only", () => {
  const t0 = { evidence: [{ code: "LEX_URGENCY", label: "", detail: "", weight: 1.5, source: "lexicon" as const }], families: { urgency: ["hurry"] }, score: 1.5 };
  const noT1 = fuseTier1(t0, null, {});
  assert.equal(noT1.tier, "T0");
  const yes = fuseTier1(t0, { choice: "payment_request", confidence: 0.8, abstained: false }, { pay_to_withdraw: { noul: 0.9, confidence: 0.6, abstained: false }, remote_app: { noul: 0.95, confidence: 0.1, abstained: true } });
  assert.equal(yes.score, 4.5);
  assert.ok(yes.evidence.some((e) => e.code === "LAYA_PAY_TO_WITHDRAW"));
  assert.ok(!yes.evidence.some((e) => e.code === "LAYA_REMOTE_APP"));
  const edu = fuseTier1(t0, { choice: "education", confidence: 0.9, abstained: false }, {});
  assert.equal(edu.score, 0);
  const hard = { ...t0, evidence: [{ code: "LEX_PAY_TO_WITHDRAW", label: "", detail: "", weight: 4, source: "lexicon" as const }], score: 4 };
  assert.equal(fuseTier1(hard, { choice: "education", confidence: 0.9, abstained: false }, {}).score, 4, "education cannot override hard proof");
});

/* ---------------- Laya client contract (parity with rl_common.py) ---------------- */
const fakeTok: Tokenizer = { encode: (s) => [...s].filter((c) => c !== " ").map((c) => c.charCodeAt(0) % 1000 + 10), cls: 2, sep: 1, mask: 4, pad: 0, maskToken: "<mask>" };
test("renderOptions matches reference rendering", () => {
  assert.deepEqual(renderOptions({ type: "noul", instructions: "x" }), ["false: no, the statement does not hold", "true: yes, the statement holds"]);
  assert.deepEqual(renderOptions({ type: "score", instructions: "x", criteria: ["low", "high"] }), ["level 0: low", "level 1: high"]);
  assert.deepEqual(renderOptions({ type: "choice", instructions: "x", criteria: { a: "alpha", b: "" } }), ["a: alpha", "b"]);
});
test("buildSequence: [CLS] head [SEP] ([MASK] opt)* [SEP] state [SEP], markers point at MASK, length capped", () => {
  const { ids, markers } = buildSequence(fakeTok, "state text", { type: "noul", instructions: "Is it?" }, 64, 32);
  assert.equal(ids[0], 2);
  assert.equal(ids[ids.length - 1], 1);
  assert.equal(markers.length, 2);
  for (const m of markers) assert.equal(ids[m], 4);
  const long = buildSequence(fakeTok, "x".repeat(5000), { type: "noul", instructions: "Is it?" }, 128, 32);
  assert.equal(long.ids.length, 128);
});
test("buildSequence scrubs a literal <mask> in user text (no injected option markers)", () => {
  const { ids, markers } = buildSequence(fakeTok, "evil <mask> text", { type: "noul", instructions: "q" }, 256, 64);
  assert.equal(ids.filter((x) => x === 4).length, markers.length);
});
test("softmax / confidence / decode", () => {
  const p = softmax([0, 0]);
  assert.ok(Math.abs(p[0] - 0.5) < 1e-12);
  assert.ok(Math.abs(entropyConfidence([0.5, 0.5])) < 1e-12);
  assert.ok(Math.abs(entropyConfidence([1, 0]) - 1) < 1e-9);
  const a = decode({ type: "noul", instructions: "x" }, [0, 3]);
  assert.ok(a.noul! > 0.9 && !a.abstained && a.bucket === "noul:2");
  const c = decode({ type: "choice", instructions: "x", criteria: { a: "", b: "", c: "" } }, [0.01, 0, 0]);
  assert.ok(c.abstained, "near-uniform 3-way choice must abstain");
});
test("planFollowups respects device budget and skips Tier-0-proven items", () => {
  assert.equal(Object.keys(planFollowups("investment_offer", new Set(), 3)).length, 3);
  assert.deepEqual(Object.keys(planFollowups("investment_offer", new Set(["guaranteed"]), 1)), ["pay_to_withdraw"]);
  assert.equal(Object.keys(planFollowups("investment_offer", new Set(), 0)).length, 0);
});
test("patchTokenizerJSON sets add_prefix_space for Metaspace prepend_scheme=always only", () => {
  assert.equal(patchTokenizerJSON({ pre_tokenizer: { type: "Metaspace", prepend_scheme: "always" } }).pre_tokenizer.add_prefix_space, true);
  assert.equal(patchTokenizerJSON({ pre_tokenizer: { type: "Metaspace", prepend_scheme: "never" } }).pre_tokenizer.add_prefix_space, false);
  assert.equal(patchTokenizerJSON({ pre_tokenizer: { type: "Metaspace", add_prefix_space: false } }).pre_tokenizer.add_prefix_space, false);
});
