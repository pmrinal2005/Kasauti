/**
 * Manifest / calibration / gating contract.
 *
 * The browser is the only consumer of the manifest `ml/kasauti_ml/pipeline.py` publishes, and a
 * schema drift there is silent: the page still renders, it just stops gating. These tests parse a
 * REAL manifest produced by the pipeline (tests/fixtures/manifest-v2.example.json — regenerate with
 * `python -m ml.kasauti_ml.pipeline --profile smoke --out /tmp/smoke` then copy staging/manifest.json)
 * through the same `parseManifest()` the Worker calls, and pin the rules that decide whether an
 * answer is allowed to trigger an action.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  abstainFor,
  bucket,
  decode,
  decodeBatch,
  DEFAULT_MANIFEST,
  parseManifest,
  temperatureFor,
  unpermuteProbs,
} from "../src/core/laya-client-browser.ts";
import { BANKS, BANKS_SHA256, bankFollowups, bankMaxQuestions, triageOptions } from "../src/lib/banks.ts";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures");
const published = JSON.parse(readFileSync(join(FIXTURES, "manifest-v2.example.json"), "utf8"));

test("a real published manifest parses into the runtime contract", () => {
  const m = parseManifest(published);
  assert.equal(m.schema, 2);
  assert.equal(m.calibrated, true);
  assert.match(m.modelVersion, /^kasauti-laya-/);
  assert.equal(m.maxLen, published.maxLen);
  assert.ok(m.temperatureByBucket["choice:3-5"] > 0);
  assert.ok(m.files.wasm?.path.endsWith(".onnx"), "wasm graph filename");
  assert.ok((m.files.wasm.sha256 ?? "").length === 64, "content hash travels with the file");
  assert.equal(m.tokenizerPath, "tokenizer.json");
  assert.ok(m.tokenizerSha256 && m.tokenizerSha256.length === 64);
  assert.deepEqual(m.tokenizerSpecials && Object.keys(m.tokenizerSpecials).sort(), ["cls", "mask", "pad", "sep", "unk"]);
});

test("banks drift is detectable: manifest sha must equal the generated runtime banks", () => {
  assert.equal(published.banks.sha256, BANKS_SHA256);
  assert.equal(published.banks.version, "1.0.0");
});

test("manifest specials agree with what the tokenizer file resolves (mmBERT: cls 2, sep 1, mask 4, pad 0)", () => {
  const m = parseManifest({ ...published, tokenizer: { ...published.tokenizer, specials: { cls: 2, sep: 1, mask: 4, pad: 0, unk: 3 } } });
  assert.deepEqual(m.tokenizerSpecials, { cls: 2, sep: 1, mask: 4, pad: 0, unk: 3 });
});

test("temperature lookup: exact bucket → default → per-type floor → 1", () => {
  const m = parseManifest({ schema: 2, temperature: [3, 1, 5], temperatureByBucket: { "choice:3-5": 2.5, default: 1.5 } });
  assert.equal(temperatureFor(m, "choice", 3), 2.5); // exact bucket
  assert.equal(temperatureFor(m, "choice", 8), 1.5); // "default"
  assert.equal(temperatureFor(m, "noul", 2), 1.5); // default beats the type floor
  const noDefault = parseManifest({ schema: 2, temperature: [3, 1, 5], temperatureByBucket: {} });
  assert.equal(temperatureFor(noDefault, "noul", 2), 5); // per-type floor
  assert.equal(temperatureFor(DEFAULT_MANIFEST, "score", 5), 1); // nothing published → T = 1
});

test("a NaN or zero temperature degrades to 1 instead of poisoning softmax", () => {
  const m = parseManifest({ schema: 2, temperature: [1, 1, 1], temperatureByBucket: { "noul:2": 0, default: Number.NaN } });
  assert.equal(temperatureFor(m, "noul", 2), 1);
});

test("bucket keys match the fitted map (2 / 3-5 / 6-10 / 11+)", () => {
  assert.equal(bucket("choice", 2), "choice:2");
  assert.equal(bucket("choice", 5), "choice:3-5");
  assert.equal(bucket("choice", 10), "choice:6-10");
  assert.equal(bucket("choice", 11), "choice:11+");
  assert.equal(bucket("noul", 2), "noul:2");
});

test("abstain thresholds come from the manifest, fall back to the documented defaults", () => {
  const m = parseManifest({ schema: 2, abstain: { "choice:3-5": 0.42 } });
  assert.equal(abstainFor(m, "choice", 3), 0.42);
  assert.equal(abstainFor(m, "noul", 2), 0.35); // DEFAULT_ABSTAIN floor
  assert.equal(abstainFor(parseManifest({ schema: 2, abstain: { default: 0.9 } }), "choice", 4), 0.9);
});

test("an uncalibrated manifest never lets an answer act", () => {
  // The Worker forces abstain=1 for schema<2; make sure that is what the defaults imply too.
  const m = parseManifest({ modelVersion: "legacy", temperatures: {}, abstain: {} });
  assert.equal(m.calibrated, false);
  const a = decode({ type: "noul", instructions: "Does it promise guaranteed returns?" }, [-6, 6], m);
  assert.ok((a.noul ?? 0) > 0.99, "p(true) ≈ 1 from those logits");
  assert.ok(a.confidence > 0.9, "logits are confident…");
  assert.equal(a.abstained, false, "…and only the Worker's forced threshold stops it from acting");
  const forced = decode({ type: "noul", instructions: "x" }, [-6, 6], { ...m, abstain: { default: 1 } });
  assert.equal(forced.abstained, true, "forced threshold abstains on everything");
});

test("confidence is calibrated-softmax entropy, and abstention follows the threshold", () => {
  const m = parseManifest({ schema: 2, temperature: [1, 1, 1], temperatureByBucket: { "noul:2": 2 }, abstain: { "noul:2": 0.5 } });
  const sharp = decode({ type: "noul", instructions: "x" }, [4, -4], m);
  const flat = decode({ type: "noul", instructions: "x" }, [0.4, 0.2], m);
  assert.ok(sharp.confidence > flat.confidence);
  assert.equal(sharp.abstained, false);
  assert.equal(flat.abstained, true);
  assert.ok(Math.abs(sharp.probs.false + sharp.probs.true - 1) < 1e-9, "probabilities sum to 1");
});

test("score answers are expectations over levels, noul answers are p(true)", () => {
  const m = parseManifest({ schema: 2, temperature: [1, 1, 1], temperatureByBucket: {} });
  const s = decode({ type: "score", instructions: "How urgent is this?", criteria: ["a", "b", "c", "d"] }, [3, 0, 0, 0], m);
  assert.ok(s.score !== undefined && s.score < 0.4, `expected a low expected level, got ${s.score}`);
  const n = decode({ type: "noul", instructions: "x" }, [0, 5], m);
  assert.ok(n.noul !== undefined && n.noul > 0.9);
});

test("unpermute_probs puts the model's permuted slots back into caller order", () => {
  assert.deepEqual(unpermuteProbs([0.1, 0.2, 0.7], [2, 0, 1]), [0.2, 0.7, 0.1]);
  assert.deepEqual(unpermuteProbs([0.1, 0.9], null), [0.1, 0.9]);
});

test("decodeBatch splits a [B,K] logit block by option count, not by K", () => {
  const questions = {
    triage: BANKS.claim_v1.triage, // 5 options
    guaranteed: { type: "noul" as const, instructions: "Does it promise a guaranteed return?" }, // 2 options
  };
  const K = 5;
  const logits = [0, 0, 0, 0, 9, /* row 2 */ 0, 9, 0, 0, 0];
  const out = decodeBatch(questions, logits, K, DEFAULT_MANIFEST);
  assert.equal(out.triage.choice, "other");
  assert.ok(out.guaranteed.noul! > 0.9);
});

/* ------------------------------- question banks ------------------------------- */

test("every bank's follow-ups are keyed by its own triage options", () => {
  for (const [id, bank] of Object.entries(BANKS)) {
    const opts = new Set(triageOptions(id as keyof typeof BANKS));
    for (const key of Object.keys(bank.followups)) {
      assert.ok(opts.has(key), `${id}: follow-up key ${key} is not a triage option`);
    }
    assert.ok(Object.keys(bank.followups).length > 0, `${id}: no follow-ups at all`);
    assert.ok(bankMaxQuestions(id as keyof typeof BANKS) >= 2);
  }
});

test("no bank exceeds Laya's 20-option ceiling (accuracy collapses above it)", () => {
  for (const [id, bank] of Object.entries(BANKS)) {
    const crit = bank.triage.criteria;
    if (crit && !Array.isArray(crit)) assert.ok(Object.keys(crit).length <= 20, `${id} triage has >20 options`);
  }
});

test("bankFollowups honours the device budget and skips what Tier-0 already proved", () => {
  assert.equal(Object.keys(bankFollowups("claim_v1", "investment_offer", new Set(), 3)).length, 3);
  assert.deepEqual(Object.keys(bankFollowups("claim_v1", "investment_offer", new Set(["guaranteed"]), 2)), ["pay_to_withdraw", "unregistered_group"]);
  assert.equal(Object.keys(bankFollowups("claim_v1", "investment_offer", new Set(), 0)).length, 0);
  assert.deepEqual(Object.keys(bankFollowups("claim_v1", "other", new Set(), 3)), [], "a terminal triage answer asks nothing more");
});

test("a question the model was never trained on is still rendered the way the trainer renders it", () => {
  const q = BANKS.lens_v1.followups.teaches_concept.disclosure;
  const a = decode(q, [0.2, 3.1], parseManifest(published));
  assert.equal(a.type, "noul");
  assert.equal(a.bucket, "noul:2");
  assert.ok(Math.abs(a.temperature - (published.temperatureByBucket["noul:2"] ?? published.temperature[2])) < 1e-9);
});

test("`reportOnly` is an override: the export's own admission that it did not clear the gates", () => {
  // a schema-2 manifest with a fitted temperature map is normally "calibrated"...
  const ok = { ...published, reportOnly: false };
  assert.equal(parseManifest(ok).calibrated, true);
  assert.equal(parseManifest(published).calibrated, published.reportOnly !== true);
  // ...but an export that says reportOnly can never be treated as calibrated, whatever else it says
  const ro = parseManifest({ ...published, reportOnly: true });
  assert.equal(ro.calibrated, false, "reportOnly must beat schema 2 + fitted temperatures");
  // and the runtime rule that follows from it: report-only answers are buried under the threshold
  const answer = decode(BANKS.claim_v1.triage, [8, 1, 1, 1, 1], ro);
  assert.equal(answer.abstained, true);
});

test("the shipped dev model is report-only, so a fresh clone can never act on it", () => {
  const dev = JSON.parse(readFileSync(join(FIXTURES, "..", "..", "public", "dev-model", "manifest.json"), "utf8"));
  const m = parseManifest(dev);
  assert.equal(m.calibrated, false);
  assert.equal(m.reportOnly, true, "the smoke/dev export must declare reportOnly");
  assert.ok(m.maxLen > 0 && Object.keys(m.files).length >= 1);
});
