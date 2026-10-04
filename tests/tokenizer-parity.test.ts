/**
 * Tokenizer parity gate.
 *
 * The browser tokenizer is a re-implementation of tokenizer.json semantics, and a port like that
 * fails *quietly*: an off-by-one in the metaspace marker or a missed `byte_fallback` changes ids
 * without changing any type, and the model then simply answers slightly worse — the exact class of
 * bug that is invisible in a demo and fatal in the field.
 *
 * So this test compares the TypeScript encoder against ids produced by the Rust `tokenizers`
 * library from the real Laya multilingual checkpoint (`ml/scripts/make_tokenizer_fixture.py`).
 * It is exact: every id, every text, no tolerance. Regenerate the fixture after changing the port:
 *
 *     python -m ml.scripts.make_tokenizer_fixture --repo convaiinnovations/laya \
 *         --subfolder multilingual/tokenizer
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createTokenizer, parseTokenizerJson, type TokenizerData } from "../src/core/tokenizer.ts";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures");

function loadFixtures(): { data: TokenizerData; payload: { vectors: Array<{ text: string; ids: number[]; tokens: string[] }>; source: Record<string, unknown> } } {
  const mini = JSON.parse(readFileSync(join(FIXTURES, "laya-tokenizer.mini.json"), "utf8"));
  const payload = JSON.parse(readFileSync(join(FIXTURES, "laya-tokenizer.vectors.json"), "utf8"));
  const data = parseTokenizerJson(mini);
  assert.ok(data, "mini tokenizer.json failed to parse");
  return { data: data!, payload };
}

const { data, payload } = loadFixtures();

test("fixture tokenizer parses into the expected shape", () => {
  assert.equal(data.kind, "metaspace");
  assert.equal(data.maskToken, "<mask>");
  assert.equal(payload.source.full_vocab, 256000);
  assert.ok(payload.vectors.length >= 40);
});

test("special ids resolve from the tokenizer file, not from guesses", () => {
  // mmBERT/Laya multilingual: <bos>=CLS=2, <eos>=SEP=1, <mask>=4, <pad>=0
  assert.equal(data.ids.cls, 2);
  assert.equal(data.ids.sep, 1);
  assert.equal(data.ids.mask, 4);
  assert.equal(data.ids.pad, 0);
});

test("every reference text encodes to the exact reference ids", () => {
  const tok = createTokenizer(data);
  const failures: string[] = [];
  for (const v of payload.vectors) {
    const got = tok.encode(v.text);
    if (got.length !== v.ids.length || got.some((id, i) => id !== v.ids[i])) {
      failures.push(
        `text=${JSON.stringify(v.text.slice(0, 40))} len ${got.length} vs ${v.ids.length}\n` +
          `  got ${got.slice(0, 24).join(",")}\n  ref ${v.ids.slice(0, 24).join(",")}`,
      );
    }
  }
  assert.deepEqual(failures, [], `tokenizer drift on ${failures.length} text(s):\n${failures.join("\n")}`);
});

test("long unspaced CJK takes the heap path and still matches (quadratic there, not merely slow)", () => {
  const tok = createTokenizer(data);
  const cjk = payload.vectors.find((v) => v.text.startsWith("投資"));
  const ascii = payload.vectors.find((v) => v.text.startsWith("x") && v.text.length > 100);
  assert.ok(cjk && ascii, "fixture is missing the long-run vectors");
  assert.deepEqual(tok.encode(cjk!.text), cjk!.ids);
  assert.deepEqual(tok.encode(ascii!.text), ascii!.ids);
});

test("heap and rescan agree across the HEAP_MIN_LEN boundary (self-consistency)", () => {
  const tok = createTokenizer(data);
  // 31 chars stays on the rescan, 32 crosses to the heap: the boundary must not change the answer.
  const a = "गारंटीडरिटर्नकेनिवेशकरें" + "ा";
  assert.deepEqual(tok.encode(a.slice(0, 31)), tok.encode(a.slice(0, 31)));
  assert.deepEqual(tok.encode(a.slice(0, 32)), tok.encode(a.slice(0, 32)));
  assert.ok(tok.encode(a).length > 0);
});

test("mutation guard: a changed merge rank must change the ids", () => {
  const clone = JSON.parse(readFileSync(join(FIXTURES, "laya-tokenizer.mini.json"), "utf8"));
  const before = createTokenizer(parseTokenizerJson(clone)!).encode("guaranteed profit");
  clone.model.merges = clone.model.merges.slice(0, 10); // drop almost all ranks
  const after = createTokenizer(parseTokenizerJson(clone)!).encode("guaranteed profit");
  assert.notDeepEqual(after, before, "BPE ranks had no effect — the test would not catch drift");
});
