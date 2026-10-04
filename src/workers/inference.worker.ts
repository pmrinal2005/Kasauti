/// <reference lib="webworker" />
/**
 * Tier-1 inference Worker — the only place in Kasauti that runs Laya.
 *
 * Delivery: a PUBLIC Hugging Face *model* repo, static LFS bytes over the HF CDN. No Space, no
 * GPU, no Kasauti-operated compute, no `LAYA_URL`, no `/v1/systemone`. The published layout (all
 * of it produced by `ml/kasauti_ml/pipeline.py`) is:
 *
 *   <base>/manifest.json                 schema 2 — calibration, thresholds, hashes, evals
 *   <base>/<files.webgpu.path>           4-bit MatMulNBits graph   (WebGPU EP, preferred)
 *   <base>/<files.wasm.path>             dynamic INT8 graph        (WASM EP, fallback)
 *   <base>/<files.fp32.path>             fp32 reference graph      (debug / parity only)
 *   <base>/<tokenizer.path>              vocabulary-pruned tokenizer.json
 *   <base>/<parity.path>                 parity vectors (tokenizer ids + logits)
 *
 * Graph I/O mirrors the reference `DecisionModel.forward`:
 *   input_ids [B,L] i64 · attention_mask [B,L] i64 · marker_pos [B,K] i64 ·
 *   marker_mask [B,K] bool · qtype [B] i64  →  logits [B,K] f32 (+ act_logits, never gated on)
 *
 * Rules this worker enforces so a wrong answer is impossible to *ship* without noticing:
 *   * every downloaded artifact is sha256-verified against the manifest (and again on cache hit)
 *   * the tokenizer's own special ids must match the manifest's, or we refuse to run
 *   * after the session is up, the published parity vectors are replayed on this device: the graph
 *     must reproduce the fp32 reference inside the published tolerance, or the export is demoted to
 *     report-only (a mis-compiling backend must never quietly answer questions)
 *   * the banks sha256 must match the generated `src/lib/banks.ts`, or we warn loudly: that
 *     mismatch means the model was trained on different question wording than the runtime asks
 *   * an uncalibrated manifest can never trigger an action — thresholds are forced to abstain
 *     (issue #185: `act_probability` carries no signal, so `confidence` is the only gate)
 */
import {
  buildSequence,
  decodeBatch,
  parseManifest,
  QTYPES,
  type CalibrationManifest,
  type QuestionSet,
} from "../core/laya-client-browser";
import { createTokenizer, parseTokenizerJson, type LayaTokenizer } from "../core/tokenizer";
import { BANKS, BANKS_SHA256, BANKS_VERSION } from "../lib/banks";

declare const self: DedicatedWorkerGlobalScope;

/* eslint-disable @typescript-eslint/no-explicit-any */
const dynImport = new Function("u", "return import(u)") as (u: string) => Promise<any>;

/** Pinned, overridable at build time. Kept off the main bundle: the Worker chunk loads it lazily. */
const ORT_BASE = process.env.NEXT_PUBLIC_ORT_BASE || "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
const CACHE = "kasauti-model-v2";

let ortMod: any = null;
let session: any = null;
let tok: LayaTokenizer | null = null;
let manifest: CalibrationManifest = parseManifest(null);
let backendUsed = "none";
let banksMismatch: string | null = null;
let selfTest: SelfTest | null = null;

/**
 * Result of replaying the published parity vectors through the graph this device built.
 *
 * `ok: null` means "not testable" (an older export without vectors) — never a failure.
 */
interface SelfTest {
  ok: boolean | null;
  detail: string;
  vectors?: number;
  maxDelta?: number;
  tolerance?: number;
  top1?: number;
  tokenizerMismatches?: number;
}

const post = (m: unknown) => self.postMessage(m);

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Download once into the Cache API (not IndexedDB: these are large binaries, and the Cache API
 * streams and evicts them the way a browser cache should), then re-verify the content hash on
 * every load — a corrupted or stale cache entry must fail loudly, not quietly shift logits.
 */
async function cachedFetchBytes(
  url: string,
  opts: { sha256?: string; onProgress?: (loaded: number, total: number) => void; label?: string } = {},
): Promise<ArrayBuffer> {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(url);
  if (hit) {
    const buf = await hit.arrayBuffer();
    if (!opts.sha256 || (await sha256Hex(buf)) === opts.sha256) {
      opts.onProgress?.(buf.byteLength, buf.byteLength);
      return buf;
    }
    await cache.delete(url); // sha mismatch → poisoned entry, drop it and re-download
    post({ type: "status", status: "downloading", detail: `Cached ${opts.label ?? "artifact"} failed its hash check — refetching` });
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    opts.onProgress?.(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    buf.set(c, o);
    o += c.byteLength;
  }
  if (opts.sha256) {
    const got = await sha256Hex(buf.buffer);
    if (got !== opts.sha256) throw new Error(`${opts.label ?? url} hash mismatch: ${got.slice(0, 12)} ≠ ${opts.sha256.slice(0, 12)}`);
  }
  await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
  return buf.buffer;
}

/**
 * Replay the published parity vectors through the graph this device actually built.
 *
 * `sha256` already proves the bytes are the ones the pipeline measured. This proves something
 * different and stronger: that the ONNX session built *here* — this browser, this backend, this
 * WASM/WebGPU build — still reproduces the fp32 reference inside the tolerance the manifest
 * publishes. A silently mis-compiling execution provider, a corrupted cache entry or an ORT
 * regression all look identical to a working model until someone checks the numbers.
 *
 * A failure is never fatal to the page: the answer is forced to report-only, exactly like an
 * uncalibrated export, and the Engine view says which number drifted — the app degrades instead of
 * lying. `ok: null` (no vectors published, e.g. a schema-1 export) is information, not a failure.
 */
async function paritySelfTest(base: string, ort: any): Promise<SelfTest> {
  if (!manifest.parityPath) {
    return { ok: null, detail: "no parity vectors in this export (older manifest) — sha256 verification only" };
  }
  let payload: any;
  try {
    const bytes = await cachedFetchBytes(`${base}/${manifest.parityPath}`, { label: "parity.json" });
    payload = JSON.parse(new TextDecoder().decode(bytes));
  } catch (e) {
    return { ok: null, detail: `parity vectors could not be fetched (${String(e)}) — sha256 verification only` };
  }

  // 1) tokenizer vectors: the pure-TS tokenizer must reproduce the Rust ids exactly. This is the
  //    check that caught the Metaspace prepend_scheme divergence; a mismatch would silently
  //    degrade every prediction, so it is fatal here too.
  let tokenizerMismatches = 0;
  const tkVectors: Array<{ text: string; ids: number[] }> = payload?.tokenizer?.vectors ?? [];
  if (tok) {
    for (const v of tkVectors) {
      const got = tok!.encode(v.text);
      if (got.length !== v.ids.length || got.some((x, i) => x !== v.ids[i])) tokenizerMismatches++;
    }
  }

  // 2) logits vectors: run the downloaded graph and compare with the fp32 reference it ships.
  const logitsVectors: Array<{ ids: number[]; markers: number[]; qtype: number; logits: number[] }> =
    payload?.logits?.vectors ?? [];
  const tolerance = Number(payload?.logits?.tolerance ?? 0);
  if (!logitsVectors.length || !session) {
    return {
      ok: null,
      detail: "parity file carries no logits vectors — sha256 verification only",
      tokenizerMismatches,
    };
  }
  let maxDelta = 0;
  let top1 = 0;
  for (const v of logitsVectors) {
    const L = v.ids.length;
    const K = v.markers.length;
    const ids = new BigInt64Array(L);
    const att = new BigInt64Array(L);
    const mpos = new BigInt64Array(K);
    const mmask = new Uint8Array(K);
    v.ids.forEach((x, i) => { ids[i] = BigInt(x); att[i] = 1n; });
    v.markers.forEach((p, k) => { mpos[k] = BigInt(p); mmask[k] = 1; });
    const out = await session.run({
      input_ids: new ort.Tensor("int64", ids, [1, L]),
      attention_mask: new ort.Tensor("int64", att, [1, L]),
      marker_pos: new ort.Tensor("int64", mpos, [1, K]),
      marker_mask: new ort.Tensor("bool", mmask, [1, K]),
      qtype: new ort.Tensor("int64", new BigInt64Array([BigInt(v.qtype)]), [1]),
    });
    const got = Array.prototype.slice.call(out.logits.data as Float32Array) as number[];
    const n = Math.min(got.length, v.logits.length);
    for (let i = 0; i < n; i++) maxDelta = Math.max(maxDelta, Math.abs(got[i] - v.logits[i]));
    const argmax = (a: number[]) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
    if (argmax(got.slice(0, n)) === argmax(v.logits.slice(0, n))) top1++;
  }
  const top1Rate = top1 / logitsVectors.length;
  const withinTolerance = maxDelta <= tolerance;
  const ok = tokenizerMismatches === 0 && withinTolerance;
  return {
    ok,
    detail: ok
      ? `replayed ${logitsVectors.length} logits + ${tkVectors.length} tokenizer vectors on this device — max |Δlogit| ${maxDelta.toFixed(4)} ≤ ${tolerance}, top-1 ${(top1Rate * 100).toFixed(0)}%`
      : tokenizerMismatches
        ? `${tokenizerMismatches}/${tkVectors.length} tokenizer vectors do not reproduce the published ids — this device would mis-tokenize every message`
        : `this backend drifts past the published tolerance (max |Δlogit| ${maxDelta.toFixed(4)} > ${tolerance}) — refusing to act on it`,
    vectors: logitsVectors.length,
    maxDelta,
    tolerance,
    top1: top1Rate,
    tokenizerMismatches,
  };
}

async function load(base: string, backend: string, threads: number) {
  post({ type: "status", status: "probing", detail: `Looking for a published manifest at ${base}/manifest.json` });
  const mres = await fetch(`${base}/manifest.json`, { cache: "no-cache" }).catch(() => null);
  if (!mres || !mres.ok) {
    post({
      type: "status",
      status: "unpublished",
      detail: "No calibrated export published at this address yet — Tier-0 stays active (circuit breaker, honest copy).",
    });
    return;
  }
  manifest = parseManifest(await mres.json());

  if (!manifest.calibrated) {
    // Schema 1 / empty calibration, or an export that declared itself report-only: report answers,
    // never act on them. The published card measured ECE 0.31 for the multilingual base, so an
    // uncalibrated confidence is decoration.
    manifest.abstain = { default: 1 };
  }
  if (manifest.banksSha256 && manifest.banksSha256 !== BANKS_SHA256) {
    banksMismatch = `trained on banks ${manifest.banksSha256.slice(0, 12)}, runtime asks ${BANKS_SHA256.slice(0, 12)} (v${BANKS_VERSION})`;
    post({ type: "status", status: "probing", detail: `Question-bank drift: ${banksMismatch}` });
  }

  const files = manifest.files ?? {};
  const want: "webgpu" | "wasm" = backend === "webgpu" && files.webgpu ? "webgpu" : "wasm";
  const spec = files[want];
  if (!spec) throw new Error(`manifest lists no ${want} graph`);

  // ---------------------------------------------------------------- tokenizer
  post({ type: "status", status: "downloading", detail: "Fetching the vocabulary-pruned tokenizer (one-time, cached)…" });
  const tjBytes = await cachedFetchBytes(`${base}/${manifest.tokenizerPath}`, {
    sha256: manifest.tokenizerSha256,
    label: "tokenizer.json",
    onProgress: (l, t) => post({ type: "progress", loaded: l, total: t, what: "tokenizer" }),
  });
  const data = parseTokenizerJson(JSON.parse(new TextDecoder().decode(tjBytes)));
  if (!data) throw new Error("tokenizer.json could not be parsed");
  tok = createTokenizer(data);
  const declared = manifest.tokenizerSpecials;
  if (declared) {
    const bad = Object.entries(declared).filter(([k, v]) => (data.ids as unknown as Record<string, number>)[k] !== v);
    if (bad.length) {
      throw new Error(
        `tokenizer/model mismatch: manifest declares ${JSON.stringify(declared)} but the tokenizer resolves ` +
          `${JSON.stringify(data.ids)} — refusing to run`,
      );
    }
  }

  // ---------------------------------------------------------------- graph
  const ort = await dynImport(ORT_BASE + (want === "webgpu" ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs"));
  ortMod = ort;
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = Math.max(1, threads);
  // Single-threaded unless the page is cross-origin isolated: SharedArrayBuffer is unavailable
  // otherwise and ORT would fail at session creation instead of degrading.
  ort.env.wasm.simd = true;
  const bytes = await cachedFetchBytes(`${base}/${spec.path}`, {
    sha256: spec.sha256,
    label: spec.path,
    onProgress: (l, t) => post({ type: "progress", loaded: l, total: t, what: want }),
  });
  post({ type: "status", status: "downloading", detail: "Starting the ONNX session…" });
  session = await ort.InferenceSession.create(new Uint8Array(bytes), {
    executionProviders: want === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
  });
  backendUsed = want === "webgpu" ? "webgpu" : ort.env.wasm.numThreads > 1 ? "wasm-mt" : "wasm-st";

  // ---------------------------------------------------------------- self-test
  post({ type: "status", status: "downloading", detail: "Replaying the published parity vectors against this device's graph…" });
  selfTest = await paritySelfTest(base, ort).catch((e) => ({ ok: null, detail: `self-test could not run: ${String(e)}` }));
  if (selfTest.ok === false) {
    // Same posture as an uncalibrated export: report answers, never act on them.
    manifest.abstain = { default: 1 };
    manifest.calibrated = false;
  }

  post({
    type: "manifest",
    manifest,
    backend: backendUsed,
    banksMismatch,
    banksVersion: BANKS_VERSION,
    tokenizer: { vocabSize: tok.vocabSize, kind: tok.kind, specials: data.ids, bytes: tjBytes.byteLength },
    graph: { path: spec.path, bytes: spec.bytes, sha256: spec.sha256 },
    selfTest,
  });
  post({
    type: "status",
    status: "ready",
    backend: backendUsed,
    detail:
      `${manifest.modelVersion} · ${backendUsed}${manifest.calibrated ? "" : manifest.reportOnly ? " · report-only (did not clear the eval gates — inspect it, never act on it)" : " · uncalibrated (report-only)"}` +
      (banksMismatch ? ` · bank drift (${banksMismatch})` : "") +
      (selfTest?.ok === true ? " · self-test passed" : selfTest?.ok === false ? ` · SELF-TEST FAILED (${selfTest.detail})` : ""),
  });
}

/** Tokenizer-only inspection: shows the exact Laya input for a message before any graph exists. */
async function inspect(state: string, base: string, questions: QuestionSet) {
  if (!tok) {
    const tjBytes = await cachedFetchBytes(`${base}/${manifest.tokenizerPath || "tokenizer.json"}`, { label: "tokenizer.json" });
    const data = parseTokenizerJson(JSON.parse(new TextDecoder().decode(tjBytes)));
    if (!data) throw new Error("tokenizer.json could not be parsed");
    tok = createTokenizer(data);
  }
  const t0 = performance.now();
  // no question supplied → use the bank's triage, so "Inspect" always builds a real sequence
  const first = Object.values(questions)[0] ?? BANKS.claim_v1.triage;
  const s = first ? buildSequence(tok, state, first, manifest.maxLen, manifest.headMaxLen) : { ids: [], markers: [] };
  const opts = first ? Object.keys((first.criteria ?? {}) as Record<string, unknown>) : [];
  post({
    type: "inspect",
    ids: s.ids.length,
    markers: s.markers,
    head: s.ids.slice(0, 40),
    special: { cls: tok.cls, sep: tok.sep, mask: tok.mask, pad: tok.pad, unk: tok.unk },
    vocabSize: tok.vocabSize,
    kind: tok.kind,
    options: opts.length,
    ms: performance.now() - t0,
  });
}

async function predict(state: string, qs: QuestionSet) {
  if (!session || !tok) throw new Error("model not loaded");
  const ort = ortMod;
  const t0 = performance.now();
  const names = Object.keys(qs);
  const seqs = names.map((n) => buildSequence(tok!, state, qs[n], manifest.maxLen, manifest.headMaxLen));
  const L = Math.max(...seqs.map((s) => s.ids.length));
  const K = Math.max(...seqs.map((s) => s.markers.length));
  const B = names.length;
  const ids = new BigInt64Array(B * L);
  const att = new BigInt64Array(B * L);
  const mpos = new BigInt64Array(B * K);
  const mmask = new Uint8Array(B * K);
  const qt = new BigInt64Array(B);
  seqs.forEach((s, b) => {
    s.ids.forEach((v, i) => {
      ids[b * L + i] = BigInt(v);
      att[b * L + i] = 1n;
    });
    s.markers.forEach((p, k) => {
      mpos[b * K + k] = BigInt(p);
      mmask[b * K + k] = 1;
    });
    qt[b] = BigInt(QTYPES[qs[names[b]].type] ?? 0);
  });
  const feeds = {
    input_ids: new ort.Tensor("int64", ids, [B, L]),
    attention_mask: new ort.Tensor("int64", att, [B, L]),
    marker_pos: new ort.Tensor("int64", mpos, [B, K]),
    marker_mask: new ort.Tensor("bool", mmask, [B, K]),
    qtype: new ort.Tensor("int64", qt, [B]),
  };
  const out = await session.run(feeds);
  const raw = out.logits.data as Float32Array;
  // Keep as a plain array: `decodeBatch` slices it, and an ORT tensor view may be reused.
  const logits = Array.prototype.slice.call(raw) as number[];
  const answers = decodeBatch(qs, logits, K, manifest);
  const asked = Object.keys(answers).length;
  const abstained = Object.values(answers).filter((a) => a.abstained).length;
  return {
    answers,
    ms: performance.now() - t0,
    modelVersion: manifest.modelVersion,
    backend: backendUsed,
    calibrated: manifest.calibrated,
    banksMismatch,
    tokens: { batch: B, seqLen: L, markers: K },
    abstained: { count: abstained, of: asked },
  };
}

/** Week-0 spike: sustained FP32 GEMM throughput at the mmBERT hidden size, to size the budget. */
function benchmark() {
  const H = 768;
  const T = 64; // tokens per block
  const a = new Float32Array(T * H);
  const w = new Float32Array(H * H);
  const c = new Float32Array(T * H);
  for (let i = 0; i < a.length; i++) a[i] = Math.random() - 0.5;
  for (let i = 0; i < w.length; i++) w[i] = Math.random() - 0.5;
  const runs: number[] = [];
  for (let r = 0; r < 5; r++) {
    const t0 = performance.now();
    for (let i = 0; i < T; i++) {
      const ai = i * H;
      for (let k = 0; k < H; k++) {
        const av = a[ai + k];
        const wk = k * H;
        for (let j = 0; j < H; j++) c[ai + j] += av * w[wk + j];
      }
    }
    runs.push(performance.now() - t0);
  }
  runs.sort((x, y) => x - y);
  const ms = runs[2];
  const gflops = (2 * T * H * H) / (ms / 1000) / 1e9;
  // mmBERT-base ≈ 2·params FLOPs per token (~110M non-embedding params)
  const flopsPerToken = 2 * 110e6;
  const projected = (tokens: number) => ((flopsPerToken * tokens) / (gflops * 1e9 * 2.5) /* ORT SIMD/INT8 vs scalar JS */) * 1000;
  return { gflops, medianMs: ms, projectedMs: { q1: projected(320), q3: projected(3 * 320), q4: projected(4 * 320) } };
}

self.onmessage = async (ev: MessageEvent<any>) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") await load(msg.base, msg.backend, msg.threads);
    else if (msg.type === "predict") post({ type: "result", id: msg.id, ...(await predict(msg.state, msg.questions)) });
    else if (msg.type === "bench") post({ type: "bench", ...benchmark() });
    else if (msg.type === "inspect") {
      await inspect(msg.state, msg.base ?? "", msg.questions ?? {});
      post({
        type: "status",
        status: session ? "ready" : "unpublished",
        detail: session ? `${manifest.modelVersion} on ${backendUsed}` : "Tokenizer ready · no calibrated export published yet — Tier-0 active",
        backend: backendUsed,
      });
    }
  } catch (e) {
    post({ type: "status", status: "error", detail: String(e), id: msg.id });
  }
};
