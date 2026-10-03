/// <reference lib="webworker" />
/**
 * Tier-1 inference Worker — hosts onnxruntime-web. Never loaded on first paint;
 * instantiated only when the user opens a tool that needs Tier-1.
 *
 * Model delivery: a PUBLIC Hugging Face *model* repo (static LFS bytes over the HF
 * CDN — no Space, no compute). Expected layout (produced by ml/export):
 *   <base>/manifest.json            { modelVersion, files:{wasm,webgpu,tokenizer}, maxLen, headMaxLen, temperatures, abstain }
 *   <base>/<files.wasm>             per-channel INT8 graph   (WASM EP)
 *   <base>/<files.webgpu>           4-bit MatMulNBits graph  (WebGPU EP)
 * Graph I/O mirrors DecisionModel.forward: input_ids, attention_mask, marker_pos,
 * marker_mask, qtype → logits [B, K].
 *
 * The Cache API (not IndexedDB) stores the binaries, keyed by content-hashed filename.
 * Also runs the Week-0 device micro-benchmark (GEMM throughput at hidden=768).
 */
import {
  buildSequence,
  decode,
  DEFAULT_MANIFEST,
  optionKeys,
  type CalibrationManifest,
  type QuestionSet,
  type Tokenizer,
} from "../core/laya-client-browser";

declare const self: DedicatedWorkerGlobalScope;

/* eslint-disable @typescript-eslint/no-explicit-any */
const dynImport = new Function("u", "return import(u)") as (u: string) => Promise<any>;
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const TOK_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2";
const CACHE = "kasauti-model-v1";

let session: any = null;
let tok: Tokenizer | null = null;
let manifest: CalibrationManifest = DEFAULT_MANIFEST;
let backendUsed = "none";

const post = (m: unknown) => self.postMessage(m);

async function cachedFetch(url: string, onProgress?: (loaded: number, total: number) => void): Promise<ArrayBuffer> {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(url);
  if (hit) return hit.arrayBuffer();
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
    onProgress?.(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.byteLength; }
  await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
  return buf.buffer;
}

async function load(base: string, backend: string, threads: number) {
  post({ type: "status", status: "probing", detail: `Looking for manifest at ${base}/manifest.json` });
  const mres = await fetch(`${base}/manifest.json`, { cache: "no-cache" }).catch(() => null);
  if (!mres || !mres.ok) {
    post({ type: "status", status: "unpublished", detail: "Quantized ONNX export not published yet — Tier-0 remains active (circuit breaker)." });
    return;
  }
  const m = await mres.json();
  manifest = { ...DEFAULT_MANIFEST, ...m };
  const useGpu = backend === "webgpu" && m.files?.webgpu;
  const file = useGpu ? m.files.webgpu : m.files.wasm;
  const ort = await dynImport(ORT_CDN + (useGpu ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs"));
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = threads;
  const bytes = await cachedFetch(`${base}/${file}`, (l, t) => post({ type: "progress", loaded: l, total: t }));
  session = await ort.InferenceSession.create(new Uint8Array(bytes), {
    executionProviders: useGpu ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
  });
  const tf = await dynImport(TOK_CDN);
  const hf = await tf.AutoTokenizer.from_pretrained(m.tokenizerRepo ?? "convaiinnovations/laya", { subfolder: m.tokenizerSubfolder ?? "multilingual/tokenizer" });
  tok = {
    encode: (s: string) => hf.encode(s, { add_special_tokens: false }),
    cls: hf.cls_token_id ?? 1, sep: hf.sep_token_id ?? 1, mask: hf.mask_token_id ?? 4, pad: hf.pad_token_id ?? 0,
  };
  backendUsed = useGpu ? "webgpu" : threads > 1 ? "wasm-mt" : "wasm-st";
  (self as any).__ort = ort;
  post({ type: "status", status: "ready", detail: `${manifest.modelVersion} on ${backendUsed}`, backend: backendUsed });
}

const QT: Record<string, number> = { choice: 0, score: 1, noul: 2 };

async function predict(state: string, qs: QuestionSet) {
  if (!session || !tok) throw new Error("model not loaded");
  const ort = (self as any).__ort;
  const t0 = performance.now();
  const names = Object.keys(qs);
  const seqs = names.map((n) => buildSequence(tok!, state, qs[n], manifest.maxLen, manifest.headMaxLen));
  const L = Math.max(...seqs.map((s) => s.ids.length));
  const K = Math.max(...seqs.map((s) => s.markers.length));
  const B = names.length;
  const ids = new BigInt64Array(B * L), att = new BigInt64Array(B * L);
  const mpos = new BigInt64Array(B * K), mmask = new Uint8Array(B * K), qt = new BigInt64Array(B);
  seqs.forEach((s, b) => {
    s.ids.forEach((v, i) => { ids[b * L + i] = BigInt(v); att[b * L + i] = 1n; });
    s.markers.forEach((p, k) => { mpos[b * K + k] = BigInt(p); mmask[b * K + k] = 1; });
    qt[b] = BigInt(QT[qs[names[b]].type]);
  });
  const out = await session.run({
    input_ids: new ort.Tensor("int64", ids, [B, L]),
    attention_mask: new ort.Tensor("int64", att, [B, L]),
    marker_pos: new ort.Tensor("int64", mpos, [B, K]),
    marker_mask: new ort.Tensor("bool", mmask, [B, K]),
    qtype: new ort.Tensor("int64", qt, [B]),
  });
  const logits = out.logits.data as Float32Array;
  const answers: Record<string, unknown> = {};
  names.forEach((n, b) => {
    const k = optionKeys(qs[n]).length;
    answers[n] = decode(qs[n], logits.subarray(b * K, b * K + k), manifest);
  });
  return { answers, ms: performance.now() - t0, modelVersion: manifest.modelVersion, backend: backendUsed };
}

/** Week-0 spike: measure sustained FP32 GEMM throughput at the encoder's hidden size. */
function benchmark() {
  const H = 768, T = 64; // hidden, tokens per block
  const a = new Float32Array(T * H), w = new Float32Array(H * H), c = new Float32Array(T * H);
  for (let i = 0; i < a.length; i++) a[i] = Math.random() - 0.5;
  for (let i = 0; i < w.length; i++) w[i] = Math.random() - 0.5;
  const runs: number[] = [];
  for (let r = 0; r < 5; r++) {
    const t0 = performance.now();
    for (let i = 0; i < T; i++) {
      const ai = i * H;
      for (let k = 0; k < H; k++) {
        const av = a[ai + k], wk = k * H;
        for (let j = 0; j < H; j++) c[ai + j] += av * w[wk + j];
      }
    }
    runs.push(performance.now() - t0);
  }
  runs.sort((x, y) => x - y);
  const ms = runs[2];
  const gflops = (2 * T * H * H) / (ms / 1000) / 1e9;
  // mmBERT-base ≈ 2·params FLOPs/token for the encoder (~110M non-embedding params)
  const flopsPerToken = 2 * 110e6;
  const projected = (tokens: number) => (flopsPerToken * tokens) / (gflops * 1e9 * 2.5 /* ORT SIMD/INT8 vs scalar JS */) * 1000;
  return { gflops, medianMs: ms, projectedMs: { q1: projected(320), q3: projected(3 * 320), q4: projected(4 * 320) } };
}

self.onmessage = async (ev: MessageEvent<any>) => {
  const msg = ev.data;
  try {
    if (msg.type === "load") await load(msg.base, msg.backend, msg.threads);
    else if (msg.type === "predict") post({ type: "result", id: msg.id, ...(await predict(msg.state, msg.questions)) });
    else if (msg.type === "bench") post({ type: "bench", ...benchmark() });
  } catch (e) {
    post({ type: "status", status: "error", detail: String(e), id: msg.id });
  }
};
