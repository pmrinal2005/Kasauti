"use client";
/**
 * Main-thread façade over the two Workers.
 *  - Tier-0 worker: created on first check (cheap, ~30 KB).
 *  - Inference worker: created ONLY when the user explicitly enables deep checking
 *    (opt-in download banner) — a visitor who only reads never triggers a model fetch.
 */
import type { Tier0Result } from "@/core/fusion";
import type { Answer, CalibrationManifest, QuestionSet } from "@/core/laya-client-browser";
import { BANKS_SHA256, BANKS_VERSION } from "@/lib/banks";
import { resolveBase } from "@/lib/url";

let t0Worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, (r: { ok: boolean; result?: Tier0Result; error?: string }) => void>();

function tier0() {
  if (!t0Worker) {
    t0Worker = new Worker(new URL("../workers/tier0.worker.ts", import.meta.url), { type: "module" });
    t0Worker.onmessage = (e) => {
      const cb = pending.get(e.data.id);
      pending.delete(e.data.id);
      cb?.(e.data);
    };
  }
  return t0Worker;
}

export async function checkTier0(text: string): Promise<Tier0Result> {
  try {
    const w = tier0();
    const id = ++seq;
    return await new Promise<Tier0Result>((res, rej) => {
      const to = setTimeout(() => { pending.delete(id); rej(new Error("tier0 timeout")); }, 4000);
      pending.set(id, (r) => { clearTimeout(to); r.ok ? res(r.result!) : rej(new Error(r.error)); });
      w.postMessage({ id, text });
    });
  } catch {
    // circuit breaker: Worker unavailable (very old browser / CSP) → run on main thread
    const { runTier0 } = await import("@/core/fusion");
    return runTier0(text);
  }
}

/* ------------------------------ inference worker ------------------------------ */

export type ModelStatus = "idle" | "probing" | "downloading" | "ready" | "unpublished" | "error" | "deferred";
export interface ModelState {
  status: ModelStatus;
  detail: string;
  progress: number; // 0..1
  backend?: string;
  /** what the worker actually loaded — null until a manifest has been read */
  manifest: CalibrationManifest | null;
  /** true when the published calibration is missing, so answers are report-only, never acted on */
  calibrated: boolean;
  /** non-null when the model was trained on different question wording than this runtime asks */
  banksMismatch: string | null;
  tokenizer: { vocabSize: number; kind: string; specials: Record<string, number>; bytes: number } | null;
  graph: { path: string; bytes?: number; sha256?: string } | null;
}

/** Cache name the worker writes model bytes into; the Engine view shows and clears it. */
export const CACHE_NAME = "kasauti-model-v2";
export interface BenchResult {
  gflops: number;
  medianMs: number;
  projectedMs: { q1: number; q3: number; q4: number };
}
export interface InspectResult {
  ids: number;
  markers: number[];
  head: number[];
  special: { cls: number; sep: number; mask: number; pad: number; unk: number };
  vocabSize: number;
  kind: string;
  options: number;
  ms: number;
}

type Listener = (s: ModelState) => void;
let infWorker: Worker | null = null;
let modelState: ModelState = {
  status: "idle",
  detail: "Deep checking not enabled",
  progress: 0,
  manifest: null,
  calibrated: false,
  banksMismatch: null,
  tokenizer: null,
  graph: null,
};
const listeners = new Set<Listener>();
export interface Tier1Result {
  answers: Record<string, Answer>;
  ms: number;
  modelVersion: string;
  backend: string;
  calibrated: boolean;
  banksMismatch: string | null;
  tokens: { batch: number; seqLen: number; markers: number };
  abstained: { count: number; of: number };
}
const predWaiters = new Map<number, (r: Tier1Result | null) => void>();
let benchWaiter: ((b: BenchResult) => void) | null = null;
let inspectWaiter: ((r: InspectResult | null) => void) | null = null;

function setState(s: Partial<ModelState>) {
  modelState = { ...modelState, ...s };
  listeners.forEach((l) => l(modelState));
}
export function subscribeModel(l: Listener) {
  listeners.add(l);
  l(modelState);
  return () => { listeners.delete(l); };
}
export function getModelState() { return modelState; }

export const MODEL_BASE = process.env.NEXT_PUBLIC_LAYA_MODEL_BASE || "https://huggingface.co/kasauti/laya-multilingual-onnx/resolve/main";

/**
 * Absolute base URL for the worker. The rule itself lives in `./url.ts` (pure, unit-tested); this
 * only supplies the page's own href.
 */
function modelBase(): string {
  return resolveBase(MODEL_BASE, typeof location === "undefined" ? "" : location.href);
}

function inference() {
  if (!infWorker) {
    infWorker = new Worker(new URL("../workers/inference.worker.ts", import.meta.url), { type: "module" });
    infWorker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "status") {
        setState({ status: m.status, detail: m.detail, backend: m.backend });
        if (m.id) { predWaiters.get(m.id)?.(null); predWaiters.delete(m.id); }
        if (m.status === "error" && inspectWaiter) { inspectWaiter(null); inspectWaiter = null; }
      } else if (m.type === "progress") setState({ status: "downloading", progress: m.total ? m.loaded / m.total : 0, detail: `${(m.loaded / 1048576).toFixed(1)} MB ${m.what ?? "model"}` });
      else if (m.type === "manifest")
        setState({
          manifest: m.manifest as CalibrationManifest,
          calibrated: Boolean((m.manifest as CalibrationManifest).calibrated),
          banksMismatch: (m.banksMismatch as string | null) ?? null,
          backend: m.backend as string,
          tokenizer: m.tokenizer ?? null,
          graph: m.graph ?? null,
        });
      else if (m.type === "result") { predWaiters.get(m.id)?.(m); predWaiters.delete(m.id); }
      else if (m.type === "bench") { benchWaiter?.(m); benchWaiter = null; }
      else if (m.type === "inspect") { inspectWaiter?.(m); inspectWaiter = null; }
    };
    infWorker.onerror = (e) => setState({ status: "error", detail: e.message || "Worker failed to start — Tier-0 only" });
  }
  return infWorker;
}

/**
 * Save-Data / 2G guard: the blueprint's rule is that a metered, slow connection is offered the
 * Tier-0 answer instead of a multi-megabyte download. The visitor can still force it — this only
 * changes the *default*, and the copy says plainly what is being avoided and why.
 */
export function deferReason(): string | null {
  if (typeof navigator === "undefined") return null;
  const c = (navigator as unknown as { connection?: { saveData?: boolean; effectiveType?: string } }).connection;
  if (!c) return null;
  if (c.saveData) return "Data Saver is on";
  if (c.effectiveType && /^(slow-)?2g$/.test(c.effectiveType)) return `connection is ${c.effectiveType}`;
  return null;
}

export function enableDeepCheck(backend: string, threads: number, opts: { force?: boolean } = {}) {
  if (modelState.status === "ready" || modelState.status === "probing" || modelState.status === "downloading") return;
  const why = deferReason();
  if (why && !opts.force) {
    setState({
      status: "deferred",
      detail: `Deep check paused — ${why}. Tier-0 still answers, all of it on this device.`,
    });
    return;
  }
  setState({ status: "probing", detail: "Starting inference worker…" });
  inference().postMessage({ type: "load", base: modelBase(), backend, threads });
}

export function predictTier1(state: string, questions: QuestionSet): Promise<Tier1Result | null> {
  if (modelState.status !== "ready") return Promise.resolve(null);
  const id = ++seq;
  return new Promise<Tier1Result | null>((res) => {
    predWaiters.set(id, res);
    inference().postMessage({ type: "predict", id, state, questions });
  });
}

export function runBenchmark(): Promise<BenchResult> {
  return new Promise((res) => {
    benchWaiter = res;
    inference().postMessage({ type: "bench" });
  });
}

/**
 * Tokenizer-only dry run: loads just the published (vocabulary-pruned) tokenizer and returns the
 * exact model input for the message — useful before any graph exists, and as a support tool.
 */
export function inspectSequence(state: string, questions?: QuestionSet): Promise<InspectResult | null> {
  return new Promise((res) => {
    inspectWaiter = res;
    inference().postMessage({ type: "inspect", state, base: modelBase(), questions: questions ?? {} });
  });
}

/** True when the loaded export was trained on the same question banks this build asks. */
export function banksInSync(): boolean {
  return modelState.banksMismatch === null;
}
export { BANKS_SHA256, BANKS_VERSION };

/* ------------------------------ local verdict cache ------------------------------ */
// Step 4 of the data flow: an on-device hash → fused-verdict map (<2 ms), checked before the network.
const LV_KEY = "kasauti.verdicts.v1";
export interface LocalVerdict { rung: string; score: number; tier: string; modelVersion: string; ts: number }
function lvLoad(): Record<string, LocalVerdict> { try { return JSON.parse(localStorage.getItem(LV_KEY) || "{}"); } catch { return {}; } }
export function localVerdict(hash: string): LocalVerdict | null { return lvLoad()[hash] ?? null; }
export function saveLocalVerdict(hash: string, v: LocalVerdict) {
  try {
    const all = lvLoad();
    all[hash] = v;
    const keys = Object.keys(all);
    if (keys.length > 500) keys.sort((a, b) => all[a].ts - all[b].ts).slice(0, keys.length - 500).forEach((k) => delete all[k]);
    localStorage.setItem(LV_KEY, JSON.stringify(all));
  } catch { /* quota / private mode */ }
}

/* ------------------------------ shared verdict cache ------------------------------ */

export async function lookupShared(hash: string): Promise<{ rung: string; codes: string[]; hits: number } | null> {
  try {
    const r = await fetch(`/api/v/${hash}`);
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export function syncShared(t: Tier0Result, modelVersion: string, fused?: { rung: string; codes: string[] }) {
  // non-blocking; compact codes only — NEVER raw text
  const body = JSON.stringify({ simhash: t.simhash, rung: fused?.rung ?? t.rung, codes: fused?.codes ?? t.evidence.map((e) => e.code), modelVersion });
  fetch(`/api/v/${t.hash}`, { method: "POST", headers: { "content-type": "application/json" }, body, keepalive: true }).catch(() => {});
}
