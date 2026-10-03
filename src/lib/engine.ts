"use client";
/**
 * Main-thread façade over the two Workers.
 *  - Tier-0 worker: created on first check (cheap, ~30 KB).
 *  - Inference worker: created ONLY when the user explicitly enables deep checking
 *    (opt-in download banner) — a visitor who only reads never triggers a model fetch.
 */
import type { Tier0Result } from "@/core/fusion";
import type { QuestionSet, Answer } from "@/core/laya-client-browser";

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
}
export interface BenchResult {
  gflops: number;
  medianMs: number;
  projectedMs: { q1: number; q3: number; q4: number };
}
export interface InspectResult {
  ids: number;
  markers: number[];
  head: number[];
  special: { cls: number; sep: number; mask: number };
  ms: number;
}

type Listener = (s: ModelState) => void;
let infWorker: Worker | null = null;
let modelState: ModelState = { status: "idle", detail: "Deep checking not enabled", progress: 0 };
const listeners = new Set<Listener>();
const predWaiters = new Map<number, (r: { answers: Record<string, Answer>; ms: number; modelVersion: string; backend: string } | null) => void>();
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

function inference() {
  if (!infWorker) {
    infWorker = new Worker(new URL("../workers/inference.worker.ts", import.meta.url), { type: "module" });
    infWorker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "status") {
        setState({ status: m.status, detail: m.detail, backend: m.backend });
        if (m.id) { predWaiters.get(m.id)?.(null); predWaiters.delete(m.id); }
        if (m.status === "error" && inspectWaiter) { inspectWaiter(null); inspectWaiter = null; }
      } else if (m.type === "progress") setState({ status: "downloading", progress: m.total ? m.loaded / m.total : 0, detail: `${(m.loaded / 1048576).toFixed(1)} MB downloaded` });
      else if (m.type === "result") { predWaiters.get(m.id)?.(m); predWaiters.delete(m.id); }
      else if (m.type === "bench") { benchWaiter?.(m); benchWaiter = null; }
      else if (m.type === "inspect") { inspectWaiter?.(m); inspectWaiter = null; }
    };
    infWorker.onerror = (e) => setState({ status: "error", detail: e.message || "Worker failed to start — Tier-0 only" });
  }
  return infWorker;
}

export function enableDeepCheck(backend: string, threads: number) {
  if (modelState.status === "ready" || modelState.status === "probing" || modelState.status === "downloading") return;
  setState({ status: "probing", detail: "Starting inference worker…" });
  inference().postMessage({ type: "load", base: MODEL_BASE, backend, threads });
}

export function predictTier1(state: string, questions: QuestionSet) {
  if (modelState.status !== "ready") return Promise.resolve(null);
  const id = ++seq;
  return new Promise<{ answers: Record<string, Answer>; ms: number; modelVersion: string; backend: string } | null>((res) => {
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

/** Tokenizer-only dry run: downloads just the Laya tokenizer (~34 MB, cached) and returns the exact model input. */
export function inspectSequence(state: string): Promise<InspectResult | null> {
  return new Promise((res) => {
    inspectWaiter = res;
    inference().postMessage({ type: "inspect", state });
  });
}

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
