/**
 * First-run capability gating: decides WASM vs WebGPU, the adaptive question
 * budget, and whether the model download should be deferred (Save-Data / 2G).
 */

export type Backend = "webgpu" | "wasm-mt" | "wasm-st" | "tier0-only";
export type Profile = "flagship" | "mid" | "low" | "minimal";

export interface Capability {
  webgpu: boolean;
  crossOriginIsolated: boolean;
  threads: number;
  memoryGB: number | null;
  saveData: boolean;
  effectiveType: string | null;
  speechRecognition: boolean;
  speechSynthesis: boolean;
  inAppBrowser: string | null;
  backend: Backend;
  profile: Profile;
  questionBudget: number; // max Stage-B noul follow-ups
  deferDownload: boolean;
  ortThreads: number;
}

interface NavExtra {
  gpu?: unknown;
  deviceMemory?: number;
  connection?: { saveData?: boolean; effectiveType?: string };
}

export function detectCapability(): Capability {
  const nav = navigator as Navigator & NavExtra;
  const webgpu = typeof nav.gpu !== "undefined";
  const coi = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  const threads = nav.hardwareConcurrency || 2;
  const memoryGB = typeof nav.deviceMemory === "number" ? nav.deviceMemory : null;
  const saveData = !!nav.connection?.saveData;
  const effectiveType = nav.connection?.effectiveType ?? null;
  const w = window as unknown as Record<string, unknown>;
  const speechRecognition = "SpeechRecognition" in w || "webkitSpeechRecognition" in w;
  const speechSynthesis = "speechSynthesis" in w;
  const ua = navigator.userAgent;
  const inAppBrowser = /WhatsApp/i.test(ua) ? "WhatsApp" : /Telegram/i.test(ua) ? "Telegram" : /FBAN|FBAV|Instagram/i.test(ua) ? "Meta" : null;

  const mem = memoryGB ?? 4;
  let profile: Profile = "mid";
  if (mem >= 8 && threads >= 8) profile = "flagship";
  else if (mem <= 2 || threads <= 2) profile = "minimal";
  else if (mem <= 3 || threads <= 4) profile = "low";

  let backend: Backend = webgpu && profile !== "minimal" ? "webgpu" : coi && threads > 2 ? "wasm-mt" : "wasm-st";
  if (profile === "minimal" && mem < 2) backend = "tier0-only";

  const questionBudget = { flagship: 3, mid: 3, low: 1, minimal: 0 }[profile];
  const deferDownload = saveData || effectiveType === "2g" || effectiveType === "slow-2g";
  // never starve the main thread on small phones
  const ortThreads = backend === "wasm-mt" ? Math.max(1, Math.min(4, threads - 1)) : 1;

  return { webgpu, crossOriginIsolated: coi, threads, memoryGB, saveData, effectiveType, speechRecognition, speechSynthesis, inAppBrowser, backend, profile, questionBudget, deferDownload, ortThreads };
}
