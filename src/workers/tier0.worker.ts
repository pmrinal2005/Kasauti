/// <reference lib="webworker" />
/**
 * Tier-0 Worker: normalisation, fingerprinting, Aho–Corasick, entity checks,
 * plausibility maths, offline registry lookup and fusion — all off the main thread.
 */
import { runTier0 } from "../core/fusion";

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = async (ev: MessageEvent<{ id: number; text: string }>) => {
  const { id, text } = ev.data;
  try {
    const result = await runTier0(text);
    self.postMessage({ id, ok: true, result });
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e) });
  }
};
