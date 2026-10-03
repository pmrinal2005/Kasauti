/**
 * Browser-native voice layer. ZERO audio bytes ever reach Kasauti's servers.
 *
 * recognizer: wraps SpeechRecognition / webkitSpeechRecognition.
 *   - requests on-device recognition (`processLocally = true`) first, falls back
 *     transparently; exposes which mode is active so the UI never over-claims.
 *   - auto-restarts on unexpected `onend` while hold-to-talk is active.
 *   - enforces a hard timeout so a stuck session never blocks the UI.
 * synthesizer: wraps speechSynthesis with a language-filtered voice chain:
 *   device voice for exact lang → same base language → (pre-rendered clip) → text only.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyRec = any;

export type RecMode = "on-device" | "online" | "unknown";

export interface RecognizerHandle {
  stop(): void;
  mode: RecMode;
}

export interface RecognizerOpts {
  lang: string;
  onInterim?: (t: string) => void;
  onFinal: (t: string) => void;
  onError?: (e: string) => void;
  onMode?: (m: RecMode) => void;
  onEnd?: () => void;
  hardTimeoutMs?: number;
}

export function recognitionSupported(): boolean {
  if (typeof window === "undefined") return false;
  return "SpeechRecognition" in window || "webkitSpeechRecognition" in window;
}

export function startRecognizer(o: RecognizerOpts): RecognizerHandle | null {
  const w = window as AnyRec;
  const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
  if (!Ctor) return null;
  let active = true;
  let finalText = "";
  let mode: RecMode = "unknown";
  let restarts = 0;

  const make = (local: boolean) => {
    const r = new Ctor();
    r.lang = o.lang; // never trust browser auto-detection for code-mixed speech
    r.interimResults = true;
    r.continuous = true;
    r.maxAlternatives = 1;
    if (local && "processLocally" in r) {
      try {
        r.processLocally = true;
        mode = "on-device";
      } catch {
        mode = "online";
      }
    } else mode = local ? "online" : mode === "unknown" ? "online" : mode;
    o.onMode?.(mode);
    r.onresult = (ev: AnyRec) => {
      let interim = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const res = ev.results[i];
        if (res.isFinal) finalText += res[0].transcript + " ";
        else interim += res[0].transcript;
      }
      o.onInterim?.((finalText + interim).trim());
    };
    r.onerror = (ev: AnyRec) => {
      const err = String(ev.error || "error");
      // on-device language pack missing → fall back to default (online) recognition
      if (mode === "on-device" && /language-not-supported|service-not-allowed|not-allowed-local/.test(err)) {
        mode = "online";
        rec = make(false);
        try { rec.start(); } catch { /* ignore */ }
        return;
      }
      if (err !== "no-speech" && err !== "aborted") o.onError?.(err);
    };
    r.onend = () => {
      if (active && restarts < 5) {
        restarts++;
        try { r.start(); return; } catch { /* fallthrough */ }
      }
      finish();
    };
    return r;
  };

  let rec = make(true);
  const timer = setTimeout(() => handle.stop(), o.hardTimeoutMs ?? 30000);
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    clearTimeout(timer);
    if (finalText.trim()) o.onFinal(finalText.trim());
    o.onEnd?.();
  };
  try {
    rec.start();
  } catch (e) {
    o.onError?.(String(e));
    return null;
  }
  const handle: RecognizerHandle = {
    get mode() { return mode; },
    stop() {
      active = false;
      try { rec.stop(); } catch { finish(); }
    },
  };
  return handle;
}

/* ------------------------------- synthesizer ------------------------------- */

export function voicesFor(lang: string): SpeechSynthesisVoice[] {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return [];
  const all = window.speechSynthesis.getVoices();
  const exact = all.filter((v) => v.lang.toLowerCase() === lang.toLowerCase());
  if (exact.length) return exact.sort((a, b) => Number(b.localService) - Number(a.localService));
  const base = lang.split("-")[0].toLowerCase();
  return all.filter((v) => v.lang.toLowerCase().startsWith(base));
}

export type SpeakResult = "device-voice" | "no-voice";

export function speak(text: string, lang: string, rate = 0.95): SpeakResult {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return "no-voice";
  const v = voicesFor(lang)[0];
  if (!v) return "no-voice"; // caller falls back to pre-rendered clip → text only
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.voice = v;
  u.lang = v.lang;
  u.rate = rate;
  window.speechSynthesis.speak(u);
  return "device-voice";
}

export function stopSpeaking() {
  if (typeof window !== "undefined" && "speechSynthesis" in window) window.speechSynthesis.cancel();
}
