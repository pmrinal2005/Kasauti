"use client";
/** Tiny useSyncExternalStore-based store — no state library. */
import { useSyncExternalStore } from "react";
import { loadLocal, saveLocal, type CheckRecord } from "./data";

export type View = "overview" | "checker" | "lens" | "voice" | "engine" | "registry" | "privacy";

/**
 * The one list of views, as a `Record<View, true>` so TypeScript fails the build if a new member is
 * added to the union without being registered here. It exists because `hydrate()` used to carry its
 * own hardcoded array: opening `/dashboard#lens` in a fresh tab silently landed on Overview, and the
 * view was only reachable by clicking. Nothing about that failure was visible in a test or a log —
 * hence the compile-time rule.
 */
const VIEW_INDEX: Record<View, true> = {
  overview: true, checker: true, lens: true, voice: true, engine: true, registry: true, privacy: true,
};
export const VIEW_IDS = Object.keys(VIEW_INDEX) as View[];
export function isView(v: string): v is View {
  return Object.prototype.hasOwnProperty.call(VIEW_INDEX, v);
}
export type Lang = "en-IN" | "hi-IN" | "mr-IN" | "ta-IN";

interface State {
  view: View;
  lang: Lang;
  checks: CheckRecord[];
  hydrated: boolean;
  toast: string | null;
  draft: string;
}

let state: State = { view: "overview", lang: "en-IN", checks: [], hydrated: false, toast: null, draft: "" };
const subs = new Set<() => void>();
const emit = () => subs.forEach((s) => s());

export function set(p: Partial<State>) {
  state = { ...state, ...p };
  emit();
}
export function get() { return state; }

export function hydrate() {
  if (state.hydrated) return;
  const lang = (localStorage.getItem("kasauti.lang") as Lang) || "en-IN";
  const v = location.hash.slice(1);
  set({ checks: loadLocal(), hydrated: true, lang, view: isView(v) ? v : "overview" });
}

export function go(view: View, draft?: string) {
  history.replaceState(null, "", `#${view}`);
  set({ view, ...(draft !== undefined ? { draft } : {}) });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

export function setLang(lang: Lang) {
  localStorage.setItem("kasauti.lang", lang);
  set({ lang });
}

export function addCheck(c: CheckRecord) {
  const checks = [c, ...state.checks.filter((x) => x.hash !== c.hash)];
  saveLocal(checks);
  set({ checks });
}
export function clearChecks() { saveLocal([]); set({ checks: [] }); }

let toastTimer: ReturnType<typeof setTimeout> | undefined;
export function toast(msg: string) {
  clearTimeout(toastTimer);
  set({ toast: msg });
  toastTimer = setTimeout(() => set({ toast: null }), 2600);
}

const server: State = state;
export function useStore<T>(sel: (s: State) => T): T {
  return useSyncExternalStore((cb) => { subs.add(cb); return () => subs.delete(cb); }, () => sel(state), () => sel(server));
}
