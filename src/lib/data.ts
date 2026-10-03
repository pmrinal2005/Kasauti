/**
 * Dashboard data layer.
 * - Network data: ILLUSTRATIVE, generated from a seeded PRNG (deterministic across SSR/CSR),
 *   standing in for the aggregate shared-verdict cache (hash + codes only, never raw text).
 * - Local data: the visitor's own checks, stored only on this device (localStorage).
 */
import type { FlagFamily } from "@/core/lexicon";
import type { Rung } from "@/core/fusion";

export function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface DayPoint {
  day: string; // ISO date
  checks: number;
  scams: number;
  cacheHits: number;
  tier1: number;
}

const BASE_DATE = Date.UTC(2026, 9, 3); // 2026-10-03

export function networkSeries(days: number): DayPoint[] {
  const r = mulberry32(1930);
  const out: DayPoint[] = [];
  let level = 2600;
  for (let i = 89; i >= 0; i--) {
    level *= 1 + (r() - 0.42) * 0.06;
    const dow = new Date(BASE_DATE - i * 864e5).getUTCDay();
    const weekend = dow === 0 || dow === 6 ? 1.18 : 1; // forwards spike on weekends
    const viral = r() > 0.94 ? 1.6 : 1;
    const checks = Math.round(level * weekend * viral);
    const scams = Math.round(checks * (0.27 + r() * 0.09) * (viral > 1 ? 1.25 : 1));
    const cacheHits = Math.round(checks * (0.52 + r() * 0.12) * (viral > 1 ? 1.2 : 1));
    out.push({ day: new Date(BASE_DATE - i * 864e5).toISOString().slice(0, 10), checks, scams, cacheHits: Math.min(cacheHits, checks), tier1: Math.round((checks - cacheHits) * 0.71) });
  }
  return out.slice(-days);
}

export const FAMILY_SHARE: Array<{ family: FlagFamily; label: string; count: number }> = [
  { family: "guaranteed_return", label: "Guaranteed return", count: 18420 },
  { family: "vip_group", label: "VIP / tips group", count: 14210 },
  { family: "pay_to_withdraw", label: "Pay-to-withdraw", count: 9630 },
  { family: "impersonation", label: "Digital arrest", count: 8120 },
  { family: "kyc_threat", label: "KYC threat", count: 7040 },
  { family: "insider_tip", label: "Insider tip", count: 5480 },
  { family: "remote_access", label: "Remote access", count: 3110 },
];

export const LANG_SHARE = [
  { lang: "हिन्दी", code: "hi", pct: 41 },
  { lang: "English", code: "en", pct: 19 },
  { lang: "मराठी", code: "mr", pct: 12 },
  { lang: "தமிழ்", code: "ta", pct: 10 },
  { lang: "বাংলা", code: "bn", pct: 8 },
  { lang: "తెలుగు", code: "te", pct: 6 },
  { lang: "Other", code: "xx", pct: 4 },
];

export interface CheckRecord {
  id: string;
  ts: number;
  excerpt: string; // local-only; never synced
  hash: string;
  rung: Rung;
  score: number;
  families: string[];
  lang: string;
  channel: "paste" | "voice" | "share" | "demo";
  tier: "T0" | "T0+T1" | "cache";
  ms: number;
  local: boolean;
}

const DEMO_TEXTS: Array<[string, Rung, string[], string]> = [
  ["Join our VIP group, guaranteed 3% daily profit, limited seats…", "known_scam", ["guaranteed_return", "vip_group", "urgency"], "en"],
  ["आपका KYC expired है, खाता बंद होगा, इस लिंक पर क्लिक करें", "likely_scam", ["kyc_threat"], "hi"],
  ["SIP kya hota hai? Monthly investing explained by AMFI", "safe_pattern", [], "hi"],
  ["CBI officer: your Aadhaar is linked to money laundering case, stay on video call", "known_scam", ["impersonation", "secrecy"], "en"],
  ["Pay 18% withdrawal tax to unlock your profit of ₹4.2 lakh", "known_scam", ["pay_to_withdraw"], "en"],
  ["SEBI warns investors: beware of unregistered tips groups", "safe_pattern", ["defensive"], "en"],
  ["Operator stock, sure shot multibagger, upper circuit tomorrow", "likely_scam", ["insider_tip"], "en"],
  ["उद्या पर्यंत हमी परतावा 20% महिना, लगेच गुंतवा", "likely_scam", ["guaranteed_return", "urgency"], "mr"],
  ["Install AnyDesk so our support team can process your refund", "likely_scam", ["remote_access"], "en"],
  ["Dividend record date announced — what it means for shareholders", "safe_pattern", [], "en"],
  ["உறுதியான லாபம் — இன்றே சேருங்கள்", "caution", ["guaranteed_return", "urgency"], "ta"],
  ["IPO allotment guaranteed through institutional account", "likely_scam", ["vip_group"], "en"],
];

export function demoRecords(): CheckRecord[] {
  const r = mulberry32(42);
  return DEMO_TEXTS.map(([excerpt, rung, families, lang], i) => ({
    id: `demo-${i}`,
    ts: BASE_DATE + 9 * 3600e3 - i * (r() * 2.4 + 0.4) * 3600e3,
    excerpt,
    hash: Array.from({ length: 8 }, () => Math.floor(r() * 16).toString(16)).join(""),
    rung,
    score: rung === "safe_pattern" ? 0 : rung === "caution" ? 3 + r() : rung === "likely_scam" ? 6 + r() * 2 : 10 + r() * 4,
    families,
    lang,
    channel: (["paste", "voice", "share", "paste"] as const)[Math.floor(r() * 4)],
    tier: r() > 0.5 ? "cache" : r() > 0.3 ? "T0+T1" : "T0",
    ms: Math.round(r() * 40 + 3),
    local: false,
  }));
}

const KEY = "kasauti.checks.v1";
export function loadLocal(): CheckRecord[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}
export function saveLocal(list: CheckRecord[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, 200)));
  } catch {
    /* quota / private mode — local history is best-effort */
  }
}
