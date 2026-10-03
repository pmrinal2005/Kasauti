/**
 * Tier-0 pipeline + transparent evidence fusion.
 * Pure functions — runs identically in a Worker, on the main thread, or in Node tests.
 */

import { normalize } from "./normalize";
import { sha256Hex, simhash64 } from "./fingerprint";
import { AhoCorasick, FAMILY_META, LEXICON, type FlagFamily } from "./lexicon";
import { extractEntities, type Entities } from "./entities";
import { lookupRegistration, type RegistryEntry } from "./registry";
import { band, fmtPct, IMPLAUSIBLE_ANNUAL, PLAUSIBLE_ANNUAL } from "./plausibility";

export { fmtPct };

export type Rung = "safe_pattern" | "unclear" | "caution" | "likely_scam" | "known_scam";

export const RUNG_META: Record<Rung, { label: string; tone: string; step: number }> = {
  safe_pattern: { label: "No scam pattern found", tone: "ok", step: 0 },
  unclear: { label: "Not enough evidence", tone: "info", step: 1 },
  caution: { label: "Be careful", tone: "warn", step: 2 },
  likely_scam: { label: "Likely scam", tone: "bad", step: 3 },
  known_scam: { label: "Matches known scam script", tone: "bad", step: 4 },
};

export interface Evidence {
  code: string;
  label: string;
  detail: string;
  weight: number;
  source: "lexicon" | "entity" | "plausibility" | "registry" | "obfuscation" | "laya";
}

export interface Tier0Result {
  hash: string;
  simhash: string;
  normalized: string;
  families: Partial<Record<FlagFamily, string[]>>;
  entities: Entities;
  registry: Array<{ reg: string; entry: RegistryEntry | null }>;
  evidence: Evidence[];
  score: number;
  rung: Rung;
  confidence: number; // 0..1, heuristic agreement — NOT a calibrated model probability
  elapsedMs: number;
}

let AC: AhoCorasick | null = null;
const automaton = () => (AC ??= new AhoCorasick(LEXICON));


export function rungFromScore(score: number, families: number): Rung {
  if (score >= 9 && families >= 3) return "known_scam";
  if (score >= 5.5) return "likely_scam";
  if (score >= 2.5) return "caution";
  if (score > 0.5) return "unclear";
  return "safe_pattern";
}

export async function runTier0(raw: string): Promise<Tier0Result> {
  const t0 = performance.now();
  const { text, obfuscation } = normalize(raw);
  const [hash, simhash] = [await sha256Hex(text), simhash64(text)];

  const families: Partial<Record<FlagFamily, string[]>> = {};
  for (const h of automaton().search(text)) {
    const arr = (families[h.family] ??= []);
    if (!arr.includes(h.term)) arr.push(h.term);
  }

  const evidence: Evidence[] = [];
  for (const fam of Object.keys(families) as FlagFamily[]) {
    const meta = FAMILY_META[fam];
    evidence.push({
      code: `LEX_${fam.toUpperCase()}`,
      label: meta.label,
      detail: `Matched: “${families[fam]!.slice(0, 3).join("”, “")}”. ${meta.advice}`,
      weight: meta.weight,
      source: "lexicon",
    });
  }

  const entities = extractEntities(text);

  for (const r of entities.returns) {
    if (r.annualised > IMPLAUSIBLE_ANNUAL) {
      evidence.push({
        code: "PLAUS_RETURN",
        label: "Implausible return",
        detail: `${r.percent}% per ${r.period} ≈ ${fmtPct(r.annualised)} per year compounded — far above the ~${PLAUSIBLE_ANNUAL}% long-run ceiling of real assets.`,
        weight: band(r.annualised) === "impossible" ? 4 : 2.5,
        source: "plausibility",
      });
    }
  }
  for (const u of entities.upi) {
    evidence.push({
      code: u.knownPsp ? "UPI_PRESENT" : "UPI_UNKNOWN_PSP",
      label: u.knownPsp ? "Payment handle present" : "Unrecognised UPI handle",
      detail: u.knownPsp
        ? `${u.id} — a direct UPI ID inside an 'investment' message means money goes to an individual, not a regulated entity.`
        : `${u.id} — '@${u.handle}' is not a known PSP suffix.`,
      weight: u.knownPsp ? 1.5 : 2,
      source: "entity",
    });
  }
  for (const p of entities.phones) {
    if (p.kind === "international")
      evidence.push({ code: "PHONE_INTL", label: "International number", detail: `${p.raw} — foreign numbers posing as Indian officials are common.`, weight: 1.5, source: "entity" });
    if (p.kind === "trai_1600")
      evidence.push({ code: "PHONE_1600", label: "TRAI 1600-series caller", detail: `${p.raw} — 1600-series is reserved for regulated BFSI entities (a positive signal).`, weight: -1, source: "entity" });
  }
  if (entities.urls.some((u) => /\.(xyz|top|club|site|online|live|link)\b/.test(u)))
    evidence.push({ code: "URL_CHEAP_TLD", label: "Throw-away domain", detail: "Link uses a low-cost TLD frequently seen in phishing.", weight: 1.5, source: "entity" });
  if (entities.urls.some((u) => /\.apk\b/.test(u)))
    evidence.push({ code: "URL_APK", label: "APK download link", detail: "Side-loaded apps can read your OTPs and screen.", weight: 3, source: "entity" });

  const registry = entities.registrations.map((reg) => ({ reg, entry: lookupRegistration(reg) }));
  for (const { reg, entry } of registry) {
    if (!entry) evidence.push({ code: "REG_NOT_FOUND", label: "Registration not found", detail: `${reg} is not in the offline SEBI snapshot.`, weight: 2.5, source: "registry" });
    else if (entry.status !== "active")
      evidence.push({ code: "REG_INACTIVE", label: "Registration inactive", detail: `${reg} (${entry.name}) is ${entry.status}.`, weight: 3, source: "registry" });
    else evidence.push({ code: "REG_ACTIVE", label: "Registered intermediary", detail: `${reg} — ${entry.name}, ${entry.kind}, valid till ${entry.validTill}. Registration ≠ endorsement.`, weight: -1.5, source: "registry" });
  }

  if (obfuscation >= 2)
    evidence.push({ code: "OBFUSCATION", label: "Hidden characters", detail: `${obfuscation} zero-width/look-alike characters removed — often used to dodge filters.`, weight: Math.min(3, obfuscation * 0.5), source: "obfuscation" });

  const score = Math.max(0, evidence.reduce((s, e) => s + e.weight, 0));
  const famCount = Object.keys(families).filter((f) => f !== "defensive").length;
  const rung = rungFromScore(score, famCount);
  const positive = evidence.filter((e) => e.weight > 0).length;
  const confidence = Math.min(0.95, 0.35 + positive * 0.1 + (rung === "safe_pattern" ? 0.1 : 0));

  return { hash, simhash, normalized: text, families, entities, registry, evidence, score, rung, confidence, elapsedMs: performance.now() - t0 };
}

/* ------------------------------ Tier-1 fusion ------------------------------ */

/** Minimal shape of a calibrated Laya answer (see laya-client-browser.Answer). */
export interface T1Answer {
  noul?: number;
  choice?: string;
  confidence: number;
  abstained: boolean;
}

/** Weight each confident "yes" follow-up adds. Strong scripts (pay-to-withdraw, digital arrest) weigh more. */
export const T1_WEIGHTS: Record<string, number> = {
  pay_to_withdraw: 3, digital_arrest: 3, transfer: 2.5, guaranteed: 2, secrecy: 2, remote_app: 2.5, unregistered_group: 1.5, selling: 1.5,
};
export const T1_YES = 0.7; // calibrated p(yes) needed to add evidence

export interface FusedVerdict {
  rung: Rung;
  score: number;
  evidence: Evidence[];
  tier: "T0" | "T0+T1";
}

/**
 * Transparent rule table: Tier-0 evidence + confident, non-abstained Laya answers.
 * Laya can only ADD evidence or soften a weak Tier-0 result when it confidently says "education";
 * it can never override hard deterministic proof (pay-to-withdraw lexicon, registry mismatch).
 */
export function fuseTier1(t0: Pick<Tier0Result, "evidence" | "families" | "score">, triage: T1Answer | null, followups: Record<string, T1Answer>): FusedVerdict {
  const evidence = [...t0.evidence];
  let score = t0.score;
  let extraFamilies = 0;
  if (!triage) {
    const fam = Object.keys(t0.families).filter((f) => f !== "defensive").length;
    return { rung: rungFromScore(score, fam), score, evidence, tier: "T0" };
  }
  for (const [k, a] of Object.entries(followups)) {
    if (a.abstained || a.noul === undefined) continue;
    if (a.noul >= T1_YES) {
      const w = T1_WEIGHTS[k] ?? 1.5;
      score += w;
      extraFamilies++;
      evidence.push({ code: `LAYA_${k.toUpperCase()}`, label: `Laya: ${k.replace(/_/g, " ")}`, detail: `On-device model answered “yes” with p=${a.noul.toFixed(2)} (confidence ${(a.confidence * 100).toFixed(0)}%).`, weight: w, source: "laya" });
    }
  }
  const hardProof = t0.evidence.some((e) => e.weight >= 4 && e.source !== "laya");
  if (!triage.abstained && triage.choice === "education" && !hardProof && score < 5.5) {
    const sellingYes = (followups.selling?.noul ?? 0) >= T1_YES;
    if (!sellingYes) {
      score = Math.max(0, score - 1.5);
      evidence.push({ code: "LAYA_EDUCATION", label: "Laya: educational content", detail: `Triage classified this as education (confidence ${(triage.confidence * 100).toFixed(0)}%).`, weight: -1.5, source: "laya" });
    }
  }
  const fam = Object.keys(t0.families).filter((f) => f !== "defensive").length + extraFamilies;
  return { rung: rungFromScore(score, fam), score, evidence, tier: "T0+T1" };
}
