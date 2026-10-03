/**
 * laya-client-browser — the client-side contract for Laya (convaiinnovations/laya).
 *
 * Mirrors the reference Python implementation (rl_common.py) exactly where it
 * matters for an ONNX port:
 *   sequence  = [CLS] "<type> question: <ins>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
 *   noul      = options rendered as ["false: …", "true: …"]  → p[1] is the noul probability
 *   score     = options rendered as "level i: …"
 *   confidence= 1 − H(p)/log(k)            (Jev-style normalised entropy)
 *   calibration = one temperature per bucket "<type>:<2|3-5|6-10|11+>"
 *
 * Laya answers ONLY flat choice / score / noul questions. Question trees are adaptive:
 * one triage `choice`, then ≤ budget targeted `noul` follow-ups.
 */

export type QType = "choice" | "score" | "noul";
export interface Question {
  type: QType;
  instructions: string;
  criteria?: Record<string, string> | string[];
}
export type QuestionSet = Record<string, Question>;

export interface CalibrationManifest {
  modelVersion: string;
  maxLen: number;
  headMaxLen: number;
  temperatures: Record<string, number>; // bucket → T
  abstain: Record<string, number>; // bucket → min confidence to act
}

/** Defaults taken from multilingual/rl_agent_config.json (uncalibrated: all T = 1). */
export const DEFAULT_MANIFEST: CalibrationManifest = {
  modelVersion: "laya-multilingual@base (uncalibrated)",
  maxLen: 1024,
  headMaxLen: 256,
  temperatures: {},
  abstain: { "noul:2": 0.35, "choice:2": 0.35, "choice:3-5": 0.3, "choice:6-10": 0.25, "choice:11+": 0.2, "score:3-5": 0.3 },
};

export function renderOptions(q: Question): string[] {
  if (q.type === "choice") {
    const c = (q.criteria ?? {}) as Record<string, string>;
    return Object.entries(c).map(([k, v]) => (v ? `${k}: ${v}` : k));
  }
  if (q.type === "score") return ((q.criteria ?? []) as string[]).map((c, i) => `level ${i}: ${c}`);
  const c = (q.criteria ?? {}) as Record<string, string>;
  return [`false: ${c.false || "no, the statement does not hold"}`, `true: ${c.true || "yes, the statement holds"}`];
}

export function optionKeys(q: Question): string[] {
  if (q.type === "choice") return Object.keys((q.criteria ?? {}) as Record<string, string>);
  if (q.type === "score") return ((q.criteria ?? []) as string[]).map((_, i) => String(i));
  return ["false", "true"];
}

export function bucket(type: QType, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${type}:${size}`;
}

export interface Tokenizer {
  encode(text: string): number[];
  cls: number;
  sep: number;
  mask: number;
  pad: number;
}

/** Port of build_sequence(): returns input ids and per-option [MASK] marker positions. */
export function buildSequence(tok: Tokenizer, state: string, q: Question, maxLen: number, headMaxLen: number) {
  const opts = renderOptions(q);
  let head = tok.encode(`${q.type} question: ${q.instructions}`);
  let optIds = opts.map((o) => [tok.mask, ...tok.encode(" " + o).slice(0, 48)]);
  let budget = headMaxLen - optIds.reduce((s, o) => s + o.length, 0);
  if (budget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    budget = headMaxLen - optIds.reduce((s, o) => s + o.length, 0);
  }
  head = head.slice(0, Math.max(8, budget));
  const ids = [tok.cls, ...head, tok.sep];
  const markers: number[] = [];
  for (const o of optIds) {
    markers.push(ids.length);
    ids.push(...o);
  }
  ids.push(tok.sep);
  const room = Math.max(0, maxLen - ids.length - 1);
  ids.push(...tok.encode(state).slice(0, room), tok.sep);
  return { ids: ids.slice(0, maxLen), markers: markers.filter((m) => m < maxLen) };
}

export function softmax(logits: ArrayLike<number>, T = 1): Float64Array {
  const n = logits.length;
  const out = new Float64Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logits[i] / T);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += out[i] = Math.exp(logits[i] / T - max);
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function entropyConfidence(p: ArrayLike<number>): number {
  const k = p.length;
  if (k < 2) return 1;
  let h = 0;
  for (let i = 0; i < k; i++) h -= p[i] * Math.log(Math.max(p[i], 1e-12));
  return 1 - h / Math.log(k);
}

export interface Answer {
  type: QType;
  probs: Record<string, number>;
  choice?: string;
  score?: number;
  noul?: number;
  confidence: number;
  abstained: boolean;
  bucket: string;
}

/** Calibrated decode of one question's marker logits. */
export function decode(q: Question, logits: ArrayLike<number>, m: CalibrationManifest = DEFAULT_MANIFEST): Answer {
  const keys = optionKeys(q);
  const b = bucket(q.type, keys.length);
  const p = softmax(logits, m.temperatures[b] ?? 1);
  const conf = entropyConfidence(p);
  const probs: Record<string, number> = {};
  keys.forEach((k, i) => (probs[k] = p[i]));
  let arg = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[arg]) arg = i;
  const a: Answer = { type: q.type, probs, confidence: conf, abstained: conf < (m.abstain[b] ?? 0.3), bucket: b };
  if (q.type === "choice") a.choice = keys[arg];
  if (q.type === "score") a.score = p.reduce((s, v, i) => s + v * i, 0);
  if (q.type === "noul") a.noul = p[1];
  return a;
}

/* ------------------------------ Question banks ------------------------------ */

export const CLAIM_V1_TRIAGE: Question = {
  type: "choice",
  instructions: "What kind of financial message is this?",
  criteria: {
    investment_offer: "promises returns, tips, trading groups or schemes",
    authority_threat: "claims to be police, CBI, customs, bank or regulator and threatens action",
    payment_request: "asks to pay a fee, tax, deposit or transfer money",
    education: "explains a concept or warns about scams without selling anything",
    other: "anything else",
  },
};

/** Stage-B follow-ups chosen by the triage answer (never a flat 12-question batch). */
export const CLAIM_V1_FOLLOWUPS: Record<string, QuestionSet> = {
  investment_offer: {
    guaranteed: { type: "noul", instructions: "Does the message promise a guaranteed or risk-free return?" },
    pay_to_withdraw: { type: "noul", instructions: "Must the reader pay money before they can withdraw profits?" },
    unregistered_group: { type: "noul", instructions: "Does it invite the reader into a private tips or VIP group?" },
  },
  authority_threat: {
    digital_arrest: { type: "noul", instructions: "Is the reader told to stay on a call or video under threat of arrest?" },
    secrecy: { type: "noul", instructions: "Is the reader told not to tell family or anyone else?" },
    transfer: { type: "noul", instructions: "Is the reader asked to transfer money to 'verify' or 'clear' their name?" },
  },
  payment_request: {
    pay_to_withdraw: { type: "noul", instructions: "Is the payment described as a tax or fee to release the reader's own money?" },
    remote_app: { type: "noul", instructions: "Is the reader asked to install an app or share their screen?" },
  },
  education: {
    selling: { type: "noul", instructions: "Despite sounding educational, does it push a specific product, stock or paid group?" },
  },
  other: {},
};

/** Choose follow-ups within the device's question budget, skipping ones Tier-0 already proved. */
export function planFollowups(triage: string, alreadyFlagged: Set<string>, budget: number): QuestionSet {
  const pool = CLAIM_V1_FOLLOWUPS[triage] ?? {};
  const out: QuestionSet = {};
  for (const [k, q] of Object.entries(pool)) {
    if (Object.keys(out).length >= budget) break;
    if (alreadyFlagged.has(k)) continue;
    out[k] = q;
  }
  return out;
}
