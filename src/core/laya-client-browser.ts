/**
 * laya-client-browser — the client-side contract for Laya (convaiinnovations/laya).
 *
 * Mirrors the reference Python implementation exactly where it matters for an ONNX port:
 *   sequence  = [CLS] "<type> question: <ins>" [SEP] ([MASK] opt)* [SEP] state [SEP]
 *   noul      = options rendered as ["false: …", "true: …"]  → p[1] is the noul probability
 *   score     = options rendered as "level i: …"
 *   confidence= 1 − H(p)/log(k)            (normalised entropy, what ships today)
 *   decode    = softmax(logits[:k] / T_bucket) then un-permute, exactly like onnx_agent._decode_answers
 *
 * Laya answers ONLY flat choice / score / noul questions, all of them in one forward pass. Question
 * trees are adaptive: one triage `choice`, then a budgeted set of targeted follow-ups (see
 * `src/lib/banks.ts`).
 *
 * The model deliberately does NOT expose `act_probability` as a gate: the published card and issue
 * #185 are explicit that it carries no usable signal. Kasauti gates on calibrated `confidence`.
 */
import { BANKS, type Question, type QuestionSet, type QType } from "../lib/banks";

export type { Question, QuestionSet, QType };

/** One published artifact: content-hashed filename plus the digest the worker verifies on load. */
export interface ManifestFile {
  path: string;
  bytes?: number;
  sha256?: string;
}

export interface CalibrationManifest {
  /** schema of the manifest this was parsed from: 2 = published by ml/kasauti_ml (calibrated) */
  schema: number;
  modelVersion: string;
  maxLen: number;
  headMaxLen: number;
  /** per-type floor temperature, from fit_temperature_map() */
  temperatures: number[];
  /** bucket key "<type>:<2|3-5|6-10|11+>" → T  (may also carry "default") */
  temperatureByBucket: Record<string, number>;
  /** bucket key → minimum calibrated confidence to act (may also carry "default") */
  abstain: Record<string, number>;
  /** which graph to fetch per backend, plus the fp32 reference */
  files: Record<string, ManifestFile>;
  tokenizerPath: string;
  tokenizerSha256?: string;
  /** special ids the export recorded — cross-checked against the tokenizer file before running */
  tokenizerSpecials?: Record<string, number>;
  parityPath?: string;
  banksSha256?: string;
  qtypes: string[];
  evals?: Record<string, number | null>;
  createdUtc?: string;
  /** null when the manifest was missing or unreadable — the caller then runs Tier-0 only */
  /**
   * True when the export itself declared `reportOnly` (short of the eval gates, or a smoke/dev
   * build). Distinct from `!calibrated`: a schema-1 manifest has no calibration at all, while a
   * report-only export may carry perfectly good temperatures that simply failed the quality bar.
   */
  reportOnly: boolean;
  /** true when the published calibration is missing, so answers are report-only, never acted on */
  calibrated: boolean;
}

/** Bucket key for the (type, option-count) pair the calibration was fitted on. */
export function bucket(type: QType, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${type}:${size}`;
}

/**
 * Defaults act as the floor of a circuit breaker, not as a working configuration: the shipped
 * checkpoints are over-confident (ECE 0.314 multilingual) so an uncalibrated answer must not be
 * allowed to trigger an action. These thresholds match the values in the model card's examples.
 */
export const DEFAULT_ABSTAIN: Record<string, number> = {
  "noul:2": 0.35,
  "choice:2": 0.35,
  "choice:3-5": 0.3,
  "choice:6-10": 0.25,
  "choice:11+": 0.2,
  "score:3-5": 0.3,
  default: 0.35,
};

/** Used before (or instead of) a published manifest: Tier-1 answers are reported, never acted on. */
export const DEFAULT_MANIFEST: CalibrationManifest = {
  schema: 1,
  modelVersion: "laya-multilingual@base (uncalibrated)",
  maxLen: 1024,
  headMaxLen: 256,
  temperatures: [1, 1, 1],
  temperatureByBucket: {},
  abstain: DEFAULT_ABSTAIN,
  files: {},
  tokenizerPath: "tokenizer.json",
  qtypes: ["choice", "score", "noul"],
  calibrated: false,
  reportOnly: true,
};

/* eslint-disable @typescript-eslint/no-explicit-any */
/** Parse a published manifest (schema 1 legacy or schema 2) into the runtime contract. */
export function parseManifest(raw: any): CalibrationManifest {
  if (!raw || typeof raw !== "object") return DEFAULT_MANIFEST;
  const schema = Number(raw.schema ?? 1);
  const legacyTemps: Record<string, number> = raw.temperatures && typeof raw.temperatures === "object" ? raw.temperatures : {};
  const byBucket: Record<string, number> = { ...legacyTemps, ...(raw.temperatureByBucket ?? {}) };
  const abort = raw.abstain && typeof raw.abstain === "object" ? raw.abstain : {};
  const filesIn = raw.files && typeof raw.files === "object" ? raw.files : {};
  // schema 2 publishes {path,bytes,sha256}; schema 1 was a bare filename string — accept both
  const files: Record<string, ManifestFile> = {};
  for (const [k, v] of Object.entries(filesIn as Record<string, unknown>)) {
    if (typeof v === "string") files[k] = { path: v };
    else if (v && typeof v === "object") files[k] = v as ManifestFile;
  }
  const tok = raw.tokenizer && typeof raw.tokenizer === "object" ? raw.tokenizer : {};
  const temps: number[] = Array.isArray(raw.temperature) ? raw.temperature.map(Number) : [1, 1, 1];
  // `reportOnly` is the export's own admission that it did not clear the eval gates (the smoke /
  // dev profile, or a deliberately uncalibrated build). It is an override, not a hint: report-only
  // means the runtime abstains on everything and the UI says so.
  const reportOnly = raw.reportOnly === true;
  const calibrated = schema >= 2 && Object.keys(byBucket).length > 0 && !reportOnly;
  return {
    schema,
    modelVersion: String(raw.modelVersion ?? DEFAULT_MANIFEST.modelVersion),
    maxLen: Number(raw.maxLen ?? DEFAULT_MANIFEST.maxLen),
    headMaxLen: Number(raw.headMaxLen ?? DEFAULT_MANIFEST.headMaxLen),
    temperatures: [Number(temps[0] ?? 1), Number(temps[1] ?? 1), Number(temps[2] ?? 1)],
    temperatureByBucket: byBucket,
    // the manifest's own map only — merging the defaults here would create phantom exact-bucket
    // entries that shadow a manifest-level "default" (e.g. an export that abstains on everything)
    abstain: { ...abort },
    files,
    tokenizerPath: String(tok.path ?? raw.tokenizerPath ?? "tokenizer.json"),
    tokenizerSha256: tok.sha256 ? String(tok.sha256) : undefined,
    tokenizerSpecials: tok.specials && typeof tok.specials === "object" ? tok.specials : undefined,
    parityPath: raw.parity?.path ? String(raw.parity.path) : raw.parityPath,
    banksSha256: raw.banks?.sha256 ? String(raw.banks.sha256) : undefined,
    qtypes: Array.isArray(raw.qtypes) ? raw.qtypes.map(String) : DEFAULT_MANIFEST.qtypes,
    evals: raw.evals && typeof raw.evals === "object" ? raw.evals : undefined,
    createdUtc: raw.createdUtc ? String(raw.createdUtc) : undefined,
    calibrated,
    reportOnly,
  };
}

/* ------------------------------ question rendering ------------------------------ */

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

export const QTYPES: Record<string, number> = { choice: 0, score: 1, noul: 2 };

/* --------------------------------- sequence --------------------------------- */

export type Tokenizer = {
  encode(text: string): number[];
  cls: number;
  sep: number;
  mask: number;
  pad: number;
  /** literal mask token string (e.g. "<mask>") — scrubbed from all text like the reference implementation */
  maskToken?: string;
};

/** Port of build_sequence(): returns input ids and per-option [MASK] marker positions. */
export function buildSequence(
  tok: Tokenizer,
  state: string,
  q: Question,
  maxLen: number,
  headMaxLen: number,
  truncateLeft = false,
) {
  const scrub = (t: string) => (tok.maskToken ? t.split(tok.maskToken).join(" ") : t);
  const opts = renderOptions(q);
  let head = tok.encode(`${q.type} question: ${scrub(q.instructions)}`);
  let optIds = opts.map((o) => [tok.mask, ...tok.encode(" " + scrub(o)).slice(0, 48)]);
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
  const st = tok.encode(scrub(state));
  ids.push(...(truncateLeft ? st.slice(-room) : st.slice(0, room)), tok.sep); // mirrors Python st[-room:] / st[:room]
  return { ids: ids.slice(0, maxLen), markers: markers.filter((m) => m < maxLen) };
}

/* ---------------------------------- decode ---------------------------------- */

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

/** Temperature for a bucket: exact bucket → "default" → per-type floor → 1.0 (upstream lookup order). */
export function temperatureFor(m: CalibrationManifest, type: QType, k: number): number {
  const b = bucket(type, k);
  const t = m.temperatureByBucket[b] ?? m.temperatureByBucket.default ?? m.temperatures[QTYPES[type] ?? 0] ?? 1;
  return Number.isFinite(t) && t > 0 ? t : 1;
}

/**
 * Abstain threshold for a bucket. Manifest first (exact bucket → "default"), then the documented
 * defaults as a floor — never the reverse, or a manifest that only widens thresholds would be
 * silently ignored for every bucket it did not name.
 */
export function abstainFor(m: CalibrationManifest, type: QType, k: number): number {
  const b = bucket(type, k);
  return m.abstain[b] ?? m.abstain.default ?? DEFAULT_ABSTAIN[b] ?? DEFAULT_ABSTAIN.default;
}

/** Reference `unpermute_probs`: the model sees a permuted option order, answers come back to the caller's. */
export function unpermuteProbs(p: ArrayLike<number>, optionOrder: number[] | null | undefined): number[] {
  const out = new Array<number>(p.length).fill(0);
  if (!optionOrder) return Array.from(p);
  for (let slot = 0; slot < p.length && slot < optionOrder.length; slot++) out[optionOrder[slot]] = p[slot];
  return out;
}

export interface Answer {
  type: QType;
  /** option key → probability (choice/noul), or level → probability (score) */
  probs: Record<string, number>;
  choice?: string;
  score?: number;
  noul?: number;
  confidence: number;
  /** entropy confidence below the calibrated bucket threshold — report it, never act on it */
  abstained: boolean;
  /** the asked question, so callers can render "why" without a second lookup */
  question: Question;
  bucket: string;
  temperature: number;
}

/**
 * Calibrated decode of one question's marker logits.
 * `optionOrder` is the permutation the sequence was built with (datagen augmentation / caller shims).
 */
export function decode(
  q: Question,
  logits: ArrayLike<number>,
  m: CalibrationManifest = DEFAULT_MANIFEST,
  optionOrder?: number[] | null,
): Answer {
  const keys = optionKeys(q);
  const k = keys.length;
  const b = bucket(q.type, k);
  const T = temperatureFor(m, q.type, k);
  const p = unpermuteProbs(softmax(logits, T), optionOrder);
  const conf = entropyConfidence(p);
  const probs: Record<string, number> = {};
  keys.forEach((key, i) => (probs[key] = p[i]));
  let arg = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[arg]) arg = i;
  const a: Answer = {
    type: q.type,
    probs,
    confidence: conf,
    abstained: conf < abstainFor(m, q.type, k),
    question: q,
    bucket: b,
    temperature: T,
  };
  if (q.type === "choice") a.choice = keys[arg];
  if (q.type === "score") a.score = p.reduce((s, v, i) => s + v * i, 0);
  if (q.type === "noul") a.noul = p[1];
  return a;
}

/** Decode a whole batch: rows of logits laid out [B, K] with K = max option count. */
export function decodeBatch(
  questions: QuestionSet,
  logits: ArrayLike<number>,
  k: number,
  m: CalibrationManifest = DEFAULT_MANIFEST,
): Record<string, Answer> {
  const out: Record<string, Answer> = {};
  Object.entries(questions).forEach(([name, q], b) => {
    const n = optionKeys(q).length;
    out[name] = decode(q, Array.prototype.slice.call(logits, b * k, b * k + n), m);
  });
  return out;
}

/* ------------------------------- legacy aliases ------------------------------- */

/**
 * @deprecated Read the banks from `src/lib/banks.ts` (generated from ml/banks.json). These names
 * survive only so existing call sites keep compiling; both are views of the same generated data.
 */
export const CLAIM_V1_TRIAGE: Question = BANKS.claim_v1.triage;
/** @deprecated see CLAIM_V1_TRIAGE */
export const CLAIM_V1_FOLLOWUPS: Record<string, QuestionSet> = BANKS.claim_v1.followups as Record<string, QuestionSet>;

/**
 * @deprecated use `bankFollowups("claim_v1", …)` from src/lib/banks.ts
 * Kept as a thin delegate so Tier-0-skip + budget behaviour is identical everywhere.
 */
export function planFollowups(triage: string, alreadyFlagged: Set<string>, budget: number): QuestionSet {
  const pool: QuestionSet = (BANKS.claim_v1.followups as Record<string, QuestionSet | undefined>)[triage] ?? {};
  const out: QuestionSet = {};
  for (const [k, q] of Object.entries(pool)) {
    if (Object.keys(out).length >= budget) break;
    if (alreadyFlagged.has(k)) continue;
    out[k] = q;
  }
  return out;
}

/**
 * Parity fix for transformers.js ≤3.x vs HF `tokenizers` (Rust) on Laya's mmBERT/Gemma-style tokenizer.
 *
 * @deprecated Kept only for the historical test and for anyone still loading transformers.js: the
 * runtime now uses src/core/tokenizer.ts, which is verified id-for-id against the Rust library, so
 * no patch is needed. See tests/tokenizer-parity.test.ts.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function patchTokenizerJSON(tj: any): any {
  const fix = (p: { type?: string; prepend_scheme?: string; add_prefix_space?: boolean } | null | undefined) => {
    if (p && p.type === "Metaspace" && p.add_prefix_space === undefined)
      p.add_prefix_space = (p.prepend_scheme ?? "always") !== "never";
  };
  fix(tj?.pre_tokenizer);
  if (Array.isArray(tj?.pre_tokenizer?.pretokenizers)) tj.pre_tokenizer.pretokenizers.forEach(fix);
  return tj;
}
