/**
 * tokenizer.ts — dependency-free HF `tokenizer.json` encoder for the Laya checkpoints.
 *
 * Ported from Laya's own TypeScript port (laya-ts/src/tokenizer.ts, Apache-2.0,
 * https://github.com/NandhaKishorM/laya) so Kasauti's inference path needs no third-party
 * runtime: the worker parses the published `tokenizer.json` (already vocabulary-pruned by
 * ml/kasauti_ml/export.py) and encodes text itself. That removes transformers.js, the CDN
 * dependency, the Metaspace `prepend_scheme` patch and ~35 MB of duplicate downloads — and it
 * makes tokenization identical to the Python `tokenizers` library the model was trained with,
 * which is the only thing that actually matters for accuracy.
 *
 * What it implements, all of it read out of tokenizer.json — never assumed:
 *   * Metaspace (SentencePiece-style) pre-tokenization with the "▁" word marker
 *   * ByteLevel/GPT-2 pre-tokenization with the byte↔unicode map (when the checkpoint uses it)
 *   * rank-ordered BPE merges, with the O(n log n) heap path for long pieces (an unspaced
 *     Chinese/Japanese state is ONE piece, and the rescan is quadratic there)
 *   * `byte_fallback` → `<0xNN>` tokens instead of a single unk
 *   * normalizer `Replace` rules, in order
 *   * `added_tokens` cut out of the raw text by a length-ordered trie, in HF's two phases
 *     (non-`normalized` tokens first on raw text, then `normalized` ones on the remainder)
 *
 * Verified against the real checkpoint by `tests/tokenizer-parity.test.ts`, which replays id
 * sequences produced by Python `tokenizers` for the same texts.
 */

export interface TokenizerLike {
  readonly clsId: number;
  readonly sepId: number;
  readonly maskId: number;
  readonly padId: number;
  readonly maskToken: string;
  encode(text: string): number[];
}

/** Special ids of the Laya ModernBERT checkpoints, used only when an alias is missing from the file. */
export const CHECKPOINT_IDS = { cls: 50281, sep: 50282, mask: 50284, pad: 50283, unk: 50280 } as const;

/** Alias lookup order per special: ModernBERT `[X]` names first, Gemma `<x>` names after. */
export const SPECIAL_ALIASES = {
  cls: ["[CLS]", "<bos>", "<s>"],
  sep: ["[SEP]", "<eos>", "</s>"],
  pad: ["[PAD]", "<pad>"],
  mask: ["[MASK]", "<mask>"],
  unk: ["[UNK]", "<unk>"],
} as const;

export interface TokenizerIds {
  cls: number;
  sep: number;
  mask: number;
  pad: number;
  unk: number;
}
export type PreTokenizerKind = "metaspace" | "bytelevel";

export interface TokenizerData {
  vocab: Map<string, number>;
  merges: Map<string, number>;
  ids: TokenizerIds;
  kind: PreTokenizerKind;
  maskToken: string;
  /** Normalizer Replace rules (pattern → content) applied in order before pre-tokenizing. */
  replaces: Array<[string, string]>;
  /** A character the vocab lacks becomes its `<0xNN>` byte tokens instead of unk (HF byte_fallback). */
  byteFallback?: boolean;
  /** Tokens cut out of the text before it is tokenized (HF added_tokens). */
  added?: AddedToken[];
}

export interface AddedToken {
  content: string;
  id: number;
  normalized: boolean;
  lstrip: boolean;
  rstrip: boolean;
}

/** Metaspace word-boundary marker (HF SentencePiece-style replacement for ' '). */
export const METASPACE_REPLACEMENT = "▁";

/** GPT-2 byte↔unicode table (same mapping as HF's ByteLevel pre-tokenizer). */
function byteUnicodeMaps(): { b2u: Map<number, string>; u2b: Map<string, number> } {
  const b2u = new Map<number, string>();
  const u2b = new Map<string, number>();
  const extra = (n: number): number => (n < 0x100 ? n + 0x100 : n);
  const ranges: Array<[number, number]> = [
    [0x21, 0x7e],
    [0xa1, 0xac],
    [0xae, 0xff],
  ];
  let k = 0;
  const inRange = (b: number): boolean => ranges.some(([lo, hi]) => b >= lo && b <= hi);
  for (let b = 0; b < 256; b++) {
    const cp = inRange(b) ? b : extra(k++);
    b2u.set(b, String.fromCodePoint(cp));
    u2b.set(String.fromCodePoint(cp), b);
  }
  return { b2u, u2b };
}

let cached: { b2u: Map<number, string>; u2b: Map<string, number> } | null = null;
function maps(): { b2u: Map<number, string>; u2b: Map<string, number> } {
  if (!cached) cached = byteUnicodeMaps();
  return cached;
}

/** GPT-2 pre-tokenizer split (HF ByteLevel use_regex=true). */
const GPT2_SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/**
 * Piece length at or above which `bpeWord` switches to the heap form. Real prose pieces are
 * ~5 bytes, so ordinary text never reaches the new code path; the switch exists for the case
 * that actually hurts — an unspaced CJK state arrives as a single piece, where the rescan is
 * quadratic (measured 1.4 s for 8 000 unspaced characters against 1.9 ms for English prose,
 * growing 4x per doubling, on the event loop).
 */
const HEAP_MIN_LEN = 32;

/** Merge one word's chars greedily by merge rank: lowest rank wins, leftmost breaks a tie. */
function bpeWord(chars: string[], rank: Map<string, number>): string[] {
  if (chars.length >= HEAP_MIN_LEN) return bpeWordHeap(chars, rank);
  let word = chars.slice();
  if (word.length <= 1) return word;
  for (;;) {
    let best = Infinity;
    let idx = -1;
    for (let i = 0; i < word.length - 1; i++) {
      const r = rank.get(word[i] + " " + word[i + 1]);
      if (r !== undefined && r < best) {
        best = r;
        idx = i;
      }
    }
    if (idx < 0) return word;
    word = [...word.slice(0, idx), word[idx] + word[idx + 1], ...word.slice(idx + 2)];
  }
}

/**
 * The same merge rule in O(n log n): a doubly linked list over the slots plus a binary min-heap
 * of candidate pairs keyed on (rank, slot). A slot only ever absorbs its right neighbour, so
 * slot indices stay in text order and ordering ties by slot reproduces the leftmost tie-break.
 * Lazy deletion (a version counter per slot) avoids heap removals.
 */
function bpeWordHeap(chars: string[], rank: Map<string, number>): string[] {
  const n = chars.length;
  const tok = chars.slice();
  if (n <= 1) return tok;

  const next = new Int32Array(n);
  const prev = new Int32Array(n);
  const version = new Int32Array(n);
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    prev[i] = i - 1;
    next[i] = i + 1 < n ? i + 1 : -1;
  }

  // 3n is a bound: n-1 initial offers, at most n-1 merges each offering at most two pairs.
  // The grow below still exists because an out-of-bounds typed-array write is silently dropped
  // in JS, which would lose a merge and emit wrong tokens with no error at all.
  let heapCap = 3 * n;
  let heapRank = new Int32Array(heapCap);
  let heapSlot = new Int32Array(heapCap);
  let heapVer = new Int32Array(heapCap);
  let heapLen = 0;

  /** Offer the pair starting at `slot`; sifts up by moving the hole down, one write per level. */
  const offer = (slot: number): void => {
    const j = next[slot];
    if (j < 0) return;
    const r = rank.get(tok[slot] + " " + tok[j]);
    if (r === undefined) return;
    if (heapLen === heapCap) {
      heapCap *= 2;
      const r2 = new Int32Array(heapCap);
      r2.set(heapRank);
      heapRank = r2;
      const s2 = new Int32Array(heapCap);
      s2.set(heapSlot);
      heapSlot = s2;
      const v2 = new Int32Array(heapCap);
      v2.set(heapVer);
      heapVer = v2;
    }
    const ver = version[slot];
    let c = heapLen++;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (heapRank[p] < r || (heapRank[p] === r && heapSlot[p] < slot)) break;
      heapRank[c] = heapRank[p];
      heapSlot[c] = heapSlot[p];
      heapVer[c] = heapVer[p];
      c = p;
    }
    heapRank[c] = r;
    heapSlot[c] = slot;
    heapVer[c] = ver;
  };

  for (let i = 0; i < n - 1; i++) offer(i);

  while (heapLen > 0) {
    const slot = heapSlot[0];
    const ver = heapVer[0];

    const lastRank = heapRank[--heapLen];
    const lastSlot = heapSlot[heapLen];
    const lastVer = heapVer[heapLen];
    if (heapLen > 0) {
      let c = 0;
      for (;;) {
        const l = 2 * c + 1;
        if (l >= heapLen) break;
        const r = l + 1;
        let m = l;
        if (
          r < heapLen &&
          (heapRank[r] < heapRank[l] || (heapRank[r] === heapRank[l] && heapSlot[r] < heapSlot[l]))
        )
          m = r;
        if (heapRank[m] > lastRank || (heapRank[m] === lastRank && heapSlot[m] > lastSlot)) break;
        heapRank[c] = heapRank[m];
        heapSlot[c] = heapSlot[m];
        heapVer[c] = heapVer[m];
        c = m;
      }
      heapRank[c] = lastRank;
      heapSlot[c] = lastSlot;
      heapVer[c] = lastVer;
    }

    if (dead[slot] || version[slot] !== ver) continue; // superseded by a later merge
    const j = next[slot];
    if (j < 0 || dead[j]) continue;

    tok[slot] = tok[slot] + tok[j];
    dead[j] = 1;
    const k = next[j];
    next[slot] = k;
    if (k >= 0) prev[k] = slot;

    version[slot]++;
    offer(slot);
    const p = prev[slot];
    if (p >= 0) {
      version[p]++;
      offer(p);
    }
  }

  const out: string[] = [];
  for (let i = 0; i >= 0; i = next[i]) out.push(tok[i]);
  return out;
}

/** Byte-level BPE encode: NFC-normalize, NO lowercasing, GPT-2 byte map + rank-order merges. */
let sharedEncoder: TextEncoder | null = null;
export function bpeEncode(vocab: Map<string, number>, merges: Map<string, number>, text: string): number[] {
  const { b2u } = maps();
  const unkId = vocab.get("[UNK]") ?? CHECKPOINT_IDS.unk;
  const out: number[] = [];
  const enc = (sharedEncoder ??= new TextEncoder());
  const parts = text.normalize("NFC").match(GPT2_SPLIT);
  if (!parts) return out;
  for (const piece of parts) {
    const chars: string[] = [];
    for (const b of enc.encode(piece)) chars.push(b2u.get(b) ?? "");
    for (const tok of bpeWord(chars, merges)) out.push(vocab.get(tok) ?? unkId);
  }
  return out;
}

/**
 * Metaspace (SentencePiece-style) BPE encode: unicode chars, NO byte map, NO lowercasing.
 * Normalizer replaces run first, then the text is cut into words at each "▁" marker (the marker
 * stays as the word prefix) with maximal `\n` runs as their own pieces.
 */
export function metaspaceEncode(
  vocab: Map<string, number>,
  merges: Map<string, number>,
  text: string,
  unkId?: number,
  replaces: ReadonlyArray<readonly [string, string]> = [[" ", METASPACE_REPLACEMENT]],
  byteFallback = false,
): number[] {
  const unk = unkId ?? vocab.get("<unk>") ?? vocab.get("[UNK]") ?? CHECKPOINT_IDS.unk;
  if (!text) return [];
  let t = text;
  for (const [from, to] of replaces) t = t.split(from).join(to);
  const out: number[] = [];
  const push = (piece: string): void => {
    for (const tok of bpeWord(Array.from(piece), merges)) {
      const id = vocab.get(tok);
      if (id !== undefined) {
        out.push(id);
        continue;
      }
      const bytes = byteFallback
        ? Array.from((sharedEncoder ??= new TextEncoder()).encode(tok), (b) =>
            vocab.get(`<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`),
          )
        : [];
      if (bytes.length > 0 && bytes.every((b) => b !== undefined)) out.push(...(bytes as number[]));
      else out.push(unk);
    }
  };
  for (const seg of t.split(/(\n+)/)) {
    if (!seg) continue;
    if (seg[0] === "\n") {
      push(seg);
    } else {
      const w = seg.startsWith(METASPACE_REPLACEMENT) ? seg : METASPACE_REPLACEMENT + seg;
      for (const chunk of w.split(METASPACE_REPLACEMENT).slice(1)) {
        push(chunk ? METASPACE_REPLACEMENT + chunk : METASPACE_REPLACEMENT);
      }
    }
  }
  return out;
}

function encodeSegment(data: TokenizerData, text: string): number[] {
  return data.kind === "metaspace"
    ? metaspaceEncode(data.vocab, data.merges, text, data.ids.unk, data.replaces, data.byteFallback)
    : bpeEncode(data.vocab, data.merges, text);
}

/**
 * Regex source matching `words` longest-first, as a prefix tree: one branch per next character
 * rather than one alternative per token, so a scan does not try every token at every position.
 */
function trieSource(words: string[]): string {
  type Node = { kids: Map<string, Node>; end: boolean };
  const root: Node = { kids: new Map(), end: false };
  for (const w of words) {
    let n = root;
    for (const ch of w) {
      let k = n.kids.get(ch);
      if (!k) n.kids.set(ch, (k = { kids: new Map(), end: false }));
      n = k;
    }
    n.end = true;
  }
  const walk = (n: Node): string => {
    const alts = [...n.kids].map(([c, k]) => c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + walk(k));
    if (!alts.length) return "";
    const body = alts.length === 1 ? alts[0] : "(?:" + alts.join("|") + ")";
    return n.end ? "(?:" + body + ")?" : body;
  };
  return walk(root);
}

type AddedPhase = { re: RegExp; byContent: Map<string, AddedToken>; normalized: boolean };
const addedPhases = new WeakMap<TokenizerData, AddedPhase[]>();

/** HF matches added tokens that are not `normalized` on the raw text first, then `normalized` ones. */
function phasesOf(data: TokenizerData): AddedPhase[] {
  let phases = addedPhases.get(data);
  if (!phases) {
    phases = [false, true].flatMap((normalized) => {
      const toks = (data.added ?? []).filter((t) => t.normalized === normalized && t.content);
      if (!toks.length) return [];
      const re = new RegExp(trieSource(toks.map((t) => t.content)), "gu");
      return [{ re, byContent: new Map(toks.map((t) => [t.content, t])), normalized }];
    });
    addedPhases.set(data, phases);
  }
  return phases;
}

function splitOnAdded(text: string, { re, byContent }: AddedPhase): Array<string | AddedToken> {
  const out: Array<string | AddedToken> = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    const tok = byContent.get(m[0])!;
    let before = text.slice(last, m.index);
    if (tok.lstrip) before = before.trimEnd();
    if (before) out.push(before);
    out.push(tok);
    last = m.index + m[0].length;
    if (tok.rstrip) while (last < text.length && /\s/u.test(text[last])) last++;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** Dispatch to the Metaspace or GPT-2/ByteLevel encoder, after cutting out added tokens as HF does. */
export function encodeWithData(data: TokenizerData, text: string): number[] {
  let pieces: Array<string | AddedToken> = [text];
  for (const phase of phasesOf(data)) {
    pieces = pieces.flatMap((p) => {
      if (typeof p !== "string") return [p];
      return splitOnAdded(phase.normalized && data.kind === "bytelevel" ? p.normalize("NFC") : p, phase);
    });
  }
  const out: number[] = [];
  for (const p of pieces) {
    if (typeof p === "string") out.push(...encodeSegment(data, p));
    else out.push(p.id);
  }
  return out;
}

function childNodes(node: unknown): unknown[] {
  if (!node || typeof node !== "object") return [];
  const o = node as Record<string, unknown>;
  const out: unknown[] = [];
  for (const k of ["normalizers", "pre_tokenizers", "decoders"]) {
    const v = o[k];
    if (Array.isArray(v)) out.push(...v);
  }
  return out;
}

/** True when a normalizer/pre-tokenizer/decoder node (or nested Sequence member) has a type. */
function hasNodeType(node: unknown, want: string): boolean {
  if (!node || typeof node !== "object") return false;
  if ((node as Record<string, unknown>)["type"] === want) return true;
  return childNodes(node).some((c) => hasNodeType(c, want));
}

/** Collect normalizer Replace rules ({pattern: {String}, content}) in order. */
function collectReplaces(node: unknown, out: Array<[string, string]>): void {
  if (!node || typeof node !== "object") return;
  const o = node as Record<string, unknown>;
  if (o["type"] === "Replace") {
    const pat = o["pattern"] as Record<string, unknown> | undefined;
    const from = pat?.["String"];
    const to = o["content"];
    if (typeof from === "string" && typeof to === "string") out.push([from, to]);
  }
  for (const c of childNodes(node)) collectReplaces(c, out);
}

/** Parse an HF tokenizer.json ({model vocab/merges, normalizer, pre_tokenizer, added_tokens}). */
export function parseTokenizerJson(raw: unknown): TokenizerData | null {
  try {
    const r = raw as {
      model?: {
        vocab?: Record<string, number>;
        merges?: Array<string | [string, string]>;
        byte_fallback?: boolean;
      };
      normalizer?: unknown;
      pre_tokenizer?: unknown;
      added_tokens?: Array<{
        id?: number;
        content?: string;
        normalized?: boolean;
        lstrip?: boolean;
        rstrip?: boolean;
      }>;
    };
    const vocabObj = r?.model?.vocab;
    if (!vocabObj || typeof vocabObj !== "object") return null;
    const vocab = new Map(Object.entries(vocabObj));
    const merges = new Map<string, number>();
    for (const [i, m] of (r.model?.merges ?? []).entries()) {
      const pair = typeof m === "string" ? m.split(" ") : m;
      if (pair.length >= 2) merges.set(pair[0] + " " + pair[1], i);
    }
    const added = new Map<string, number>();
    const addedTokens: AddedToken[] = [];
    for (const t of r.added_tokens ?? []) {
      if (typeof t?.content === "string" && typeof t?.id === "number") {
        added.set(t.content, t.id);
        addedTokens.push({
          content: t.content,
          id: t.id,
          normalized: t.normalized === true,
          lstrip: t.lstrip === true,
          rstrip: t.rstrip === true,
        });
      }
    }
    const pick = (aliases: readonly string[], fb: number): { id: number; token: string } => {
      for (const a of aliases) {
        const v = added.get(a) ?? vocab.get(a);
        if (v !== undefined) return { id: v, token: a };
      }
      return { id: fb, token: aliases[0] };
    };
    const cls = pick(SPECIAL_ALIASES.cls, CHECKPOINT_IDS.cls);
    const sep = pick(SPECIAL_ALIASES.sep, CHECKPOINT_IDS.sep);
    const mask = pick(SPECIAL_ALIASES.mask, CHECKPOINT_IDS.mask);
    const pad = pick(SPECIAL_ALIASES.pad, CHECKPOINT_IDS.pad);
    const unk = pick(SPECIAL_ALIASES.unk, CHECKPOINT_IDS.unk);
    const kind: PreTokenizerKind = hasNodeType(r?.pre_tokenizer, "Metaspace") ? "metaspace" : "bytelevel";
    const replaces: Array<[string, string]> = [];
    collectReplaces(r?.normalizer, replaces);
    if (kind === "metaspace" && replaces.length === 0) replaces.push([" ", METASPACE_REPLACEMENT]);
    return {
      vocab,
      merges,
      ids: { cls: cls.id, sep: sep.id, mask: mask.id, pad: pad.id, unk: unk.id },
      kind,
      maskToken: mask.token,
      replaces,
      byteFallback: r.model?.byte_fallback === true,
      added: addedTokens,
    };
  } catch {
    return null;
  }
}

/* --------------------------------- Kasauti adapter --------------------------------- */

/** What the Laya sequence builder needs, plus the bits the UI reports. */
export interface LayaTokenizer {
  encode(text: string): number[];
  cls: number;
  sep: number;
  mask: number;
  pad: number;
  unk: number;
  /** literal mask token string (e.g. "<mask>") — scrubbed from all text, as the reference does */
  maskToken: string;
  vocabSize: number;
  kind: PreTokenizerKind;
  ids: TokenizerIds;
}

/** Wrap parsed tokenizer data in the runtime interface. */
export function createTokenizer(data: TokenizerData): LayaTokenizer {
  return {
    encode: (text: string) => encodeWithData(data, text),
    cls: data.ids.cls,
    sep: data.ids.sep,
    mask: data.ids.mask,
    pad: data.ids.pad,
    unk: data.ids.unk,
    maskToken: data.maskToken,
    vocabSize: data.vocab.size,
    kind: data.kind,
    ids: data.ids,
  };
}

/** Fetch + parse a tokenizer.json. `bytes` is returned so callers can report/hash what they loaded. */
export async function loadTokenizer(
  url: string,
  init?: RequestInit,
): Promise<{ tokenizer: LayaTokenizer; data: TokenizerData; bytes: number }> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`tokenizer fetch failed: HTTP ${res.status} for ${url}`);
  const text = await res.text();
  const data = parseTokenizerJson(JSON.parse(text));
  if (!data) throw new Error(`could not parse tokenizer.json from ${url}`);
  return { tokenizer: createTokenizer(data), data, bytes: text.length };
}
