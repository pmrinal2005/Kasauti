/**
 * Long-transcript windowing.
 *
 * Laya answers about *one* state at a time and the sequence builder has a hard context budget, so a
 * 40-minute call transcript cannot be handed over whole. This module cuts it into overlapping windows
 * at the most meaningful boundary it can find — speaker turns first, then blank lines, then sentences
 * — and reports honestly what it dropped. It is pure and synchronous: every rule is visible and
 * unit-tested, and nothing here talks to a model.
 *
 * The two failure modes this design is trying to avoid:
 *   * cutting mid-sentence, which changes what a "no" means;
 *   * silently truncating a long transcript, which makes a scam that lives at the end invisible.
 */

export interface Window {
  index: number;
  text: string;
  /** character offsets into the *original* text, so the UI can quote the source */
  start: number;
  end: number;
  /** how many speaker turns / paragraphs this window contains */
  units: number;
}

export interface Segmentation {
  windows: Window[];
  chars: number;
  /** how the boundaries were chosen — shown in the UI, because it changes what the reading means */
  strategy: "single" | "turns" | "paragraphs" | "sentences" | "hard";
  /** windows that did not fit the budget, dropped from the middle */
  dropped: number;
  clipped: boolean;
}

export interface SegmentOptions {
  /** target window size in characters (default 900 — comfortably inside head_max_len=256) */
  maxChars?: number;
  /** how much of the previous window is repeated (default 120) so a cut never splits a claim in two */
  overlapChars?: number;
  /** hard cap on how many windows are analysed (default 8) — bounds the on-device compute */
  maxWindows?: number;
}

const TURN_RE = /(^|\n)\s*(?:[A-Z][\w .'-]{0,24}|speaker\s*\d{1,2}|caller|agent|scammer|victim|customer|advisor)\s*:\s?/gi;

interface Unit {
  text: string;
  start: number;
  end: number;
}

function pushUnits(units: Unit[], text: string, base: number) {
  let m: RegExpExecArray | null;
  TURN_RE.lastIndex = 0;
  const starts: number[] = [];
  while ((m = TURN_RE.exec(text)) !== null) starts.push(m.index + (m[1] ? m[1].length : 0));
  if (starts.length < 2) return false;
  for (let i = 0; i < starts.length; i++) {
    const from = starts[i];
    const to = i + 1 < starts.length ? starts[i + 1] : text.length;
    const slice = text.slice(from, to);
    if (slice.trim()) units.push({ text: slice, start: base + from, end: base + to });
  }
  return true;
}

/**
 * Cut at the given boundary offsets, keeping every character.
 *
 * The units a window is rebuilt from MUST concatenate back to the original text: if the separators
 * (blank lines, trailing spaces) were dropped while packing, `text.slice(start, end)` would stop
 * matching the window and a window boundary would silently delete a line break — and, worse, the
 * offsets the UI quotes back at the user would be wrong.
 */
function sliceUnits(units: Unit[], text: string, base: number, boundaries: number[]) {
  let from = 0;
  for (const to of [...boundaries, text.length]) {
    if (to <= from) continue;
    const slice = text.slice(from, to);
    units.push({ text: slice, start: base + from, end: base + to });
    from = to;
  }
}

function paragraphUnits(units: Unit[], text: string, base: number) {
  const boundaries: number[] = [];
  const re = /\n{2,}/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) boundaries.push(m.index + m[0].length);
  sliceUnits(units, text, base, boundaries);
}

function sentenceUnits(units: Unit[], text: string, base: number) {
  const boundaries: number[] = [];
  // a sentence ends at . ! ? or the Devanagari danda, plus any closing quote/bracket and the
  // whitespace that follows — everything else (including a bare newline) stays inside the unit
  const re = /[.!?।]+["')\]”’]*\s+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) boundaries.push(m.index + m[0].length);
  sliceUnits(units, text, base, boundaries);
}

/** Split one oversized unit on whitespace so a single wall of text still becomes windows. */
function hardSplit(units: Unit[], text: string, base: number, maxChars: number) {
  const approx = Math.max(1, Math.ceil(text.length / maxChars));
  const step = Math.ceil(text.length / approx);
  for (let i = 0; i < text.length; i += step) {
    const slice = text.slice(i, i + step);
    units.push({ text: slice, start: base + i, end: base + i + slice.length });
  }
}

export function segmentTranscript(input: string, opts: SegmentOptions = {}): Segmentation {
  const maxChars = Math.max(120, opts.maxChars ?? 900);
  const overlapChars = Math.max(0, Math.min(opts.overlapChars ?? 120, maxChars - 40));
  const maxWindows = Math.max(1, opts.maxWindows ?? 8);
  const text = String(input ?? "").replace(/\r\n?/g, "\n");
  const chars = text.length;

  if (chars <= maxChars) {
    return {
      windows: text.trim() ? [{ index: 0, text, start: 0, end: chars, units: 1 }] : [],
      chars, strategy: "single", dropped: 0, clipped: false,
    };
  }

  const units: Unit[] = [];
  let strategy: Segmentation["strategy"] = "turns";
  if (!pushUnits(units, text, 0)) {
    paragraphUnits(units, text, 0);
    strategy = units.length > 1 ? "paragraphs" : "sentences";
    if (units.length <= 1) {
      units.length = 0;
      sentenceUnits(units, text, 0);
      strategy = units.length > 1 ? "sentences" : "hard";
      if (units.length <= 1) {
        units.length = 0;
        hardSplit(units, text, 0, maxChars);
        strategy = "hard";
      }
    }
  }

  // pack units into overlapping windows
  const windows: Window[] = [];
  let i = 0;
  while (i < units.length) {
    let end = i;
    let size = 0;
    while (end < units.length && (size === 0 || size + units[end].text.length <= maxChars)) {
      size += units[end].text.length;
      end++;
    }
    if (end === i) {
      // a single unit larger than the whole budget (one unbroken wall of text, no punctuation):
      // replace it with character-sized slices of itself, then pack again from the same index
      const oversize = units[i];
      const pieces: Unit[] = [];
      hardSplit(pieces, oversize.text, oversize.start, maxChars);
      units.splice(i, 1, ...pieces);
      continue;
    }
    const slice = units.slice(i, end);
    windows.push({
      index: windows.length,
      // units are contiguous by construction (see sliceUnits), so joining them reproduces the source
      text: slice.map((u) => u.text).join(""),
      start: slice[0].start,
      end: slice[slice.length - 1].end,
      units: slice.length,
    });
    if (end >= units.length) break;
    // step back far enough to cover overlapChars, but always advance at least one unit
    let back = end;
    let covered = 0;
    while (back > i + 1 && covered + units[back - 1].text.length <= overlapChars) {
      back--;
      covered += units[back].text.length;
    }
    i = back;
  }

  let dropped = 0;
  let clipped = false;
  let kept = windows;
  if (windows.length > maxWindows) {
    // keep the opening and the closing — a script that builds up and then asks for money lives at
    // both ends, and truncation in the middle is exactly the case that would hide it
    const head = Math.ceil(maxWindows / 2);
    const tail = maxWindows - head;
    dropped = windows.length - maxWindows;
    clipped = true;
    kept = [...windows.slice(0, head), ...windows.slice(windows.length - tail)];
    kept = kept.map((w, idx) => ({ ...w, index: idx }));
  }

  return { windows: kept, chars, strategy, dropped, clipped };
}
