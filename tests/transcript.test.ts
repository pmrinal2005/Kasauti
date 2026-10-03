/**
 * Transcript windowing.
 *
 * The rules that matter: never lose the end of a transcript, never cut a claim in half without
 * overlapping, always terminate, and always say what was dropped. A silent truncation here would
 * make a scam that lives in the final minute invisible — the worst possible failure for this app.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { segmentTranscript } from "../src/lib/transcript.ts";

const turn = (n: number, body: string) => `Speaker ${n}: ${body}\n`;

test("a short text is one window and no windowing is claimed", () => {
  const s = segmentTranscript("Pay ₹5,000 to vipfund@ybl for guaranteed 3% daily profit.");
  assert.equal(s.windows.length, 1);
  assert.equal(s.strategy, "single");
  assert.equal(s.clipped, false);
  assert.equal(s.windows[0].text, "Pay ₹5,000 to vipfund@ybl for guaranteed 3% daily profit.");
});

test("empty input produces no windows instead of a phantom one", () => {
  assert.equal(segmentTranscript("").windows.length, 0);
  assert.equal(segmentTranscript("   \n\n  ").windows.length, 0);
});

test("speaker turns are preferred as boundaries", () => {
  const text = Array.from({ length: 12 }, (_, i) => turn(i % 2 ? 1 : 2, `Line number ${i} about the scheme and its returns.`)).join("");
  const s = segmentTranscript(text, { maxChars: 200, overlapChars: 40 });
  assert.equal(s.strategy, "turns");
  assert.ok(s.windows.length > 1);
  // every window starts at a speaker label, i.e. no window begins mid-sentence
  for (const w of s.windows) assert.match(w.text.trimStart().slice(0, 12), /Speaker\s*\d/);
});

test("windows overlap, so a claim split by a boundary appears whole in one of them", () => {
  // ~60-char paragraphs, 420-char windows, 120-char overlap → the step back covers whole units,
  // so consecutive windows genuinely share text
  const text = Array.from({ length: 30 }, (_, i) => `Speaker 1: point number ${i} explained briefly here.\n`).join("");
  const s = segmentTranscript(text, { maxChars: 420, overlapChars: 120 });
  assert.ok(s.windows.length > 1);
  for (let i = 1; i < s.windows.length; i++) {
    assert.ok(s.windows[i].start < s.windows[i - 1].end, `window ${i} does not overlap the previous one`);
  }
  for (const w of s.windows) assert.ok(w.text.length <= 420 + 200, "a window blew past its budget");
});

test("when a single unit is bigger than the overlap budget, windows stay contiguous and lose nothing", () => {
  // this is the honest limit of unit-aligned windowing: the boundary is clean, so the overlap is 0.
  // (Six paragraphs, so the window cap does not clip anything.)
  const text = Array.from({ length: 6 }, (_, i) => `Paragraph ${i}: ` + "word ".repeat(30) + "\n\n").join("");
  const s = segmentTranscript(text, { maxChars: 300, overlapChars: 40 });
  assert.ok(s.windows.length > 1);
  assert.equal(s.clipped, false);
  for (let i = 1; i < s.windows.length; i++) {
    assert.equal(s.windows[i].start, s.windows[i - 1].end, "contiguous: no text is skipped");
  }
  assert.equal(s.windows[0].start, 0);
  assert.equal(s.windows[s.windows.length - 1].end, text.length);
});

test("window offsets are verbatim slices for every strategy (nothing is re-joined badly)", () => {
  const samples = [
    "A: hello\nB: hi\nA: pay now\nB: ok\nA: one more line here\nB: fine\nA: last one\nB: bye\n",
    Array.from({ length: 6 }, (_, i) => `Paragraph ${i}: ` + "word ".repeat(30) + "\n\n").join(""),
    Array.from({ length: 25 }, (_, i) => `Sentence number ${i} has some body text in it. `).join(""),
  ];
  for (const text of samples) {
    const s = segmentTranscript(text, { maxChars: 260, overlapChars: 80 });
    assert.ok(s.windows.length > 0);
    for (const w of s.windows) {
      assert.equal(text.slice(w.start, w.end), w.text, `strategy ${s.strategy}, window ${w.index}`);
    }
  }
});

test("offsets point back into the original text", () => {
  const text = "A: hello there\nB: " + "x".repeat(400) + "\nA: bye now\nB: " + "y".repeat(400) + "\n";
  const s = segmentTranscript(text, { maxChars: 200, overlapChars: 30 });
  for (const w of s.windows) {
    assert.equal(text.slice(w.start, w.end), w.text, "offsets must select the window text verbatim");
  }
});

test("a transcript longer than the budget keeps the first and last windows and says what it dropped", () => {
  const text = Array.from({ length: 200 }, (_, i) => `Speaker 1: turn ${i} with a little body text.\n`).join("");
  const s = segmentTranscript(text, { maxChars: 180, overlapChars: 20, maxWindows: 6 });
  assert.equal(s.windows.length, 6);
  assert.equal(s.clipped, true);
  assert.ok(s.dropped > 0, "dropped count must be reported, not hidden");
  assert.ok(s.windows[0].start === 0, "the opening must survive");
  assert.ok(s.windows[s.windows.length - 1].end === text.length, "the ending must survive — that is where the ask lives");
  assert.deepEqual(s.windows.map((w) => w.index), [0, 1, 2, 3, 4, 5]);
});

test("a single wall of text with no punctuation still terminates and respects the cap", () => {
  const text = "a".repeat(5000);
  const s = segmentTranscript(text, { maxChars: 500, overlapChars: 50, maxWindows: 4 });
  assert.ok(s.windows.length > 0 && s.windows.length <= 4);
  assert.equal(s.strategy, "hard");
  assert.ok(s.windows.every((w) => w.text.length <= 700));
});

test("mid-sentence cuts are impossible when sentences are the only boundaries", () => {
  const text = Array.from({ length: 40 }, (_, i) => `Ye point number ${i} hai aur ismein thoda vivran diya gaya hai. `).join("");
  const s = segmentTranscript(text, { maxChars: 250, overlapChars: 60 });
  assert.equal(s.strategy, "sentences");
  for (const w of s.windows) {
    assert.ok(/[.।]\s*$/.test(w.text.trim()), `window ends mid-sentence: …${w.text.trim().slice(-30)}`);
  }
});

test("Hindi/Devanagari danda counts as a sentence end", () => {
  const text = Array.from({ length: 30 }, (_, i) => `यह वाक्य संख्या ${i} है और इसमें जानकारी दी गई है। `).join("");
  const s = segmentTranscript(text, { maxChars: 200, overlapChars: 40 });
  assert.ok(s.windows.length > 1);
  assert.equal(s.strategy, "sentences");
});
