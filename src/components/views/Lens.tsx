"use client";
/**
 * Content Lens — the SEBI promotion/education line.
 *
 * The question this answers is not "is this a scam?" but "is this content *selling* something while
 * sounding educational?" That is a judgement a keyword list cannot make and a generative model must
 * not be trusted to invent: Laya answers it as typed decisions with calibrated probabilities
 * (`lens_v1`: one triage `choice`, then the follow-ups the triage answer makes worth asking).
 *
 * Two honesty rules are visible in this UI on purpose:
 *   * an answer below the fitted abstention threshold is shown as *unsettled*, not as a verdict;
 *   * when no calibrated export is published, the page says so and falls back to the Tier-0
 *     lexical read instead of dressing up an uncalibrated score as an answer.
 */
import { useEffect, useState } from "react";
import { BANKS, bankFollowups, type QuestionSet } from "@/lib/banks";
import { checkTier0, getModelState, predictTier1, subscribeModel, type ModelState } from "@/lib/engine";
import { addCheck, go, useStore } from "@/lib/store";
import { segmentTranscript, type Segmentation } from "@/lib/transcript";
import type { Answer } from "@/core/laya-client-browser";
import type { Tier0Result } from "@/core/fusion";
import { IAlert, ISpark } from "../icons";

const SAMPLES = [
  "Nifty ka target 26000 hai, ye stock 3 mahine me double karega. Paid VIP group join karo, link bio me.",
  "SEBI-registered research analyst explains how expense ratio affects long-term returns. No stock recommendations. Disclosure: INH000009001.",
  "Loss recovery me ho? Meri team ne 40 clients ka loss recover kiya, screenshots attached. Fees advance me.",
  "Mutual fund sahi hai ya FD? Dono ke risk aur return ka farak samjha raha hoon, koi product nahi bech raha.",
];

/** One window's reading, kept next to the offsets it came from so the UI can quote the source. */
interface WindowRead {
  index: number;
  start: number;
  end: number;
  units: number;
  chars: number;
  triage: Answer | null;
  follows: Record<string, Answer>;
}

interface Outcome {
  /** present when the input was windowed (a transcript, not a single post) */
  segmentation: Segmentation | null;
  perWindow: WindowRead[];
  /** mean calibrated probability per triage option across the windows that answered */
  aggregate: Record<string, number>;
  /** the window that leaned hardest towards promotion, if any */
  worst: WindowRead | null;
  t0: Tier0Result;
  triage: Answer | null;
  followups: Record<string, Answer>;
  ms: number;
  modelVersion: string;
  backend: string;
  calibrated: boolean;
  /** how many answers sat below the fitted threshold, out of how many that were asked */
  abstained: number;
  answersOf: number;
}

const LINE_COPY: Record<string, { label: string; tone: string }> = {
  teaches_concept: { label: "Education", tone: "ok" },
  reports_facts: { label: "Reporting", tone: "ok" },
  pushes_product: { label: "Promotion — asks you to buy/join/open", tone: "bad" },
  pushes_security: { label: "Promotion — names a security to buy or sell", tone: "bad" },
  fear_selling: { label: "Promotion — pressure and urgency", tone: "bad" },
  testimonial_lure: { label: "Promotion — profit proof as the hook", tone: "warn" },
  other: { label: "Unclear", tone: "neutral" },
};

export function Lens() {
  const lang = useStore((s) => s.lang);
  const [text, setText] = useState("");
  const [mode, setMode] = useState<"single" | "transcript">("single");
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<Outcome | null>(null);
  const [model, setModel] = useState<ModelState>(getModelState());

  useEffect(() => subscribeModel(setModel), []);

  /** One Laya reading of one state. Adaptive: triage, then only the follow-ups it earns. */
  async function readOne(state: string) {
    const usable = getModelState().status === "ready";
    let triage: Answer | null = null;
    let follows: Record<string, Answer> = {};
    let ms = 0;
    let modelVersion = "tier0-only";
    let backend = "none";
    if (usable) {
      const first = await predictTier1(state, { triage: BANKS.lens_v1.triage });
      triage = first?.answers.triage ?? null;
      ms = first?.ms ?? 0;
      modelVersion = first?.modelVersion ?? modelVersion;
      backend = first?.backend ?? backend;
      if (triage && !triage.abstained && triage.choice) {
        const plan: QuestionSet = bankFollowups("lens_v1", triage.choice, new Set(), 2);
        if (Object.keys(plan).length) {
          const second = await predictTier1(state, plan);
          follows = second?.answers ?? {};
          ms += second?.ms ?? 0;
        }
      }
    }
    return { triage, follows, ms, modelVersion, backend };
  }

  async function analyse(input = text) {
    const s = input.trim();
    if (!s) return;
    setBusy(true);
    try {
      const t0 = await checkTier0(s);
      const usable = getModelState().status === "ready";

      // Windowing only when the input is long enough to need it. The segmentation is a pure
      // function (src/lib/transcript.ts); a short post stays a single state, exactly as before.
      const seg = mode === "transcript" ? segmentTranscript(s) : segmentTranscript(s, { maxChars: 1e9 });
      const perWindow: WindowRead[] = [];
      let triage: Answer | null = null;
      let followups: Record<string, Answer> = {};
      let ms = 0;
      let modelVersion = "tier0-only";
      let backend = "none";
      let abstained = 0;
      let answersOf = 0;

      for (const w of seg.windows) {
        const r = await readOne(w.text);
        perWindow.push({ index: w.index, start: w.start, end: w.end, units: w.units, chars: w.text.length,
                         triage: r.triage, follows: r.follows });
        ms += r.ms;
        answersOf += 1 + Object.keys(r.follows).length;
        modelVersion = r.modelVersion !== "tier0-only" ? r.modelVersion : modelVersion;
        backend = r.backend !== "none" ? r.backend : backend;
      }
      const agg: Record<string, number> = {};
      let answered = 0;
      let worst: WindowRead | null = null;
      let worstScore = -1;
      for (const w of perWindow) {
        if (w.triage?.abstained) abstained += 1;
        abstained += Object.values(w.follows).filter((a) => a.abstained).length;
        if (!w.triage || w.triage.abstained) continue;
        answered += 1;
        for (const [k, p] of Object.entries(w.triage.probs)) agg[k] = (agg[k] ?? 0) + p;
        const promo = Object.entries(w.triage.probs)
          .filter(([k]) => LINE_COPY[k]?.tone === "bad")
          .reduce((m, [, p]) => Math.max(m, p), 0);
        if (promo > worstScore) { worstScore = promo; worst = w; }
      }
      for (const k of Object.keys(agg)) agg[k] = agg[k] / answered;
      // The headline reading is the aggregate: a 40-minute transcript is one piece of content.
      // A single window is still the unit that answered it, so the first window that answered wins.
      const headline = perWindow.find((w) => w.triage && !w.triage.abstained) ?? perWindow[0] ?? null;
      triage = headline?.triage ?? null;
      followups = headline?.follows ?? {};
      setOut({
        t0, triage, followups, ms, modelVersion, backend,
        calibrated: getModelState().calibrated, abstained, answersOf,
        segmentation: seg.windows.length > 1 || seg.clipped ? seg : null,
        perWindow, aggregate: agg, worst,
      });
      const key = triage && !triage.abstained ? triage.choice : undefined;
      addCheck({
        id: t0.hash.slice(0, 12), ts: Date.now(), excerpt: s.slice(0, 140), hash: t0.hash,
        rung: key && LINE_COPY[key]?.tone !== "bad" ? "safe_pattern" : t0.rung,
        score: triage?.confidence ?? t0.score, families: Object.keys(t0.families),
        lang: lang.slice(0, 2), channel: "paste", tier: usable ? "T0+T1" : "T0",
        ms: Math.round(t0.elapsedMs + ms), local: true,
      });
    } finally {
      setBusy(false);
    }
  }

  const triage = out?.triage ?? null;
  const line = triage && !triage.abstained && triage.choice ? LINE_COPY[triage.choice] : undefined;

  return (
    <>
      <section className="page-head">
        <div>
          <h1>Content Lens</h1>
          <p>
            Where does this content sit on the line between <b>education</b> and <b>promotion</b>? One triage
            question plus the follow-ups it makes worth asking — answered on your device by Laya, never a keyword guess.
          </p>
        </div>
        <span className="pill"><ISpark width={14} />lens_v1 · {model.status === "ready" ? model.backend : "Tier-0 only"}</span>
      </section>

      {!out?.calibrated && (
        <div className="note warn">
          <IAlert width={15} />{" "}
          {model.status === "ready"
            ? "This export is uncalibrated (or report-only): every answer below is shown as unsettled, on purpose."
            : "No calibrated export is published yet — this page reports the lexical (Tier-0) read only, and says so."}
        </div>
      )}

      <div className="card">
        <div className="row between">
          <label className="lbl" htmlFor="lens-in">Post, script, reel caption, article — or a whole transcript</label>
          <div className="row" style={{ gap: 6 }}>
            <button className={`btn ${mode === "single" ? "primary" : "ghost"}`} onClick={() => setMode("single")}>Single post</button>
            <button className={`btn ${mode === "transcript" ? "primary" : "ghost"}`} onClick={() => setMode("transcript")}>Transcript / call notes</button>
          </div>
        </div>
        <textarea id="lens-in" rows={mode === "transcript" ? 10 : 6} value={text} onChange={(e) => setText(e.target.value)}
          placeholder={mode === "transcript"
            ? "Paste the transcript. Speaker turns, paragraphs or sentences are used as cut points; overlapping windows are read separately and the readings are averaged."
            : SAMPLES[0]} />
        <div className="row">
          <button className="btn primary" disabled={busy || !text.trim()} onClick={() => void analyse()}>
            {busy ? "Reading…" : mode === "transcript" && text.length > 900 ? "Read it window by window" : "Place it on the line"}
          </button>
          {SAMPLES.map((s, i) => (
            <button key={i} className="btn ghost" onClick={() => { setText(s); void analyse(s); }}>
              sample {i + 1}
            </button>
          ))}
          {text.length > 900 && mode === "transcript" && (
            <span className="hint">{[...text].length} characters — cut into overlapping windows so nothing is dropped silently</span>
          )}
        </div>
      </div>

      {out && (
        <>
          <div className="card">
            <h2>Reading</h2>
            <div className="kv">
              <div>
                <span className="lbl">Placement</span>
                <b className={line ? `tag ${line.tone}` : "tag neutral"}>
                  {line ? line.label : "Unsettled — not enough signal to place it"}
                </b>
              </div>
              <div>
                <span className="lbl">Confidence (calibrated)</span>
                <b>{triage ? `${(triage.confidence * 100).toFixed(0)}%` : "—"}</b>
                <span className="hint">
                  {triage
                    ? `entropy over ${Object.keys(triage.probs).length} options · bucket ${triage.bucket} · T=${triage.temperature.toFixed(2)}`
                    : "the triage question needs the model"}
                </span>
              </div>
              <div>
                <span className="lbl">Asked</span>
                <b>{out.answersOf}</b>
                <span className="hint">
                  {out.ms.toFixed(0)} ms on {out.backend} · {out.modelVersion}
                  {out.segmentation ? ` · ${out.perWindow.length} windows${out.segmentation.clipped ? `, ${out.segmentation.dropped} dropped` : ""}` : ""}
                </span>
              </div>
            </div>

            {out.segmentation && (
              <>
                <h3 className="sub">How it was cut</h3>
                <p className="hint">
                  {out.perWindow.length} window{out.perWindow.length === 1 ? "" : "s"} of ≤ 900 characters, split on{" "}
                  <b>{out.segmentation.strategy === "turns" ? "speaker turns" : out.segmentation.strategy === "paragraphs" ? "paragraph breaks" : out.segmentation.strategy === "sentences" ? "sentence ends" : "character count"}</b>
                  , overlapping where the boundaries allow it.{" "}
                  {out.segmentation.clipped
                    ? `${out.segmentation.dropped} middle window${out.segmentation.dropped === 1 ? "" : "s"} were dropped to stay inside the on-device budget — the opening and the ending were kept, because that is where a script builds up and then asks for money.`
                    : "Nothing was dropped."}
                </p>
                <div className="row" style={{ gap: 4, alignItems: "stretch", flexWrap: "wrap" }}>
                  {out.perWindow.map((w) => {
                    const key = w.triage && !w.triage.abstained ? w.triage.choice : undefined;
                    const tone = key ? LINE_COPY[key]?.tone ?? "neutral" : "neutral";
                    const p = key ? Math.max(0, ...Object.entries(w.triage!.probs).filter(([k]) => LINE_COPY[k]?.tone === "bad").map(([, v]) => v)) : 0;
                    return (
                      <div key={w.index} className="note" style={{ flex: "1 1 120px", minWidth: 110, borderTop: `3px solid var(--${tone === "bad" ? "bad" : tone === "warn" ? "warn" : tone === "ok" ? "ok" : "line"})` }}>
                        <div className="mono" style={{ fontSize: 11 }}>window {w.index + 1}</div>
                        <div className="muted" style={{ fontSize: 11 }}>
                          {w.chars} chars · {w.units} {w.units === 1 ? "part" : "parts"}
                          {w.start > 0 || w.end < out.segmentation!.chars ? ` · [${w.start}–${w.end}]` : ""}
                        </div>
                        <b style={{ fontSize: 12 }}>{key ? LINE_COPY[key]?.label ?? key : "unsettled"}</b>
                        {key && <div className="muted" style={{ fontSize: 11 }}>{key} · promotion {(p * 100).toFixed(0)}%</div>}
                      </div>
                    );
                  })}
                </div>
                {out.worst && out.worst.triage && (
                  <p className="hint">
                    Strongest promotion signal sits in window {out.worst.index + 1} (characters {out.worst.start}–{out.worst.end}):{" "}
                    “{out.worst.triage.choice}”. Averaging can hide a single hard sell — this is the window to read yourself.
                  </p>
                )}
              </>
            )}

            {triage && (
              <>
                <h3 className="sub">Every option, with its probability{out.segmentation ? " (averaged across the windows that answered)" : ""}</h3>
                {Object.entries(triage.probs)
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, p]) => (
                    <div key={k} className="bar">
                      <span className="mono">{k}</span>
                      <span className="track"><i style={{ width: `${Math.max(1, p * 100)}%` }} /></span>
                      <span className="num">{(p * 100).toFixed(0)}%</span>
                    </div>
                  ))}
              </>
            )}
          </div>

          <div className="card">
            <h2>What the triage answer made worth asking</h2>
            {Object.keys(out.followups).length === 0 ? (
              <p className="hint">
                Nothing: {triage?.abstained ? "the triage answer was unsettled, so asking further would only add noise"
                  : `${triage?.choice ?? "this answer"} needs no follow-up in this bank`}.
              </p>
            ) : (
              <ul className="list">
                {Object.entries(out.followups).map(([qid, a]) => (
                  <li key={qid}>
                    <span className="mono">{qid}</span> — <b>{a.noul !== undefined ? (a.noul > 0.5 ? "yes" : "no") : "—"}</b>{" "}
                    <span className="hint">p(yes)={((a.noul ?? 0) * 100).toFixed(0)}% · {a.abstained ? "unsettled" : "confident"}</span>
                  </li>
                ))}
              </ul>
            )}
            {out.abstained > 0 && (
              <p className="hint">
                {out.abstained} of {out.answersOf} answers were below the fitted threshold — they are reported as
                unsettled rather than being turned into a claim.
                {out.segmentation && out.abstained === out.answersOf
                  ? " Every window was unsettled, so this page is not placing the content on the line: that is the honest answer, not a failure."
                  : ""}
              </p>
            )}
          </div>

          <div className="card">
            <h2>On-device lexical read (Tier-0)</h2>
            <p className="hint">
              Deterministic, ~1 ms, runs before the model and never uses the network. Families:{" "}
              {Object.keys(out.t0.families).length ? Object.keys(out.t0.families).join(", ") : "none"}.
            </p>
            {out.t0.evidence.length > 0 && (
              <ul className="list">
                {out.t0.evidence.map((e, i) => (
                  <li key={i}><span className="mono">{e.code}</span> — {e.detail}</li>
                ))}
              </ul>
            )}
            <div className="row">
              <button className="btn ghost" onClick={() => go("checker")}>Check a message instead</button>
            </div>
          </div>
        </>
      )}
    </>
  );
}
