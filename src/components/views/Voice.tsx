"use client";
import { useEffect, useState } from "react";
import { CONCEPTS, conceptForBankKey, conceptForFamily, routeConcept, type Concept } from "@/lib/concepts";
import { BANKS, bankFollowups, type QuestionSet } from "@/lib/banks";
import { getModelState, predictTier1, subscribeModel, type ModelState } from "@/lib/engine";
import { recognitionSupported, speak, startRecognizer, stopSpeaking, voicesFor, type RecMode, type RecognizerHandle } from "@/core/voice";
import { set, toast, useStore } from "@/lib/store";
import { IMic, ISpeaker, IBook, ICpu } from "../icons";

/** What routed the last question, and how sure it was. Shown, never hidden. */
interface RouteTrace {
  source: "laya" | "tier0-lexical" | "none";
  family?: string;
  concept?: string;
  confidence?: number;
  ms?: number;
  abstained?: boolean;
  calibrated?: boolean;
  backend?: string;
  /** the Laya answer disagreed with the lexical read, so the leaf shows the disambiguation cards */
  refined: boolean;
}

export function Voice() {
  const lang = useStore((s) => s.lang);
  const draft = useStore((s) => s.draft);
  const [canMic, setCanMic] = useState(false);
  const [voiceCount, setVoiceCount] = useState(0);
  const [rec, setRec] = useState<RecognizerHandle | null>(null);
  const [mode, setMode] = useState<RecMode>("unknown");
  const [transcript, setTranscript] = useState("");
  const [typed, setTyped] = useState("");
  const [active, setActive] = useState<Concept | null>(null);
  const [alts, setAlts] = useState<Concept[]>([]);
  const [speaking, setSpeaking] = useState<string | null>(null);
  const [model, setModel] = useState<ModelState>(getModelState());
  const [trace, setTrace] = useState<RouteTrace>({ source: "none", refined: false });

  useEffect(() => subscribeModel(setModel), []);

  useEffect(() => {
    setCanMic(recognitionSupported());
    const upd = () => setVoiceCount(voicesFor(lang).length);
    upd();
    if (typeof speechSynthesis !== "undefined") speechSynthesis.onvoiceschanged = upd;
    return () => { if (typeof speechSynthesis !== "undefined") speechSynthesis.onvoiceschanged = null; stopSpeaking(); };
  }, [lang]);

  useEffect(() => {
    if (!draft) return;
    const c = CONCEPTS.find((x) => x.id === draft);
    if (c) { setActive(c); setAlts([]); }
    set({ draft: "" });
  }, [draft]);

  const textFor = (c: Concept) => (lang === "hi-IN" || lang === "mr-IN" ? c.text["hi-IN"] : c.text["en-IN"]);
  const sayLang = lang === "mr-IN" ? "hi-IN" : lang === "ta-IN" ? "en-IN" : lang;

  /**
   * Two-tier router. Tier-0 (lexical, instant, deterministic) picks a concept and speaks it.
   * Tier-1 (Laya `intent_concepts_v1`, only when a calibrated export is loaded) refines that:
   * it names the family and, when it is sure enough, the leaf concept. Abstentions are respected —
   * the model never invents a topic, and an uncalibrated export never speaks for the app.
   */
  function route(t: string) {
    const r = routeConcept(t);
    setActive(r.concept);
    setAlts(r.alternatives);
    setTrace({ source: r.concept ? "tier0-lexical" : "none", concept: r.concept?.id, refined: false });
    if (r.concept) play(r.concept);
    void refine(t, r);
  }

  async function refine(t: string, r: { concept: Concept | null; alternatives: Concept[] }) {
    if (getModelState().status !== "ready" || !t.trim()) return;
    const first = await predictTier1(t, { triage: BANKS.intent_concepts_v1.triage });
    const triage = first?.answers.triage;
    if (!first || !triage) return;
    const base: RouteTrace = {
      source: "laya",
      family: triage.choice,
      confidence: triage.confidence,
      ms: first.ms,
      abstained: triage.abstained,
      calibrated: first.calibrated,
      backend: first.backend,
      refined: false,
    };
    if (!first.calibrated) { setTrace({ ...base, source: "tier0-lexical", abstained: true }); return; }
    // exactly one follow-up turn, and only for a non-abstained triage with leaves in the bank
    const plan: QuestionSet = triage.abstained ? {} : bankFollowups("intent_concepts_v1", triage.choice!, new Set(), 1);
    const second = Object.keys(plan).length ? await predictTier1(t, plan) : null;
    let leaf: Concept | null = null;
    let leafKey: string | undefined;
    let leafConf = 0;
    for (const [k, a] of Object.entries(second?.answers ?? {})) {
      const p = a.noul ?? 0;
      if (!a.abstained && p > leafConf) { leafConf = p; leafKey = k; }
    }
    leaf = conceptForBankKey(leafKey);
    const familyConcept = conceptForFamily(triage.choice);
    const win = leaf ?? familyConcept;
    if (!win) { setTrace(base); return; }
    const changed = win.id !== r.concept?.id;
    setTrace({
      ...base,
      concept: win.id,
      ms: first.ms + (second?.ms ?? 0),
      refined: changed,
    });
    if (changed) {
      setActive(win);
      // the routed leaf replaces the lexical guess, but the alternatives stay reachable
      setAlts([r.concept, ...r.alternatives].filter((c): c is Concept => !!c && c.id !== win.id).slice(0, 3));
      if (leafKey) toast(`Laya routed this to “${win.title}” (${(leafConf * 100).toFixed(0)}%)`);
    }
  }

  function play(c: Concept) {
    const res = speak(textFor(c), sayLang);
    if (res === "no-voice") { toast("No device voice for this language — showing text only"); setSpeaking(null); return; }
    setSpeaking(c.id);
    const iv = setInterval(() => { if (!speechSynthesis.speaking) { setSpeaking(null); clearInterval(iv); } }, 400);
  }

  function hold() {
    if (rec) { rec.stop(); return; }
    const h = startRecognizer({ lang, onInterim: setTranscript, onFinal: (t) => { setTranscript(t); route(t); }, onMode: setMode, onError: (e) => toast(`Voice: ${e}`), onEnd: () => setRec(null) });
    if (!h) { setCanMic(false); return; }
    setRec(h);
  }

  return (
    <>
      <section className="page-head">
        <div><h1>Voice-First Explainer</h1><p>Ask “SIP kya hota hai?” — speech is converted to text by your own browser; Kasauti never receives audio.</p></div>
        <div className="row">
          <span className="pill"><span className={`dot ${canMic ? "" : "warn"}`} />Mic {canMic ? "available" : "unavailable"}</span>
          <span className="pill"><span className={`dot ${voiceCount ? "" : "warn"}`} />{voiceCount} device voice{voiceCount === 1 ? "" : "s"} for {sayLang}</span>
        </div>
      </section>

      <section className="grid row-2">
        <article className="card">
          <div className="card-head"><div><h3>Ask a question</h3><p>Closed concept set → robust to transcription noise. Laya refines the routing when a calibrated export is loaded.</p></div>
            <div className="row">
              <span className={`pill`}><span className={`dot ${model.status === "ready" && model.calibrated ? "" : "info"}`} /><ICpu width={12} height={12} />{model.status === "ready" ? (model.calibrated ? `Laya router · ${model.backend}` : "Laya uncalibrated · lexical only") : "Tier-0 lexical router"}</span>
              {rec && <span className="chip info">{mode === "on-device" ? "processed by your phone" : "processed online by your browser"}</span>}
            </div>
          </div>
          <div className="stack" style={{ alignItems: "center", padding: "10px 0 6px" }}>
            {canMic ? (
              <button className={`btn ${rec ? "rec" : "primary"}`} style={{ height: 64, padding: "0 28px", fontSize: 15, borderRadius: 18 }} onClick={hold} aria-pressed={!!rec}><IMic width={22} height={22} />{rec ? "Listening… tap to stop" : "Tap to speak"}</button>
            ) : (
              <div className="note" style={{ textAlign: "center" }}>Voice input isn’t available in this browser (e.g. Firefox or an in-app webview). Typing works exactly the same.</div>
            )}
            {transcript && <div className="muted" style={{ fontSize: 13 }}>“{transcript}”</div>}
          </div>
          <form className="row" style={{ marginTop: 10, flexWrap: "nowrap" }} onSubmit={(e) => { e.preventDefault(); setTranscript(typed); route(typed); }}>
            <label htmlFor="typed-q" className="sr-only">Type your question</label>
            <input id="typed-q" className="input" placeholder="Type instead — e.g. what is NAV?" value={typed} onChange={(e) => setTyped(e.target.value)} />
            <button className="btn" type="submit">Ask</button>
          </form>
          {active ? (
            <div className="note" style={{ marginTop: 14, color: "var(--text)" }}>
              <div className="row between"><b>{active.title}</b><button className="btn" onClick={() => (speaking ? (stopSpeaking(), setSpeaking(null)) : play(active))}><ISpeaker />{speaking ? "Stop" : "Play"}</button></div>
              <p style={{ margin: "8px 0 0", lineHeight: 1.65 }}>{textFor(active)}</p>
              <div className="muted" style={{ fontSize: 11.5, marginTop: 8 }}>
                Routed by {trace.source === "laya"
                  ? <>Laya <span className="mono">intent_concepts_v1</span> — family <b>{trace.family}</b>{trace.confidence !== undefined ? <> ({(trace.confidence * 100).toFixed(0)}% conf{trace.abstained ? ", abstained" : ""})</> : null}, {trace.ms?.toFixed(0)} ms on-device{trace.refined ? ", corrected the lexical guess" : ""}</>
                  : <>the Tier-0 lexical router (<span className="mono">concepts.ts</span> keywords){trace.abstained ? " — Laya abstained or is uncalibrated, so its answer was not used" : ""}</>}
                .
              </div>
            </div>
          ) : transcript ? (
            <div style={{ marginTop: 14 }}>
              <div className="muted" style={{ fontSize: 12.5, marginBottom: 6 }}>Not sure what you meant — did you mean:</div>
              <div className="row">{alts.map((c) => <button key={c.id} className="btn" onClick={() => { setActive(c); play(c); }}>{c.title}</button>)}</div>
            </div>
          ) : null}
        </article>

        <article className="card">
          <div className="card-head"><div><h3>Synthesizer fallback chain</h3><p>Never leaves a non-reading user with silent text.</p></div></div>
          <div className="stack" style={{ gap: 8 }}>
            {[
              ["1", "On-device voice (exact language)", voiceCount > 0, "speechSynthesis.getVoices() filtered by BCP-47 — the only step that needs no network"],
              ["2", "Pre-rendered IndicF5 clip", false, "~300 static concepts · /packs/[lang]/audio/*.opus (pending render)"],
              ["3", "Text display", true, "Always available"],
            ].map(([n, t, ok, d]) => (
              <div className="tier" key={n as string}>
                <div className="tier-n" style={{ background: "var(--glass-2)" }}>{n as string}</div>
                <div style={{ flex: 1 }}><div className="row between"><b style={{ fontSize: 13 }}>{t as string}</b><span className={`chip ${ok ? "ok" : "neutral"}`}>{ok ? "ready" : "unavailable"}</span></div><div className="muted" style={{ fontSize: 12 }}>{d as string}</div></div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="card">
        <div className="card-head"><div><h3>Concept library</h3><p>Human-written analogies. Never names a fund, stock or broker.</p></div><IBook width={18} className="muted" /></div>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: 10 }}>
          {CONCEPTS.map((c) => (
            <button key={c.id} className="tier" style={{ cursor: "pointer", textAlign: "left", borderColor: active?.id === c.id ? "var(--accent)" : undefined }} onClick={() => { setActive(c); setTranscript(""); }}>
              <div style={{ minWidth: 0 }}>
                <span className={`chip ${c.family === "red_flags" ? "bad" : c.family === "safety" ? "warn" : "info"}`}>{c.family.replace("_", " ")}</span>
                <div style={{ fontWeight: 600, marginTop: 6, fontSize: 13 }}>{c.title}</div>
                <div className="muted" style={{ fontSize: 12, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{c.text["en-IN"]}</div>
              </div>
            </button>
          ))}
        </div>
      </section>
    </>
  );
}
