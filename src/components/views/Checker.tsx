"use client";
import { useEffect, useRef, useState } from "react";
import { fuseTier1, RUNG_META, type Evidence, type Rung, type Tier0Result } from "@/core/fusion";
import { CLAIM_V1_TRIAGE, planFollowups, type Answer } from "@/core/laya-client-browser";
import { checkTier0, getModelState, localVerdict, lookupShared, predictTier1, saveLocalVerdict, subscribeModel, syncShared, type ModelState } from "@/lib/engine";
import { addCheck, go, set, toast, useStore } from "@/lib/store";
import { detectCapability } from "@/lib/capability";
import { recognitionSupported, speak, startRecognizer, stopSpeaking, type RecMode, type RecognizerHandle } from "@/core/voice";
import { IAlert, ICheck, IMic, IShield, ISpark, ISpeaker } from "../icons";

const SAMPLES = [
  "Congratulations! Join our VIP stock tips group. Guaranteed 3% daily profit, zero risk. Limited seats, join today only. Pay ₹5,000 to vipfund@ybl",
  "This is CBI officer. Your Aadhaar is linked to a money laundering case. Stay on video call, do not tell anyone in your family, transfer ₹2 lakh for verification.",
  "Your profit of ₹4.8 lakh is ready. Pay 18% withdrawal tax to unlock your profit. Install AnyDesk so our team can help.",
  "SEBI warns investors: beware of unregistered tips groups promising guaranteed return. Report fraud to 1930.",
  "आपका KYC expired है, खाता बंद होगा। तुरंत इस लिंक पर क्लिक करें: http://kyc-update.xyz",
  "Research by INH000009001: quarterly results summary of the auto sector, no recommendations.",
];

interface Outcome {
  t0: Tier0Result;
  shared: { rung: string; hits: number } | null;
  local: { rung: string; ts: number } | null;
  evidence: Evidence[];
  score: number;
  t1: {
    triage?: Answer;
    followups: Record<string, Answer>;
    ms: number;
    modelVersion: string;
    backend: string;
    /** false = the export published no usable calibration, so its answers are never acted on */
    calibrated: boolean;
    /** non-null when the export was trained on different question wording than this runtime asks */
    banksMismatch: string | null;
    abstained: { count: number; of: number };
    tokens: { batch: number; seqLen: number; markers: number };
  } | null;
  finalRung: Rung;
}

export function Checker() {
  const draft = useStore((s) => s.draft);
  const lang = useStore((s) => s.lang);
  const [text, setText] = useState(draft);
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<Outcome | null>(null);
  const [why, setWhy] = useState<string>("");
  const [whyBusy, setWhyBusy] = useState(false);
  const [model, setModel] = useState<ModelState>(getModelState());
  const [rec, setRec] = useState<RecognizerHandle | null>(null);
  const [recMode, setRecMode] = useState<RecMode>("unknown");
  const [canMic, setCanMic] = useState(false);
  const autoRan = useRef(false);

  useEffect(() => subscribeModel(setModel), []);
  useEffect(() => setCanMic(recognitionSupported()), []);
  useEffect(() => {
    if (draft && draft !== text) setText(draft);
    if (draft && !autoRan.current) { autoRan.current = true; void run(draft); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  async function run(input = text, channel: "paste" | "voice" = "paste") {
    const s = input.trim();
    if (!s) return;
    setBusy(true); setWhy(""); stopSpeaking();
    try {
      // 1–3: normalise, fingerprint, Tier-0 (Worker) — renders immediately
      const t0 = await checkTier0(s);
      // 4: on-device verdict cache (<2 ms) — a previous fused verdict for this exact text
      const lv = localVerdict(t0.hash);
      setOut({ t0, shared: null, local: lv ? { rung: lv.rung, ts: lv.ts } : null, evidence: t0.evidence, score: t0.score, t1: null, finalRung: t0.rung });
      // 5–7: edge/shared cache lookup in parallel with on-device Laya (never blocks the Tier-0 render)
      const [shared, t1] = await Promise.all([lookupShared(t0.hash), runTier1(s, t0)]);
      // 8: transparent evidence fusion (pure, unit-tested in core/fusion)
      const fused = fuseTier1(t0, t1?.triage ?? null, t1?.followups ?? {});
      setOut({ t0, shared: shared ? { rung: shared.rung, hits: shared.hits } : null, local: lv ? { rung: lv.rung, ts: lv.ts } : null, evidence: fused.evidence, score: fused.score, t1, finalRung: fused.rung });
      const modelVersion = t1?.modelVersion ?? "tier0-only";
      addCheck({ id: t0.hash.slice(0, 12), ts: Date.now(), excerpt: s.slice(0, 140), hash: t0.hash, rung: fused.rung, score: fused.score, families: Object.keys(t0.families), lang: lang.slice(0, 2), channel, tier: shared ? "cache" : fused.tier, ms: Math.round(t0.elapsedMs + (t1?.ms ?? 0)), local: true });
      saveLocalVerdict(t0.hash, { rung: fused.rung, score: fused.score, tier: fused.tier, modelVersion, ts: Date.now() });
      // 9: non-blocking sync of hash + codes only (never raw text) so future visitors get a cache hit
      syncShared(t0, modelVersion, { rung: fused.rung, codes: fused.evidence.map((e) => e.code) });
    } catch (e) {
      toast(`Check failed: ${String(e)}`);
    } finally {
      setBusy(false);
      set({ draft: "" });
    }
  }

  async function runTier1(s: string, t0: Tier0Result) {
    if (getModelState().status !== "ready") return null;
    const cap = detectCapability();
    const tri = await predictTier1(s, { triage: CLAIM_V1_TRIAGE });
    if (!tri) return null;
    const triage = tri.answers.triage;
    const flagged = new Set<string>();
    if (t0.families.pay_to_withdraw) flagged.add("pay_to_withdraw");
    if (t0.families.guaranteed_return) flagged.add("guaranteed");
    if (t0.families.secrecy) flagged.add("secrecy");
    const fu = triage.abstained ? {} : planFollowups(triage.choice!, flagged, cap.questionBudget);
    const fr = Object.keys(fu).length ? await predictTier1(s, fu) : null;
    const answers = { triage, ...(fr?.answers ?? {}) };
    const values = Object.values(answers);
    return {
      triage,
      followups: fr?.answers ?? {},
      ms: tri.ms + (fr?.ms ?? 0),
      modelVersion: tri.modelVersion,
      backend: tri.backend,
      calibrated: tri.calibrated,
      banksMismatch: tri.banksMismatch,
      abstained: { count: values.filter((a) => a.abstained).length, of: values.length },
      tokens: fr?.tokens ?? tri.tokens,
    };
  }

  function toggleMic() {
    if (rec) { rec.stop(); setRec(null); return; }
    const h = startRecognizer({
      lang,
      onInterim: (t) => setText(t),
      onFinal: (t) => { setText(t); void run(t, "voice"); },
      onMode: setRecMode,
      onError: (e) => { toast(e === "not-allowed" ? "Microphone blocked — please type instead" : `Voice error: ${e}`); setRec(null); },
      onEnd: () => setRec(null),
    });
    if (!h) { toast("Voice input unavailable here — type instead"); setCanMic(false); return; }
    setRec(h);
  }

  async function explain() {
    if (!out) return;
    setWhyBusy(true); setWhy("");
    try {
      const r = await fetch("/api/explain", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, codes: out.evidence.map((e) => e.code), lang }) });
      const reader = r.body!.getReader();
      const dec = new TextDecoder();
      let acc = "";
      for (;;) { const { done, value } = await reader.read(); if (done) break; acc += dec.decode(value, { stream: true }); setWhy(acc); }
    } catch { setWhy("Explanation unavailable offline. The on-device evidence above still applies."); }
    setWhyBusy(false);
  }

  const m = out ? RUNG_META[out.finalRung] : null;
  return (
    <>
      <section className="page-head">
        <div><h1>Message Checker</h1><p>Paste a forward, SMS, or speak it. Tier-0 runs on this device in milliseconds; nothing is uploaded except an anonymous hash.</p></div>
        <span className="pill"><span className={`dot ${model.status === "ready" ? "" : "info"}`} />Laya: {model.status === "ready" ? `active (${model.backend})` : model.status === "unpublished" ? "awaiting ONNX export · T0 only" : model.status === "idle" ? "not enabled · T0 only" : model.status}</span>
      </section>

      <section className="grid row-2">
        <article className="card">
          <div className="card-head"><div><h3>Input</h3><p>Typed, pasted or spoken — all paths are equal.</p></div>
            {recMode !== "unknown" && rec && <span className="chip info">{recMode === "on-device" ? "processed by your phone" : "processed online by your browser"}</span>}
          </div>
          <label htmlFor="msg" className="sr-only">Message text</label>
          <textarea id="msg" className="input" rows={7} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the message you received…" onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") void run(); }} />
          <div className="row between" style={{ marginTop: 12 }}>
            <div className="row">
              <button className="btn primary" onClick={() => run()} disabled={busy || !text.trim()}><IShield />{busy ? "Checking…" : "Check message"}</button>
              {canMic ? (
                <button className={`btn ${rec ? "rec" : ""}`} onClick={toggleMic} aria-pressed={!!rec}><IMic />{rec ? "Stop" : "Speak"}</button>
              ) : (
                <span className="chip neutral" title="SpeechRecognition unavailable in this browser / in-app webview">Voice unavailable — type instead</span>
              )}
              <button className="btn ghost" onClick={() => { setText(""); setOut(null); setWhy(""); }}>Clear</button>
            </div>
            <span className="faint" style={{ fontSize: 11.5 }}>Ctrl + Enter</span>
          </div>
          <div style={{ marginTop: 16 }}>
            <div className="faint" style={{ fontSize: 11.5, marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>Try a sample</div>
            <div className="row" style={{ gap: 6 }}>
              {SAMPLES.map((s, i) => <button key={i} className="chip neutral" style={{ border: 0, cursor: "pointer" }} onClick={() => { setText(s); void run(s); }}>{["VIP tips", "Digital arrest", "Pay-to-withdraw", "SEBI warning", "KYC (Hindi)", "Registered RA"][i]}</button>)}
            </div>
          </div>
        </article>

        <article className="card" aria-live="polite">
          <div className="card-head"><div><h3>Evidence Ladder</h3><p>Uncertainty is shown, never hidden.</p></div></div>
          {!out ? <div className="empty">Results appear here instantly.</div> : (
            <div className="stack">
              <div className={`verdict-banner ${m!.tone}`}>
                {m!.tone === "ok" ? <ICheck width={26} /> : <IAlert width={26} />}
                <div><h4>{m!.label}</h4><div className="muted" style={{ fontSize: 12 }}>Score {out.score.toFixed(1)} · heuristic agreement {(out.t0.confidence * 100).toFixed(0)}% · {out.t0.elapsedMs.toFixed(1)} ms on-device</div></div>
              </div>
              <div className="ladder">{(["safe_pattern", "unclear", "caution", "likely_scam", "known_scam"] as Rung[]).map((r) => <div key={r} className={`rung ${RUNG_META[r].step <= m!.step ? "on " + m!.tone : ""}`} />)}</div>
              <div className="rung-labels"><span>Safe</span><span>Unclear</span><span>Caution</span><span>Likely</span><span>Known</span></div>
              {out.local && <div className="note">Checked on this device before ({new Date(out.local.ts).toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })}) — previous verdict: {RUNG_META[out.local.rung as Rung]?.label ?? out.local.rung}.</div>}
              {out.shared && <div className="note">Shared cache: this exact text was seen <b>{out.shared.hits}×</b> before (verdict: {out.shared.rung.replace("_", " ")}).</div>}
              {out.t1 && (
                <div className="note">
                  <div className="row between" style={{ alignItems: "baseline" }}>
                    <div>
                      <b>Laya Tier-1</b> ({out.t1.backend}, {out.t1.ms.toFixed(0)} ms): triage = <b>{out.t1.triage?.choice}</b> ({((out.t1.triage?.confidence ?? 0) * 100).toFixed(0)}% conf{out.t1.triage?.abstained ? ", abstained" : ""})
                      {Object.entries(out.t1.followups).map(([k, a]) => <div key={k} className="mono">{k}: p={(a.noul ?? 0).toFixed(2)}{a.abstained ? " (abstain)" : ""}</div>)}
                    </div>
                    <span className={`chip ${out.t1.calibrated ? "ok" : "warn"}`}>{out.t1.calibrated ? "calibrated" : "uncalibrated · report-only"}</span>
                  </div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                    {out.t1.abstained.count} of {out.t1.abstained.of} answers sat below the abstention threshold
                    {out.t1.abstained.count === out.t1.abstained.of ? " — so nothing above changed the verdict; the ladder is still the deterministic one" : ""}.
                    {" "}{out.t1.tokens.seqLen} tokens, {out.t1.tokens.markers} option markers, {out.t1.modelVersion}
                  </div>
                  {out.t1.banksMismatch && (
                    <div className="muted" style={{ fontSize: 12, marginTop: 4, color: "var(--warn)" }}>
                      Question-bank drift: {out.t1.banksMismatch} The numbers above are shown for transparency but carry no weight in the verdict.
                    </div>
                  )}
                </div>
              )}
              {!out.t1 && (model.status === "deferred" || model.status === "unpublished" || model.status === "idle") && (
                <div className="note">
                  <b>Deep checking is off</b> — {model.status === "deferred" ? "Data Saver on your connection asked to be spared the download" : model.status === "unpublished" ? "no calibrated ONNX export is published yet" : "the model has not been enabled on this device"}.
                  {" "}The verdict above is deterministic (Tier-0) only. That is the tier we trust for acting: no model was consulted, so none can be wrong.
                </div>
              )}
              {(m!.step >= 2) && (
                <div className="row">
                  <a className="btn" href="tel:1930"><IAlert />Call 1930</a>
                  <a className="btn" href="https://www.sebi.gov.in/intermediaries.html" target="_blank" rel="noreferrer">SEBI check</a>
                </div>
              )}
            </div>
          )}
        </article>
      </section>

      {out && (
        <section className="grid row-2b">
          <article className="card">
            <div className="card-head"><div><h3>How we know</h3><p>Every signal, its weight and its source.</p></div><span className="chip neutral">{out.evidence.length} signals</span></div>
            <div className="stack" style={{ gap: 8 }}>
              {out.evidence.length === 0 && <div className="empty">No deterministic scam signals. Absence of evidence is not proof of safety.</div>}
              {out.evidence.map((e, i) => (
                <div className="evidence" key={i}>
                  <span className="w" style={{ color: e.weight > 0 ? "var(--bad)" : "var(--ok)" }}>{e.weight > 0 ? "+" : ""}{e.weight.toFixed(1)}</span>
                  <div style={{ minWidth: 0 }}><b style={{ fontSize: 13 }}>{e.label}</b> <span className="chip neutral" style={{ marginLeft: 4 }}>{e.source}</span><div className="muted" style={{ fontSize: 12.5 }}>{e.detail}</div></div>
                </div>
              ))}
            </div>
          </article>
          <article className="card">
            <div className="card-head"><div><h3>Extracted entities &amp; fingerprint</h3><p>Normalised text → SHA-256 + SimHash</p></div></div>
            <dl className="kv">
              <dt>SHA-256</dt><dd className="mono">{out.t0.hash.slice(0, 24)}…</dd>
              <dt>SimHash-64</dt><dd className="mono">{out.t0.simhash}</dd>
              <dt>UPI IDs</dt><dd>{out.t0.entities.upi.map((u) => u.id).join(", ") || "—"}</dd>
              <dt>Phones</dt><dd>{out.t0.entities.phones.map((p) => `${p.raw} (${p.kind})`).join(", ") || "—"}</dd>
              <dt>Links</dt><dd style={{ wordBreak: "break-all" }}>{out.t0.entities.urls.join(", ") || "—"}</dd>
              <dt>Amounts</dt><dd>{out.t0.entities.money.map((x) => `₹${x.rupees.toLocaleString("en-IN")}`).join(", ") || "—"}</dd>
              <dt>Return claims</dt><dd>{out.t0.entities.returns.map((r) => `${r.percent}%/${r.period}`).join(", ") || "—"}</dd>
              <dt>SEBI reg. no.</dt><dd>{out.t0.registry.map((r) => `${r.reg} ${r.entry ? "✓ " + r.entry.status : "✗ not found"}`).join(", ") || "—"}</dd>
            </dl>
            <div className="row" style={{ marginTop: 14 }}>
              <button className="btn primary" onClick={explain} disabled={whyBusy}><ISpark />{whyBusy ? "Explaining…" : "Why? (text-only)"}</button>
              {why && <button className="btn" onClick={() => { const r = speak(why, lang); if (r === "no-voice") toast("No device voice for this language — showing text"); }}><ISpeaker />Read aloud</button>}
              <button className="btn ghost" onClick={() => go("voice")}>Learn the concept</button>
            </div>
            {why && <div className="note" style={{ marginTop: 12, whiteSpace: "pre-wrap", color: "var(--text)" }}>{why}</div>}
          </article>
        </section>
      )}
    </>
  );
}
