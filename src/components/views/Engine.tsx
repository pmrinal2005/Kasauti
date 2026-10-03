"use client";
import { Fragment, useEffect, useState } from "react";
import { detectCapability, type Capability } from "@/lib/capability";
import { enableDeepCheck, inspectSequence, MODEL_BASE, runBenchmark, subscribeModel, type BenchResult, type InspectResult, type ModelState } from "@/lib/engine";
import { CLAIM_V1_FOLLOWUPS, CLAIM_V1_TRIAGE, DEFAULT_MANIFEST, renderOptions } from "@/core/laya-client-browser";
import { BarChart } from "../charts";
import { ICpu, IDownload, IBolt } from "../icons";

const yes = (b: boolean) => <span className={`chip ${b ? "ok" : "neutral"}`}>{b ? "yes" : "no"}</span>;

export function Engine() {
  const [cap, setCap] = useState<Capability | null>(null);
  const [model, setModel] = useState<ModelState | null>(null);
  const [bench, setBench] = useState<BenchResult | null>(null);
  const [benching, setBenching] = useState(false);
  const [cacheMB, setCacheMB] = useState<number | null>(null);
  const [probe, setProbe] = useState("Guaranteed 3% daily profit, join VIP group today");
  const [insp, setInsp] = useState<InspectResult | null>(null);
  const [inspBusy, setInspBusy] = useState(false);

  useEffect(() => { setCap(detectCapability()); return subscribeModel(setModel); }, []);
  useEffect(() => {
    navigator.storage?.estimate?.().then((e) => setCacheMB((e.usage ?? 0) / 1048576)).catch(() => {});
  }, [model?.status]);

  const doBench = async () => { setBenching(true); setBench(await runBenchmark()); setBenching(false); };
  const doInspect = async () => {
    setInspBusy(true);
    const r = await inspectSequence(probe);
    setInsp(r);
    setInspBusy(false);
    navigator.storage?.estimate?.().then((e) => setCacheMB((e.usage ?? 0) / 1048576)).catch(() => {});
  };
  const clearCache = async () => { await caches.delete("kasauti-model-v1"); setCacheMB(0); };

  return (
    <>
      <section className="page-head">
        <div><h1>On-Device Engine</h1><p>Laya runs as a quantized ONNX graph in a Web Worker — loaded from a public Hugging Face <b>model</b> repo, never a server you rent.</p></div>
        <span className="pill"><ICpu width={14} />{cap ? `${cap.profile} profile · ${cap.backend}` : "detecting…"}</span>
      </section>

      <section className="grid row-3">
        <article className="card">
          <div className="card-head"><div><h3>Capability gating</h3><p>Decides backend + question budget</p></div></div>
          {cap && (
            <dl className="kv">
              <dt>WebGPU (navigator.gpu)</dt><dd>{yes(cap.webgpu)}</dd>
              <dt>Cross-origin isolated</dt><dd>{yes(cap.crossOriginIsolated)}</dd>
              <dt>Logical cores</dt><dd>{cap.threads}</dd>
              <dt>Device memory</dt><dd>{cap.memoryGB ? `${cap.memoryGB} GB` : "n/a (not exposed)"}</dd>
              <dt>Save-Data / network</dt><dd>{cap.saveData ? "on" : "off"} · {cap.effectiveType ?? "n/a"}</dd>
              <dt>SpeechRecognition</dt><dd>{yes(cap.speechRecognition)}</dd>
              <dt>SpeechSynthesis</dt><dd>{yes(cap.speechSynthesis)}</dd>
              <dt>In-app browser</dt><dd>{cap.inAppBrowser ?? "none"}</dd>
              <dt>Chosen backend</dt><dd><b>{cap.backend}</b> ({cap.ortThreads} ORT thread{cap.ortThreads > 1 ? "s" : ""})</dd>
              <dt>Stage-B question budget</dt><dd><b>{cap.questionBudget}</b> noul follow-up{cap.questionBudget === 1 ? "" : "s"}</dd>
            </dl>
          )}
        </article>

        <article className="card">
          <div className="card-head"><div><h3>Model delivery</h3><p>Opt-in download · Cache API · circuit breaker</p></div></div>
          <div className="stack">
            <div className="note">
              {cap?.deferDownload ? "Save-Data / 2G detected — download deferred. Tier-0 checking stays fully active." : "Deep checking downloads the quantized checker once (tens–low hundreds of MB) and then works offline."}
            </div>
            <dl className="kv">
              <dt>Status</dt><dd><b>{model?.status}</b></dd>
              <dt>Detail</dt><dd style={{ fontSize: 11.5 }}>{model?.detail}</dd>
              <dt>Origin storage used</dt><dd>{cacheMB == null ? "n/a" : `${cacheMB.toFixed(1)} MB`}</dd>
              <dt>Repo base</dt><dd className="mono" style={{ wordBreak: "break-all", fontSize: 11 }}>{MODEL_BASE}</dd>
            </dl>
            {model?.status === "downloading" && <div className="meter"><span style={{ width: `${(model.progress * 100).toFixed(0)}%` }} /></div>}
            <div className="row">
              <button className="btn primary" disabled={!cap || cap.backend === "tier0-only" || model?.status === "ready"} onClick={() => cap && enableDeepCheck(cap.backend, cap.ortThreads)}><IDownload />{cap?.deferDownload ? "Download anyway" : "Enable deep checking"}</button>
              <button className="btn ghost" onClick={clearCache}>Clear model cache</button>
            </div>
          </div>
        </article>

        <article className="card">
          <div className="card-head"><div><h3>Week-0 device benchmark</h3><p>No published number exists — measure it.</p></div></div>
          <div className="stack">
            <button className="btn" onClick={doBench} disabled={benching}><IBolt />{benching ? "Measuring in Worker…" : "Run micro-benchmark"}</button>
            {bench ? (
              <>
                <dl className="kv">
                  <dt>Scalar FP32 GEMM</dt><dd><b>{bench.gflops.toFixed(2)}</b> GFLOP/s</dd>
                  <dt>Block median</dt><dd>{bench.medianMs.toFixed(1)} ms</dd>
                </dl>
                <BarChart height={170} data={[
                  { label: "1 question", value: Math.round(bench.projectedMs.q1), color: "#38e1c4" },
                  { label: "1 + 2 (low)", value: Math.round(bench.projectedMs.q3), color: "#7c8cff" },
                  { label: "1 + 3 (full)", value: Math.round(bench.projectedMs.q4), color: "#fb7185" },
                ]} />
                <div className="faint" style={{ fontSize: 11.5 }}>Rough projected ms per call for mmBERT-base (~320 tokens/question). An estimate for budgeting only — real ORT timings replace this once the export is published.</div>
              </>
            ) : <div className="muted" style={{ fontSize: 12.5 }}>Runs a 768-wide matrix multiply inside the inference Worker (main thread stays responsive).</div>}
          </div>
        </article>
      </section>

      <section className="grid row-2b">
        <article className="card">
          <div className="card-head"><div><h3>Question bank · claim_v1</h3><p>One triage choice, then ≤ budget targeted noul calls — never a flat batch</p></div></div>
          <div className="note mono" style={{ whiteSpace: "pre-wrap", fontSize: 11.5, color: "var(--text)" }}>
            {`[CLS] choice question: ${CLAIM_V1_TRIAGE.instructions} [SEP]\n` + renderOptions(CLAIM_V1_TRIAGE).map((o) => `  [MASK] ${o}`).join("\n") + "\n[SEP] <message> [SEP]"}
          </div>
          <div className="stack" style={{ marginTop: 10, gap: 6 }}>
            {Object.entries(CLAIM_V1_FOLLOWUPS).map(([k, qs]) => (
              <div key={k} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
                <span className="chip info" style={{ minWidth: 120, justifyContent: "center" }}>{k}</span>
                <div className="muted" style={{ fontSize: 12 }}>{Object.values(qs).map((q) => q.instructions).join(" · ") || "no follow-ups (Tier-0 only)"}</div>
              </div>
            ))}
          </div>
        </article>
        <article className="card">
          <div className="card-head"><div><h3>Calibration manifest</h3><p>Shipped as a JSON sidecar next to the weights</p></div></div>
          <dl className="kv">
            <dt>Model version</dt><dd>{DEFAULT_MANIFEST.modelVersion}</dd>
            <dt>max_len / head_max_len</dt><dd>{DEFAULT_MANIFEST.maxLen} / {DEFAULT_MANIFEST.headMaxLen}</dd>
            <dt>Temperature buckets</dt><dd>{Object.keys(DEFAULT_MANIFEST.temperatures).length || "none yet (T = 1)"}</dd>
            {Object.entries(DEFAULT_MANIFEST.abstain).map(([b, v]) => <Fragment key={b}><dt>Abstain below · {b}</dt><dd>{v}</dd></Fragment>)}
          </dl>
          <div className="note" style={{ marginTop: 12 }}>
            Design rules from Laya’s published limits: ≤20 options per choice (Banking77 collapse), option-order augmentation, <span className="mono">score</span> treated as weakest primitive, gate on <span className="mono">confidence</span> not <span className="mono">act_probability</span> (#185), and per-bucket temperature refit (ECE 0.314 → 0.106 on multilingual).
          </div>
        </article>
      </section>

      <section className="card" id="tokenizer-dry-run">
        <div className="card-head">
          <div><h3>Tokenizer dry-run · exact Laya input</h3><p>Fetches only the real <span className="mono">laya-multilingual</span> tokenizer (≈34 MB, cached) and builds the <span className="mono">claim_v1</span> triage sequence in the Worker — verifiable before any ONNX graph is published.</p></div>
        </div>
        <div className="row" style={{ flexWrap: "nowrap" }}>
          <label htmlFor="probe" className="sr-only">Message to tokenize</label>
          <input id="probe" className="input" value={probe} onChange={(e) => setProbe(e.target.value)} />
          <button className="btn" onClick={doInspect} disabled={inspBusy || !probe.trim()}><ICpu />{inspBusy ? "Tokenizing…" : "Build sequence"}</button>
        </div>
        {insp ? (
          <div className="grid row-2b" style={{ marginTop: 12 }}>
            <dl className="kv">
              <dt>Sequence length</dt><dd><b>{insp.ids}</b> / {DEFAULT_MANIFEST.maxLen} tokens</dd>
              <dt>[MASK] option markers</dt><dd className="mono">{insp.markers.join(", ")}</dd>
              <dt>CLS / SEP / MASK ids</dt><dd className="mono">{insp.special.cls} / {insp.special.sep} / {insp.special.mask}</dd>
              <dt>Build time</dt><dd>{insp.ms.toFixed(1)} ms</dd>
            </dl>
            <div className="note mono" style={{ fontSize: 11, wordBreak: "break-all", color: "var(--text)" }}>[{insp.head.join(", ")}, …]</div>
          </div>
        ) : <div className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>Byte-identical to Python <span className="mono">tokenizers</span> after the Metaspace parity patch (see <span className="mono">patchTokenizerJSON</span>).</div>}
      </section>
    </>
  );
}
