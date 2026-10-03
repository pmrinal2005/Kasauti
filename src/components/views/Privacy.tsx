"use client";
import { useEffect, useState } from "react";
import { clearChecks, useStore } from "@/lib/store";
import { ILock, ICheck } from "../icons";

const FLOWS = [
  { what: "Message text (Tier-0)", where: "This device (Web Worker)", leaves: "Never", tone: "ok" },
  { what: "Laya classification (Tier-1)", where: "This device (ONNX in Worker)", leaves: "Never", tone: "ok" },
  { what: "Voice → text", where: "Your browser’s speech engine", leaves: "Only to Google/Apple if on-device mode is unavailable — never to Kasauti", tone: "warn" },
  { what: "Text → voice", where: "Your device (speechSynthesis)", leaves: "Never", tone: "ok" },
  { what: "Verdict fingerprint", where: "Shared cache (/api/v)", leaves: "SHA-256 + SimHash + codes only, no raw text", tone: "info" },
  { what: "“Why?” explanation", where: "Groq (text-only)", leaves: "Only when you tap “Why?”", tone: "info" },
  { what: "Telephony / IVR (not built yet)", where: "Telephony provider", leaves: "The one documented exception — no browser on a phone call", tone: "warn" },
];

export function Privacy() {
  const checks = useStore((s) => s.checks);
  const [health, setHealth] = useState<{ store: string; groq: boolean; audioRoutes: number } | null>(null);
  useEffect(() => { fetch("/api/health").then((r) => r.json()).then(setHealth).catch(() => {}); }, []);
  return (
    <>
      <section className="page-head">
        <div><h1>Privacy Ledger</h1><p>Exactly what goes where. Kasauti’s servers never receive, store or transmit a single audio byte.</p></div>
        <span className="pill"><ILock width={14} />No trackers · no SMS/contacts permissions</span>
      </section>
      <section className="grid row-2">
        <article className="card">
          <div className="card-head"><div><h3>Data-flow table</h3><p>Precise attribution — no over-claiming.</p></div></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Data</th><th>Processed by</th><th>Leaves the device?</th></tr></thead>
              <tbody>{FLOWS.map((f) => <tr key={f.what} style={{ cursor: "default" }}><td><b>{f.what}</b></td><td className="muted">{f.where}</td><td><span className={`chip ${f.tone}`}>{f.leaves}</span></td></tr>)}</tbody>
            </table>
          </div>
        </article>
        <article className="card">
          <div className="card-head"><div><h3>Live server audit</h3><p>From /api/health</p></div></div>
          <dl className="kv">
            <dt>Audio-handling routes</dt><dd><b style={{ color: "var(--ok)" }}>{health ? health.audioRoutes : "…"}</b></dd>
            <dt>Verdict store</dt><dd>{health?.store ?? "…"}</dd>
            <dt>Groq (text-only)</dt><dd>{health ? (health.groq ? "configured" : "not configured → deterministic") : "…"}</dd>
            <dt>Checks on this device</dt><dd>{checks.length}</dd>
          </dl>
          <div className="stack" style={{ marginTop: 14, gap: 6 }}>
            {["No /api/asr route exists", "Check history + verdict cache stored only on device", "Model weights are public — no secret logic inside", "Abuse limits live on write paths only"].map((t) => <div key={t} className="row" style={{ gap: 8, flexWrap: "nowrap", fontSize: 12.5 }}><ICheck width={15} style={{ color: "var(--ok)", flex: "none" }} />{t}</div>)}
          </div>
          <button className="btn" style={{ marginTop: 16 }} onClick={() => { if (confirm("Delete all checks stored on this device?")) { clearChecks(); try { localStorage.removeItem("kasauti.verdicts.v1"); } catch { /* ignore */ } } }}>Erase local history</button>
        </article>
      </section>
    </>
  );
}
