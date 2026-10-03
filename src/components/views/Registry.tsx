"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { lookupRegistration, REGISTRY_SNAPSHOT } from "@/core/registry";
import { annualise, band, doublingYears, ponziCollapseMonth, fmtPct } from "@/core/plausibility";
import { mulberry32 } from "@/lib/data";

export function Registry() {
  return (
    <>
      <section className="page-head">
        <div><h1>Registry &amp; Calculators</h1><p>Offline SEBI registry lookup, return-plausibility maths and a seeded market simulator — all on-device.</p></div>
        <span className="pill"><span className="dot info" />Snapshot {REGISTRY_SNAPSHOT.version}</span>
      </section>
      <section className="grid row-2b">
        <RegLookup />
        <Plausibility />
      </section>
      <Simulator />
    </>
  );
}

function RegLookup() {
  const [q, setQ] = useState("");
  const e = q.trim().length >= 6 ? lookupRegistration(q.trim()) : null;
  const valid = /^IN[AH]\d{9}$/i.test(q.trim());
  return (
    <article className="card">
      <div className="card-head"><div><h3>Is this adviser registered?</h3><p>{REGISTRY_SNAPSHOT.source}</p></div></div>
      <label htmlFor="reg" className="sr-only">Registration number</label>
      <input id="reg" className="input mono" placeholder="INA000017001 or INH000009001" value={q} onChange={(x) => setQ(x.target.value)} />
      <div style={{ marginTop: 12 }}>
        {!q ? <div className="muted" style={{ fontSize: 12.5 }}>Investment Advisers start with <b>INA</b>, Research Analysts with <b>INH</b>.</div>
          : !valid ? <span className="chip warn">Format should be INA/INH + 9 digits</span>
          : e ? (
            <div className={`verdict-banner ${e.status === "active" ? "ok" : "bad"}`}>
              <div><h4>{e.name}</h4><div className="muted" style={{ fontSize: 12 }}>{e.kind} · {e.status} · valid till {e.validTill}. Registration is not an endorsement — and real advisers never take payment to personal UPI IDs.</div></div>
            </div>
          ) : <div className="verdict-banner bad"><div><h4>Not found in snapshot</h4><div className="muted" style={{ fontSize: 12 }}>Treat as unregistered until verified on SEBI’s official intermediary search.</div></div></div>}
      </div>
      <div className="row" style={{ marginTop: 12, gap: 6 }}>{REGISTRY_SNAPSHOT.entries.map((x) => <button key={x.reg} className="chip neutral mono" style={{ border: 0, cursor: "pointer" }} onClick={() => setQ(x.reg)}>{x.reg}</button>)}</div>
    </article>
  );
}

function Plausibility() {
  const [pct, setPct] = useState(3);
  const [period, setPeriod] = useState<"day" | "week" | "month" | "year">("day");
  const [amt, setAmt] = useState(10000);
  const ann = annualise(pct, period);
  const b = band(ann);
  const tone = b === "implausible" || b === "impossible" ? "bad" : b === "stretch" ? "warn" : "ok";
  const dbl = doublingYears(ann);
  const monthly = (Math.pow(1 + ann / 100, 1 / 12) - 1) * 100;
  const collapse = tone === "bad" ? ponziCollapseMonth(monthly, 100) : null;
  const after1y = amt * (1 + Math.min(ann, 1e9) / 100);
  return (
    <article className="card">
      <div className="card-head"><div><h3>Return plausibility</h3><p>What does “{pct}% per {period}” really mean?</p></div></div>
      <div className="row" style={{ flexWrap: "nowrap" }}>
        <label className="stack" style={{ gap: 4, flex: 1 }}><span className="muted" style={{ fontSize: 12 }}>Promised %</span><input className="input" type="number" min={0} step={0.1} value={pct} onChange={(e) => setPct(Math.max(0, +e.target.value))} /></label>
        <label className="stack" style={{ gap: 4, flex: 1 }}><span className="muted" style={{ fontSize: 12 }}>Per</span>
          <select className="input" value={period} onChange={(e) => setPeriod(e.target.value as typeof period)}><option value="day">day</option><option value="week">week</option><option value="month">month</option><option value="year">year</option></select></label>
        <label className="stack" style={{ gap: 4, flex: 1 }}><span className="muted" style={{ fontSize: 12 }}>Amount ₹</span><input className="input" type="number" min={0} value={amt} onChange={(e) => setAmt(Math.max(0, +e.target.value))} /></label>
      </div>
      <div className={`verdict-banner ${tone}`} style={{ marginTop: 14 }}>
        <div>
          <h4>≈ {fmtPct(ann)} per year, compounded</h4>
          <div className="muted" style={{ fontSize: 12.5 }}>
            ₹{amt.toLocaleString("en-IN")} would become {ann > 1e6 ? "more money than exists in India" : `₹${Math.round(after1y).toLocaleString("en-IN")}`} in one year.
            {tone === "bad" ? " No real asset sustains this — it is how Ponzi schemes advertise." : tone === "warn" ? " Possible only with high risk; never 'guaranteed'." : " Within the range of real long-run asset returns — still not guaranteed."}
          </div>
          <dl className="kv" style={{ marginTop: 10 }}>
            <dt>Money doubles every</dt><dd>{!Number.isFinite(dbl) ? "—" : dbl < 1 / 12 ? `${Math.max(1, Math.round(dbl * 365))} day(s)` : dbl < 1 ? `${(dbl * 12).toFixed(1)} months` : `${dbl.toFixed(1)} years`}</dd>
            {collapse !== null && <><dt>Ponzi arithmetic</dt><dd>Paying this from new deposits, 100 investors must grow past India’s population in <b>{collapse} months</b>.</dd></>}
          </dl>
        </div>
      </div>
    </article>
  );
}

/** Seeded Monte-Carlo crash-and-recovery walk on Canvas 2D (fictional index, zero-allocation hot loop). */
function Simulator() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [seed, setSeed] = useState(7);
  const [years, setYears] = useState(10);
  const [panic, setPanic] = useState(false);
  const N = years * 252;
  const paths = useMemo(() => {
    const r = mulberry32(seed);
    const P = 40;
    const buf = new Float32Array(P * N);
    const crashAt = Math.floor(N * 0.3);
    let finalSum = 0, panicSum = 0;
    for (let p = 0; p < P; p++) {
      let v = 100, sold = -1;
      for (let i = 0; i < N; i++) {
        const z = Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
        let ret = 0.11 / 252 + (0.18 / Math.sqrt(252)) * z;
        if (i >= crashAt && i < crashAt + 40) ret -= 0.009; // ~30% crash over 2 months
        v *= Math.exp(ret);
        if (sold < 0 && i > crashAt && v < 75) sold = v;
        buf[p * N + i] = v;
      }
      finalSum += v;
      panicSum += sold > 0 ? sold * Math.pow(1.065, (N - crashAt) / 252) : v; // panic seller moves to a 6.5% FD
    }
    return { buf, P, crashAt, avgHold: finalSum / P, avgPanic: panicSum / P };
  }, [seed, N]);

  useEffect(() => {
    const c = ref.current!;
    const dpr = Math.min(2, devicePixelRatio || 1);
    const w = c.clientWidth, h = 260;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const { buf, P, crashAt } = paths;
    let max = 0;
    for (let i = 0; i < buf.length; i++) if (buf[i] > max) max = buf[i];
    max = Math.min(max, 600);
    const x = (i: number) => (i / (N - 1)) * w, y = (v: number) => h - 8 - (Math.min(v, max) / max) * (h - 16);
    g.fillStyle = "rgba(251,113,133,0.08)";
    g.fillRect(x(crashAt), 0, x(crashAt + 40) - x(crashAt), h);
    for (let p = 0; p < P; p++) {
      g.beginPath();
      g.strokeStyle = p === 0 ? "#7c8cff" : "rgba(124,140,255,0.13)";
      g.lineWidth = p === 0 ? 2 : 1;
      for (let i = 0; i < N; i += 3) { const v = buf[p * N + i]; i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v)); }
      g.stroke();
    }
    g.setLineDash([4, 4]); g.strokeStyle = "rgba(139,147,167,0.5)"; g.beginPath(); g.moveTo(0, y(100)); g.lineTo(w, y(100)); g.stroke(); g.setLineDash([]);
  }, [paths, N]);

  return (
    <article className="card">
      <div className="card-head">
        <div><h3>Crash &amp; recovery simulator</h3><p>40 seeded paths of a <b>fictional</b> index (11% drift, 18% vol) with a ~30% crash. No real instruments.</p></div>
        <div className="row">
          <div className="seg">{[5, 10, 15].map((y) => <button key={y} className={years === y ? "on" : ""} onClick={() => setYears(y)}>{y}y</button>)}</div>
          <button className="btn" onClick={() => setSeed((s) => s + 1)}>New seed</button>
          <button className={`btn ${panic ? "primary" : ""}`} onClick={() => setPanic((p) => !p)}>{panic ? "Hide" : "Show"} panic-seller</button>
        </div>
      </div>
      <canvas ref={ref} style={{ width: "100%", height: 260, display: "block" }} aria-label="Simulated market paths" role="img" />
      <div className="row" style={{ marginTop: 10, gap: 18 }}>
        <span className="muted">Stayed invested: <b style={{ color: "var(--ok)" }}>₹100 → ₹{paths.avgHold.toFixed(0)}</b></span>
        {panic && <span className="muted">Sold in the crash, moved to FD: <b style={{ color: "var(--bad)" }}>₹100 → ₹{paths.avgPanic.toFixed(0)}</b></span>}
        <span className="faint mono">seed {seed}</span>
      </div>
    </article>
  );
}
