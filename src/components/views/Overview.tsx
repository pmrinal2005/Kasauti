"use client";
import { useMemo, useState } from "react";
import { BarChart, Donut, LineChart, Sparkline } from "../charts";
import { IAlert, IBolt, ICpu, IShield, IUsers, IDownload } from "../icons";
import { demoRecords, FAMILY_SHARE, LANG_SHARE, networkSeries, type CheckRecord } from "@/lib/data";
import { RUNG_META, type Rung } from "@/core/fusion";
import { go, useStore } from "@/lib/store";

const RANGES = { "7D": 7, "30D": 30, "90D": 90 } as const;
const LANG_COLORS = ["#7c8cff", "#38e1c4", "#f5c66b", "#fb7185", "#60a5fa", "#a78bfa", "#64748b"];

function pct(a: number, b: number) { return b ? ((a - b) / b) * 100 : 0; }

export function Overview() {
  const [range, setRange] = useState<keyof typeof RANGES>("30D");
  const [selFam, setSelFam] = useState<number | null>(null);
  const local = useStore((s) => s.checks);
  const series = useMemo(() => networkSeries(RANGES[range] * 2), [range]);
  const cur = series.slice(-RANGES[range]);
  const prev = series.slice(0, RANGES[range]);
  const sum = (a: typeof cur, k: "checks" | "scams" | "cacheHits" | "tier1") => a.reduce((s, d) => s + d[k], 0);

  const kpis = [
    { label: "Messages checked", value: sum(cur, "checks"), prev: sum(prev, "checks"), icon: IShield, color: "#7c8cff", spark: cur.map((d) => d.checks), fmt: (v: number) => v.toLocaleString("en-IN"), good: "up" },
    { label: "Scam patterns caught", value: sum(cur, "scams"), prev: sum(prev, "scams"), icon: IAlert, color: "#fb7185", spark: cur.map((d) => d.scams), fmt: (v: number) => v.toLocaleString("en-IN"), good: "up" },
    { label: "Shared-cache hit rate", value: (sum(cur, "cacheHits") / sum(cur, "checks")) * 100, prev: (sum(prev, "cacheHits") / sum(prev, "checks")) * 100, icon: IBolt, color: "#38e1c4", spark: cur.map((d) => d.cacheHits / d.checks), fmt: (v: number) => `${v.toFixed(1)}%`, good: "up" },
    { label: "Server inference cost", value: 0, prev: 0, icon: ICpu, color: "#f5c66b", spark: cur.map((d) => d.tier1), fmt: () => "₹0", good: "flat", sub: `${sum(cur, "tier1").toLocaleString("en-IN")} on-device Laya runs` },
  ];

  const labels = cur.map((d) => d.day.slice(5).replace("-", "/"));
  const famData = FAMILY_SHARE.map((f) => ({ label: f.label, value: Math.round(f.count * (RANGES[range] / 30)) }));

  return (
    <>
      <section className="page-head" id="overview-head">
        <div>
          <h1>Protection Overview</h1>
          <p>Network signals from the shared verdict cache (hashes &amp; codes only) — plus your own checks, which never leave this device.</p>
        </div>
        <div className="row">
          <div className="seg" role="tablist" aria-label="Time range">
            {(Object.keys(RANGES) as Array<keyof typeof RANGES>).map((r) => <button key={r} role="tab" aria-selected={range === r} className={range === r ? "on" : ""} onClick={() => setRange(r)}>{r}</button>)}
          </div>
          <button className="btn primary" onClick={() => go("checker")}><IShield />Check a message</button>
        </div>
      </section>

      <section className="grid kpis" aria-label="Key metrics">
        {kpis.map((k) => {
          const d = pct(k.value, k.prev);
          const Icon = k.icon;
          return (
            <article className="card kpi" key={k.label}>
              <div className="kpi-glow" style={{ background: k.color }} />
              <div className="kpi-top">
                <span className="kpi-label">{k.label}</span>
                <span className="kpi-ico" style={{ background: `color-mix(in srgb, ${k.color} 16%, transparent)`, color: k.color }}><Icon /></span>
              </div>
              <div className="kpi-value">{k.fmt(k.value)}</div>
              <div className="kpi-foot">
                {k.good === "flat" ? <span className="trend up">0 servers</span> : <span className={`trend ${d >= 0 ? "up" : "down"}`}>{d >= 0 ? "▲" : "▼"} {Math.abs(d).toFixed(1)}%</span>}
                <Sparkline values={k.spark} color={k.color} />
              </div>
              {k.sub && <div className="faint" style={{ fontSize: 11.5 }}>{k.sub}</div>}
            </article>
          );
        })}
      </section>

      <section className="grid row-2">
        <article className="card">
          <div className="card-head">
            <div><h3>Check volume vs. scam detections</h3><p>Weekend spikes = forwards going viral. Cache hits skip even on-device inference.</p></div>
            <div className="legend"><span><i style={{ background: "#7c8cff" }} />Checks</span><span><i style={{ background: "#38e1c4" }} />Cache hits</span><span><i style={{ background: "#fb7185" }} />Scams</span></div>
          </div>
          <LineChart labels={labels} series={[
            { name: "Checks", color: "#7c8cff", values: cur.map((d) => d.checks), area: true },
            { name: "Cache hits", color: "#38e1c4", values: cur.map((d) => d.cacheHits) },
            { name: "Scams", color: "#fb7185", values: cur.map((d) => d.scams), area: true },
          ]} />
        </article>
        <article className="card">
          <div className="card-head">
            <div><h3>Scam families</h3><p>{selFam == null ? "Tap a bar to inspect" : `${famData[selFam].label}: ${famData[selFam].value.toLocaleString("en-IN")} detections`}</p></div>
          </div>
          <BarChart data={famData} onSelect={setSelFam} selected={selFam} />
          {selFam != null && (
            <div className="note" style={{ marginTop: 8 }}>
              Learn how this script works → <button className="btn ghost" style={{ height: 28, padding: "0 8px" }} onClick={() => go("voice", FAMILY_SHARE[selFam].family === "impersonation" ? "digital_arrest" : FAMILY_SHARE[selFam].family)}>Open explainer</button>
            </div>
          )}
        </article>
      </section>

      <section className="grid row-3">
        <article className="card">
          <div className="card-head"><div><h3>Languages</h3><p>Routed to <span className="mono">laya-multilingual</span></p></div><IUsers width={18} className="muted" /></div>
          <div className="row" style={{ alignItems: "center", gap: 18, flexWrap: "nowrap" }}>
            <Donut parts={LANG_SHARE.map((l, i) => ({ label: l.lang, value: l.pct, color: LANG_COLORS[i] }))} size={132} />
            <div className="stack" style={{ gap: 5, fontSize: 12.5, flex: 1 }}>
              {LANG_SHARE.map((l, i) => <div key={l.code} className="row between" style={{ flexWrap: "nowrap" }}><span><i style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: LANG_COLORS[i], marginRight: 7 }} />{l.lang}</span><b>{l.pct}%</b></div>)}
            </div>
          </div>
        </article>
        <TierCard />
        <article className="card">
          <div className="card-head"><div><h3>Post-incident fast path</h3><p>Lost money? Minutes matter.</p></div></div>
          <ol className="stack" style={{ margin: 0, paddingLeft: 18, gap: 6, fontSize: 13 }}>
            <li><b>Call 1930</b> (National Cyber Crime Helpline) immediately.</li>
            <li>Report at <a href="https://cybercrime.gov.in" target="_blank" rel="noreferrer">cybercrime.gov.in</a>; keep screenshots, UPI IDs, numbers.</li>
            <li>Ask your bank to freeze/flag the transaction.</li>
            <li><b>Beware the “recovery agent”</b> — anyone promising to recover money for a fee is the second scam.</li>
          </ol>
          <a className="btn" style={{ marginTop: 12 }} href="tel:1930"><IAlert />Call 1930</a>
        </article>
      </section>

      <RecentTable local={local} />
    </>
  );
}

function TierCard() {
  const tiers = [
    { n: "T0", name: "Deterministic, on-device", d: "Aho–Corasick lexicon, UPI/caller checks, return maths, offline registry", t: "<10 ms", c: "#38e1c4" },
    { n: "T1", name: "Laya, on-device (ONNX)", d: "1 triage choice → ≤3 noul follow-ups, calibrated", t: "device-bound", c: "#7c8cff" },
    { n: "T2", name: "“Why?” explanation", d: "Text-only Groq call, safeguard-screened, only on request", t: "~1 s", c: "#f5c66b" },
  ];
  return (
    <article className="card">
      <div className="card-head"><div><h3>Evidence cascade</h3><p>Deterministic first, Laya second, LLM last</p></div></div>
      <div className="stack" style={{ gap: 8 }}>
        {tiers.map((t) => (
          <div className="tier" key={t.n}>
            <div className="tier-n" style={{ background: `color-mix(in srgb, ${t.c} 18%, transparent)`, color: t.c }}>{t.n}</div>
            <div style={{ minWidth: 0 }}><div className="row between" style={{ flexWrap: "nowrap" }}><b style={{ fontSize: 13 }}>{t.name}</b><span className="faint mono">{t.t}</span></div><div className="muted" style={{ fontSize: 12 }}>{t.d}</div></div>
          </div>
        ))}
      </div>
    </article>
  );
}

type SortKey = "ts" | "score" | "rung";
const RUNG_ORDER: Record<Rung, number> = { safe_pattern: 0, unclear: 1, caution: 2, likely_scam: 3, known_scam: 4 };

function RecentTable({ local }: { local: CheckRecord[] }) {
  const [filter, setFilter] = useState<"all" | "risky" | "mine">("all");
  const [sort, setSort] = useState<{ k: SortKey; dir: 1 | -1 }>({ k: "ts", dir: -1 });
  const [page, setPage] = useState(0);
  const rows = useMemo(() => {
    let r = [...local.map((c) => ({ ...c, local: true })), ...demoRecords()];
    if (filter === "risky") r = r.filter((x) => RUNG_ORDER[x.rung] >= 2);
    if (filter === "mine") r = r.filter((x) => x.local);
    r.sort((a, b) => sort.dir * (sort.k === "rung" ? RUNG_ORDER[a.rung] - RUNG_ORDER[b.rung] : (a[sort.k] as number) - (b[sort.k] as number)));
    return r;
  }, [local, filter, sort]);
  const PER = 8;
  const pages = Math.max(1, Math.ceil(rows.length / PER));
  const view = rows.slice(page * PER, page * PER + PER);
  const th = (k: SortKey, label: string) => <th aria-sort={sort.k === k ? (sort.dir === 1 ? "ascending" : "descending") : "none"}><button onClick={() => setSort((s) => ({ k, dir: s.k === k ? (s.dir === 1 ? -1 : 1) : -1 }))}>{label}{sort.k === k ? (sort.dir === 1 ? " ↑" : " ↓") : ""}</button></th>;

  const exportCsv = () => {
    const csv = ["time,hash,verdict,score,families,lang,channel,tier,ms,source", ...rows.map((r) => [new Date(r.ts).toISOString(), r.hash.slice(0, 16), r.rung, r.score.toFixed(1), r.families.join("|"), r.lang, r.channel, r.tier, r.ms, r.local ? "this-device" : "network-sample"].join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "kasauti-checks.csv";
    a.click();
  };

  return (
    <article className="card" id="recent-checks">
      <div className="card-head">
        <div><h3>Recent checks</h3><p>Excerpts shown only for <b>your</b> checks (stored locally). Network rows are illustrative samples.</p></div>
        <div className="row">
          <div className="seg">{(["all", "risky", "mine"] as const).map((f) => <button key={f} className={filter === f ? "on" : ""} onClick={() => { setFilter(f); setPage(0); }}>{f === "all" ? "All" : f === "risky" ? "Risky" : "This device"}</button>)}</div>
          <button className="btn" onClick={exportCsv}><IDownload />CSV</button>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr>{th("ts", "Time")}<th>Message</th>{th("rung", "Verdict")}{th("score", "Score")}<th>Signals</th><th>Lang</th><th>Path</th></tr></thead>
          <tbody>
            {view.length === 0 && <tr><td colSpan={7} className="empty">No checks yet on this device — try the Message Checker.</td></tr>}
            {view.map((r) => {
              const m = RUNG_META[r.rung];
              return (
                <tr key={r.id} onClick={() => r.local && go("checker", r.excerpt)} title={r.local ? "Re-open in checker" : undefined}>
                  <td className="muted mono" style={{ whiteSpace: "nowrap" }}>{new Date(r.ts).toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })}</td>
                  <td><div className="truncate">{r.local && <span className="chip neutral" style={{ marginRight: 6 }}>you</span>}{r.excerpt}</div></td>
                  <td><span className={`chip ${m.tone}`}>{m.label}</span></td>
                  <td className="mono">{r.score.toFixed(1)}</td>
                  <td className="muted" style={{ fontSize: 12 }}>{r.families.slice(0, 2).join(", ").replace(/_/g, " ") || "—"}</td>
                  <td className="mono">{r.lang}</td>
                  <td><span className="chip neutral">{r.tier}</span> <span className="faint mono">{r.ms}ms</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row between" style={{ marginTop: 30, fontSize: 12 }}>
        <span className="muted">{rows.length} rows · page {page + 1}/{pages}</span>
        <div className="row"><button className="btn" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>Prev</button><button className="btn" disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)}>Next</button></div>
      </div>
    </article>
  );
}
