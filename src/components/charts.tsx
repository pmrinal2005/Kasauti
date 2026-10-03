"use client";
/** Hand-rolled SVG charts — zero dependencies, responsive via viewBox, keyboard + pointer tooltips. */
import { useId, useMemo, useRef, useState } from "react";

export interface Series {
  name: string;
  color: string;
  values: number[];
  area?: boolean;
}

const fmt = (n: number) => (n >= 1e5 ? `${(n / 1e5).toFixed(1)}L` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${Math.round(n)}`);

function niceMax(v: number) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

/** Monotone-ish smooth path (Catmull-Rom → cubic Bézier, tension 0.5). */
function smooth(pts: Array<[number, number]>) {
  if (pts.length < 2) return "";
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] ?? p2;
    const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
  }
  return d;
}

export function LineChart({ labels, series, height = 260 }: { labels: string[]; series: Series[]; height?: number }) {
  const W = 760, H = height, pl = 40, pr = 12, pt = 12, pb = 26;
  const uid = useId().replace(/:/g, "");
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement>(null);
  const max = niceMax(Math.max(1, ...series.flatMap((s) => s.values)));
  const n = labels.length;
  const x = (i: number) => pl + (i * (W - pl - pr)) / Math.max(1, n - 1);
  const y = (v: number) => pt + (1 - v / max) * (H - pt - pb);
  const paths = useMemo(
    () => series.map((s) => { const pts = s.values.map((v, i) => [x(i), y(v)] as [number, number]); return { line: smooth(pts), area: `${smooth(pts)} L${x(n - 1)},${H - pb} L${x(0)},${H - pb} Z` }; }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [series, n, max]
  );
  const tickEvery = Math.ceil(n / 7);
  const onMove = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    setHover(Math.max(0, Math.min(n - 1, Math.round(((px - pl) / (W - pl - pr)) * (n - 1)))));
  };
  return (
    <div style={{ position: "relative" }}>
      <svg ref={ref} className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Line chart: ${series.map((s) => s.name).join(", ")}`}
        onPointerMove={onMove} onPointerLeave={() => setHover(null)} tabIndex={0}
        onKeyDown={(e) => { if (e.key === "ArrowRight") setHover((h) => Math.min(n - 1, (h ?? -1) + 1)); if (e.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? n) - 1)); }}>
        <defs>
          {series.map((s, i) => (
            <linearGradient key={i} id={`g${uid}${i}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity="0.32" />
              <stop offset="100%" stopColor={s.color} stopOpacity="0" />
            </linearGradient>
          ))}
          <filter id={`glow${uid}`} x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="grid-line" x1={pl} x2={W - pr} y1={y(max * t)} y2={y(max * t)} />
            <text x={pl - 8} y={y(max * t) + 3} textAnchor="end">{fmt(max * t)}</text>
          </g>
        ))}
        {labels.map((l, i) => (i % tickEvery === 0 || i === n - 1 ? <text key={i} x={x(i)} y={H - 6} textAnchor="middle">{l}</text> : null))}
        {series.map((s, i) => (
          <g key={s.name}>
            {s.area && <path d={paths[i].area} fill={`url(#g${uid}${i})`} />}
            <path d={paths[i].line} fill="none" stroke={s.color} strokeWidth={2.2} filter={i === 0 ? `url(#glow${uid})` : undefined} />
          </g>
        ))}
        {hover !== null && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={pt} y2={H - pb} stroke="var(--stroke-2)" />
            {series.map((s) => <circle key={s.name} cx={x(hover)} cy={y(s.values[hover])} r={4} fill="var(--bg-2)" stroke={s.color} strokeWidth={2} />)}
          </g>
        )}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${(x(hover) / W) * 100}%`, top: `${(Math.min(...series.map((s) => y(s.values[hover]))) / H) * 100}%` }}>
          <div className="faint" style={{ marginBottom: 4 }}>{labels[hover]}</div>
          {series.map((s) => <div key={s.name}><i style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: s.color, marginRight: 6 }} />{s.name}: <b>{s.values[hover].toLocaleString("en-IN")}</b></div>)}
        </div>
      )}
    </div>
  );
}

export function BarChart({ data, color = "var(--accent)", height = 240, onSelect, selected }: {
  data: Array<{ label: string; value: number; color?: string }>; color?: string; height?: number; onSelect?: (i: number | null) => void; selected?: number | null;
}) {
  const W = 520, H = height, pl = 36, pr = 8, pt = 14, pb = 40;
  const uid = useId().replace(/:/g, "");
  const [hover, setHover] = useState<number | null>(null);
  const max = niceMax(Math.max(1, ...data.map((d) => d.value)));
  const bw = (W - pl - pr) / data.length;
  const y = (v: number) => pt + (1 - v / max) * (H - pt - pb);
  return (
    <div style={{ position: "relative" }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Bar chart">
        <defs>
          <linearGradient id={`b${uid}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity="1" /><stop offset="100%" stopColor={color} stopOpacity="0.35" /></linearGradient>
        </defs>
        {[0, 0.5, 1].map((t) => (
          <g key={t}><line className="grid-line" x1={pl} x2={W - pr} y1={y(max * t)} y2={y(max * t)} /><text x={pl - 6} y={y(max * t) + 3} textAnchor="end">{fmt(max * t)}</text></g>
        ))}
        {data.map((d, i) => {
          const h = H - pb - y(d.value);
          const dim = selected != null && selected !== i;
          return (
            <g key={d.label} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect className="bar" x={pl + i * bw + bw * 0.18} y={y(d.value)} width={bw * 0.64} height={Math.max(0, h)} rx={7}
                fill={d.color ?? `url(#b${uid})`} opacity={dim ? 0.3 : 1} tabIndex={0} role="button" aria-label={`${d.label}: ${d.value}`}
                onClick={() => onSelect?.(selected === i ? null : i)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onSelect?.(selected === i ? null : i); }} />
              <text x={pl + i * bw + bw / 2} y={H - 22} textAnchor="middle">{d.label.length > 11 ? d.label.slice(0, 10) + "…" : d.label}</text>
            </g>
          );
        })}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${((pl + hover * bw + bw / 2) / W) * 100}%`, top: `${(y(data[hover].value) / H) * 100}%` }}>
          {data[hover].label}: <b>{data[hover].value.toLocaleString("en-IN")}</b>
        </div>
      )}
    </div>
  );
}

export function Sparkline({ values, color, width = 110, height = 34 }: { values: number[]; color: string; width?: number; height?: number }) {
  const uid = useId().replace(/:/g, "");
  const min = Math.min(...values), max = Math.max(...values);
  const pts = values.map((v, i) => [(i / (values.length - 1)) * width, height - 3 - ((v - min) / (max - min || 1)) * (height - 6)] as [number, number]);
  const d = smooth(pts);
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <defs><linearGradient id={`s${uid}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity="0.35" /><stop offset="100%" stopColor={color} stopOpacity="0" /></linearGradient></defs>
      <path d={`${d} L${width},${height} L0,${height} Z`} fill={`url(#s${uid})`} />
      <path d={d} fill="none" stroke={color} strokeWidth={1.8} />
    </svg>
  );
}

export function Donut({ parts, size = 150 }: { parts: Array<{ label: string; value: number; color: string }>; size?: number }) {
  const total = parts.reduce((s, p) => s + p.value, 0) || 1;
  const r = size / 2 - 12, c = 2 * Math.PI * r;
  let acc = 0;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Distribution">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--glass-2)" strokeWidth={14} />
      {parts.map((p) => {
        const len = (p.value / total) * c;
        const el = <circle key={p.label} cx={size / 2} cy={size / 2} r={r} fill="none" stroke={p.color} strokeWidth={14} strokeDasharray={`${Math.max(0, len - 3)} ${c}`} strokeDashoffset={-acc} transform={`rotate(-90 ${size / 2} ${size / 2})`} strokeLinecap="round" />;
        acc += len;
        return el;
      })}
    </svg>
  );
}
