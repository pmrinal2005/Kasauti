"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { go, hydrate, setLang, useStore, type Lang, type View } from "@/lib/store";
import { IBook, ICpu, IGrid, ILock, IMenu, IMic, IMoon, ISearch, IShield, ISidebar, ISun, IDb, IBell } from "./icons";
import { CONCEPTS } from "@/lib/concepts";

const NAV: Array<{ id: View; label: string; icon: (p: React.SVGProps<SVGSVGElement>) => React.ReactElement; badge?: string; group: string }> = [
  { id: "overview", label: "Overview", icon: IGrid, group: "Monitor" },
  { id: "checker", label: "Message Checker", icon: IShield, badge: "T0·T1", group: "Protect" },
  { id: "voice", label: "Voice Explainer", icon: IMic, group: "Protect" },
  { id: "registry", label: "Registry & Calculators", icon: IDb, group: "Protect" },
  { id: "engine", label: "On-Device Engine", icon: ICpu, badge: "Laya", group: "System" },
  { id: "privacy", label: "Privacy Ledger", icon: ILock, group: "System" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const view = useStore((s) => s.view);
  const lang = useStore((s) => s.lang);
  const checks = useStore((s) => s.checks);

  useEffect(() => {
    hydrate();
    const t = (localStorage.getItem("kasauti.theme") as "dark" | "light") || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    setTheme(t);
    setCollapsed(localStorage.getItem("kasauti.collapsed") === "1");
  }, []);
  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);

  const toggleCollapse = () => { setCollapsed((c) => { localStorage.setItem("kasauti.collapsed", c ? "0" : "1"); return !c; }); };
  const toggleTheme = () => setTheme((t) => { const n = t === "dark" ? "light" : "dark"; localStorage.setItem("kasauti.theme", n); return n; });

  let lastGroup = "";
  return (
    <div className={`shell ${collapsed ? "collapsed" : ""} ${mobileOpen ? "mobile-open" : ""}`}>
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">क</div>
          <div className="brand-text"><div className="brand-name">Kasauti</div><div className="brand-sub">Touchstone for every forward</div></div>
        </div>
        <nav className="stack" style={{ gap: 2 }}>
          {NAV.map((n) => {
            const header = n.group !== lastGroup ? <div className="nav-label">{n.group}</div> : null;
            lastGroup = n.group;
            const Icon = n.icon;
            return (
              <div key={n.id}>
                {header}
                <button className={`nav-item ${view === n.id ? "active" : ""}`} onClick={() => { go(n.id); setMobileOpen(false); }} title={collapsed ? n.label : undefined} aria-current={view === n.id ? "page" : undefined}>
                  <Icon /><span className="nav-text">{n.label}</span>{n.badge && <span className="nav-badge">{n.badge}</span>}
                </button>
              </div>
            );
          })}
        </nav>
        <div className="side-foot">
          <div className="row" style={{ gap: 8, justifyContent: collapsed ? "center" : undefined }}><span className="dot" /><span className="side-foot-text"><b>Offline-ready</b></span></div>
          <div className="side-foot-text muted" style={{ marginTop: 6 }}>Fraud? Call <b style={{ color: "var(--text)" }}>1930</b> · <a href="https://cybercrime.gov.in" target="_blank" rel="noreferrer">cybercrime.gov.in</a></div>
        </div>
      </aside>
      {mobileOpen && <div className="drawer-bg mobile-only" style={{ zIndex: 45 }} onClick={() => setMobileOpen(false)} />}

      <div className="main">
        <header className="topbar">
          <button className="icon-btn mobile-only" aria-label="Open menu" onClick={() => setMobileOpen(true)}><IMenu /></button>
          <button className="icon-btn desktop-only" aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} onClick={toggleCollapse}><ISidebar /></button>
          <CommandSearch checksCount={checks.length} />
          <div className="top-right">
            <span className="pill desktop-only"><span className="dot" />0 audio bytes sent</span>
            <select className="lang" aria-label="Language" value={lang} onChange={(e) => setLang(e.target.value as Lang)}>
              <option value="en-IN">English</option><option value="hi-IN">हिन्दी</option><option value="mr-IN">मराठी</option><option value="ta-IN">தமிழ்</option>
            </select>
            <button className="icon-btn" aria-label="Toggle theme" onClick={toggleTheme}>{theme === "dark" ? <ISun /> : <IMoon />}</button>
            <button className="icon-btn desktop-only" aria-label="Alerts" onClick={() => go("overview")}><IBell /></button>
            <div className="avatar" title="No account needed — your profile lives only on this device">
              <div className="avatar-img">RK</div>
              <div className="avatar-meta"><b>Raksha Profile</b><span>Local device · no login</span></div>
            </div>
          </div>
        </header>
        <main className="content" id="main">{children}</main>
      </div>
    </div>
  );
}

function CommandSearch({ checksCount }: { checksCount: number }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [sel, setSel] = useState(0);
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const k = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); ref.current?.focus(); setOpen(true); } };
    addEventListener("keydown", k);
    return () => removeEventListener("keydown", k);
  }, []);

  const results = useMemo(() => {
    const s = q.trim().toLowerCase();
    const items: Array<{ label: string; hint: string; run: () => void; icon: React.ReactNode }> = [];
    for (const n of NAV) if (!s || n.label.toLowerCase().includes(s)) items.push({ label: n.label, hint: "Go to", run: () => go(n.id), icon: <n.icon width={16} height={16} /> });
    for (const c of CONCEPTS) if (s && (c.title.toLowerCase().includes(s) || c.keys.some((k) => k.includes(s)))) items.push({ label: c.title, hint: "Concept", run: () => go("voice", c.id), icon: <IBook width={16} height={16} /> });
    if (s.length > 12) items.unshift({ label: `Check “${q.slice(0, 40)}${q.length > 40 ? "…" : ""}”`, hint: "Run Tier-0", run: () => go("checker", q), icon: <IShield width={16} height={16} /> });
    return items.slice(0, 9);
  }, [q]);

  const run = (i: number) => { results[i]?.run(); setOpen(false); setQ(""); ref.current?.blur(); };
  return (
    <div className="search">
      <ISearch />
      <label htmlFor="global-search" className="sr-only">Search or paste a message</label>
      <input id="global-search" ref={ref} value={q} placeholder={`Search concepts, tools — or paste a message to check (${checksCount} local)`} autoComplete="off"
        onChange={(e) => { setQ(e.target.value); setSel(0); setOpen(true); }} onFocus={() => setOpen(true)} onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => { if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(results.length - 1, s + 1)); } if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); } if (e.key === "Enter") run(sel); if (e.key === "Escape") { setOpen(false); ref.current?.blur(); } }}
        role="combobox" aria-expanded={open} aria-controls="search-pop" />
      <span className="kbd desktop-only">Ctrl K</span>
      {open && results.length > 0 && (
        <div className="search-pop" id="search-pop" role="listbox">
          {results.map((r, i) => (
            <button key={i} role="option" aria-selected={i === sel} onMouseDown={(e) => { e.preventDefault(); run(i); }} onMouseEnter={() => setSel(i)}>
              <span className="muted">{r.icon}</span><span>{r.label}</span><span className="faint" style={{ marginLeft: "auto", fontSize: 11 }}>{r.hint}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
