"use client";
import { useEffect } from "react";
import { Shell } from "./Shell";
import { useStore } from "@/lib/store";
import { Overview } from "./views/Overview";
import { Checker } from "./views/Checker";
import { Lens } from "./views/Lens";
import { Voice } from "./views/Voice";
import { Engine } from "./views/Engine";
import { Registry } from "./views/Registry";
import { Privacy } from "./views/Privacy";

export function Dashboard() {
  const view = useStore((s) => s.view);
  const hydrated = useStore((s) => s.hydrated);
  const toast = useStore((s) => s.toast);

  useEffect(() => {
    if ("serviceWorker" in navigator && location.hostname !== "localhost") navigator.serviceWorker.register("/sw.js").catch(() => {});
  }, []);

  return (
    <Shell>
      {!hydrated ? <div className="empty">Loading…</div> :
        view === "checker" ? <Checker /> :
        view === "lens" ? <Lens /> :
        view === "voice" ? <Voice /> :
        view === "engine" ? <Engine /> :
        view === "registry" ? <Registry /> :
        view === "privacy" ? <Privacy /> : <Overview />}
      {toast && <div className="toast" role="status">{toast}</div>}
    </Shell>
  );
}
