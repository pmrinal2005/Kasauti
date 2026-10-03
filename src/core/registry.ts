/**
 * Offline SEBI IA/RA registry lookup.
 *
 * PRODUCTION: the full Brotli-compressed IA/RA snapshot is fetched once by the
 * service worker (cache-first, content-hashed) from /registry/<hash>.json.br.
 * DEMO: this file ships a tiny ILLUSTRATIVE snapshot with fictional entries so
 * the lookup path is exercised end to end. Never treat these as real records.
 */

export interface RegistryEntry {
  reg: string;
  name: string;
  kind: "IA" | "RA";
  status: "active" | "cancelled" | "suspended";
  validTill: string;
}

export const REGISTRY_SNAPSHOT = {
  version: "demo-2026-10-01",
  source: "Illustrative snapshot (fictional entries) — production loads the full SEBI IA/RA list",
  entries: [
    { reg: "INA000017001", name: "Demo Prudent Advisors LLP", kind: "IA", status: "active", validTill: "2029-03-31" },
    { reg: "INA000017002", name: "Demo Nivesh Mitra Pvt Ltd", kind: "IA", status: "suspended", validTill: "2027-06-30" },
    { reg: "INH000009001", name: "Demo Equity Research Desk", kind: "RA", status: "active", validTill: "2030-01-15" },
    { reg: "INH000009002", name: "Demo Bull Signals", kind: "RA", status: "cancelled", validTill: "2025-02-01" },
    { reg: "INH000009003", name: "Demo Analytics Co-op", kind: "RA", status: "active", validTill: "2028-11-30" },
  ] as RegistryEntry[],
};

const INDEX = new Map(REGISTRY_SNAPSHOT.entries.map((e) => [e.reg, e]));

export function lookupRegistration(reg: string): RegistryEntry | null {
  return INDEX.get(reg.toUpperCase()) ?? null;
}
