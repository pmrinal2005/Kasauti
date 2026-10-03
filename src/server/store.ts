/**
 * Shared verdict cache store (server-only).
 * Uses Supabase PostgREST when SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set
 * (table: verdicts(hash text pk, simhash text, rung text, codes text[], model_version text, hits int, updated_at timestamptz)).
 * Otherwise falls back to a per-instance in-memory LRU so the app still works on a bare Vercel deploy.
 * Only hashes + codes are ever stored — never raw message text.
 */
export interface Verdict { hash: string; simhash: string; rung: string; codes: string[]; modelVersion: string; hits: number; updatedAt: string }

const mem = new Map<string, Verdict>();
const SB_URL = process.env.SUPABASE_URL;
const SB_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const sb = SB_URL && SB_KEY;

function headers() {
  return { apikey: SB_KEY!, Authorization: `Bearer ${SB_KEY}`, "content-type": "application/json" };
}

export const storeKind = sb ? "supabase" : "memory";

export async function getVerdict(hash: string): Promise<Verdict | null> {
  if (sb) {
    const r = await fetch(`${SB_URL}/rest/v1/verdicts?hash=eq.${hash}&select=*`, { headers: headers(), cache: "no-store" });
    if (!r.ok) return null;
    const [row] = await r.json();
    return row ? { hash: row.hash, simhash: row.simhash, rung: row.rung, codes: row.codes, modelVersion: row.model_version, hits: row.hits, updatedAt: row.updated_at } : null;
  }
  return mem.get(hash) ?? null;
}

export async function putVerdict(v: Omit<Verdict, "hits" | "updatedAt">): Promise<Verdict> {
  const prev = await getVerdict(v.hash);
  const next: Verdict = { ...v, hits: (prev?.hits ?? 0) + 1, updatedAt: new Date().toISOString() };
  if (sb) {
    await fetch(`${SB_URL}/rest/v1/verdicts`, {
      method: "POST",
      headers: { ...headers(), Prefer: "resolution=merge-duplicates" },
      body: JSON.stringify({ hash: next.hash, simhash: next.simhash, rung: next.rung, codes: next.codes, model_version: next.modelVersion, hits: next.hits, updated_at: next.updatedAt }),
    }).catch(() => {});
  } else {
    if (mem.size > 5000) mem.delete(mem.keys().next().value as string);
    mem.set(v.hash, next);
  }
  return next;
}
