import { NextResponse } from "next/server";
import { getVerdict, putVerdict } from "@/server/store";

export const runtime = "nodejs";

const HEX64 = /^[0-9a-f]{64}$/;
const HEX16 = /^[0-9a-f]{16}$/;
const RUNGS = new Set(["safe_pattern", "unclear", "caution", "likely_scam", "known_scam"]);
const writes = new Map<string, { n: number; t: number }>();

export async function GET(_req: Request, ctx: { params: Promise<{ hash: string }> }) {
  const { hash } = await ctx.params;
  if (!HEX64.test(hash)) return NextResponse.json({ error: "bad hash" }, { status: 400 });
  const v = await getVerdict(hash);
  if (!v) return NextResponse.json({ error: "miss" }, { status: 404, headers: { "cache-control": "public, s-maxage=30" } });
  return NextResponse.json(v, { headers: { "cache-control": "public, s-maxage=300, stale-while-revalidate=86400" } });
}

export async function POST(req: Request, ctx: { params: Promise<{ hash: string }> }) {
  const { hash } = await ctx.params;
  if (!HEX64.test(hash)) return NextResponse.json({ error: "bad hash" }, { status: 400 });
  // crude per-IP write limit — abuse enforcement lives on write paths, not model secrecy
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "anon";
  const now = Date.now();
  const w = writes.get(ip);
  if (w && now - w.t < 60_000 && w.n >= 30) return NextResponse.json({ error: "rate limited" }, { status: 429 });
  writes.set(ip, w && now - w.t < 60_000 ? { n: w.n + 1, t: w.t } : { n: 1, t: now });

  let body: unknown;
  try { body = await req.json(); } catch { return NextResponse.json({ error: "bad json" }, { status: 400 }); }
  const b = body as { simhash?: unknown; rung?: unknown; codes?: unknown; modelVersion?: unknown };
  if (typeof b.simhash !== "string" || !HEX16.test(b.simhash) || typeof b.rung !== "string" || !RUNGS.has(b.rung) || !Array.isArray(b.codes))
    return NextResponse.json({ error: "invalid payload" }, { status: 422 });
  const codes = b.codes.filter((c): c is string => typeof c === "string" && /^[A-Z0-9_]{2,40}$/.test(c)).slice(0, 30);
  const v = await putVerdict({ hash, simhash: b.simhash, rung: b.rung, codes, modelVersion: String(b.modelVersion ?? "").slice(0, 80) });
  return NextResponse.json(v);
}
