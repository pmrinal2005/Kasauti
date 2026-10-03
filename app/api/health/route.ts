import { NextResponse } from "next/server";
import { storeKind } from "@/server/store";

export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json({
    ok: true,
    service: "kasauti",
    store: storeKind,
    groq: Boolean(process.env.GROQ_API_KEY),
    audioRoutes: 0, // there is deliberately no /api/asr — no audio ever reaches this server
    time: new Date().toISOString(),
  });
}
