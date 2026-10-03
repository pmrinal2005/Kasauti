import { NextResponse } from "next/server";
import { storeKind } from "@/server/store";

export const dynamic = "force-dynamic";

export function GET() {
  // `audioRoutes` counts server-side paths that can receive speech. The browser path is 0 by
  // construction (SpeechRecognition runs on the device, no /api/asr exists). The telephony/IVR
  // adapter is the one documented exception, and it counts only when it is switched on.
  const ivr = process.env.IVR_ENABLED === "1";
  return NextResponse.json({
    ok: true,
    service: "kasauti",
    store: storeKind,
    groq: Boolean(process.env.GROQ_API_KEY),
    audioRoutes: ivr ? 1 : 0,
    ivrEnabled: ivr,
    audioStored: false,
    time: new Date().toISOString(),
  });
}
