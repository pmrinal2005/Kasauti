/**
 * Telephony / IVR — the ONE documented audio exception.
 *
 * Everywhere else in Kasauti, audio never touches a Kasauti-operated process: the browser's own
 * `SpeechRecognition` turns speech into text on the visitor's device. A phone line cannot work that
 * way — the telco/CPaaS provider must transcribe the call, and that transcript reaches this server.
 * The honest way to ship that is to say so, loudly and in the same breath as the feature:
 *
 *   * the route is **off unless `IVR_ENABLED=1`**, so a default deploy has `audioRoutes: 0`;
 *   * the provider (Twilio, Exotel, Plivo…) does the recognition. We never receive, store or forward
 *     audio, and we never write the transcript to disk or to the shared store;
 *   * what we *do* keep is the same thing the browser keeps for a cache hit: a SHA-256 of the
 *     normalised text plus the evidence codes — the identical contract as `/api/v/[hash]`;
 *   * the spoken answer comes from the deterministic Tier-0 core, so it can be spoken in one breath
 *     and audited line by line. A generative model is never allowed to be the voice on a phone call.
 *
 * This module is pure: it takes a `Tier0Result` and produces words and XML. The route does the I/O.
 */
import type { Rung, Tier0Result } from "./fusion";

export type IvrLang = "en" | "hi";

export interface IvrLine {
  /** what the caller hears, in the requested language */
  say: string;
  /** machine-readable, returned in JSON mode so a scaffold/CTI system can route on it */
  rung: Rung;
  score: number;
  codes: string[];
  /** true when the answer must not be spoken as a verdict (nothing decisive was found) */
  uncertain: boolean;
}

const HELPLINE_EN = "Note the caller's number and the payment address, then call 1 9 3 0, or report at cybercrime dot gov dot in.";
const HELPLINE_HI = "कॉलर का नंबर और भुगतान पता नोट करें, फिर 1 9 3 0 पर कॉल करें, या cybercrime dot gov dot in पर रिपोर्ट करें।";

/**
 * One spoken line per rung. Deliberately conservative:
 *   * `unclear` is said as "we could not confirm", never as "safe";
 *   * `safe_pattern` carries its own disclaimer, because absence of evidence is not safety;
 *   * nothing here recommends, predicts or promises — the same guardrail as the text explainer.
 */
export function spokenVerdict(t: Pick<Tier0Result, "rung" | "score" | "evidence">, lang: IvrLang = "en"): IvrLine {
  const codes = t.evidence.map((e) => e.code);
  const rung = t.rung as Rung;
  const helpline = lang === "hi" ? HELPLINE_HI : HELPLINE_EN;

  const say = ((): string => {
    switch (rung) {
      case "known_scam":
      case "likely_scam":
        return lang === "hi"
          ? `सावधान। इस संदेश में ठगी के स्पष्ट संकेत मिले हैं। कोई पैसा न भेजें, ऐप इंस्टॉल न करें। ${helpline}`
          : `Warning. This message shows clear signs of a scam. Do not send money and do not install any app they ask for. ${helpline}`;
      case "caution":
        return lang === "hi"
          ? `सतर्क रहें। इसमें कुछ चेतावनी भरे संकेत हैं, इसलिए पैसा भेजने से पहले जाँच करें। ${helpline}`
          : `Be careful. There are some warning signs here, so verify before sending anything. ${helpline}`;
      case "unclear":
        return lang === "hi"
          ? `हम इस संदेश की पुष्टि नहीं कर पाए। निष्कर्ष निकालने से पहले जानकार व्यक्ति से जाँच करें। ${helpline}`
          : `We could not confirm this one. Before acting, have someone you trust check it. ${helpline}`;
      default:
        return lang === "hi"
          ? `इस संदेश में ठगी का कोई मज़बूत संकेत नहीं मिला। यह सुरक्षा की गारंटी नहीं है। ${helpline}`
          : `No strong scam pattern was found in this message. That is not a guarantee of safety. ${helpline}`;
    }
  })();

  return { say, rung, score: t.score, codes, uncertain: rung === "unclear" };
}

/** XML text escaping — the transcript is attacker-controlled and lands in our response. */
export function xmlEscape(s: string): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** Strip anything that has no business being spoken: control chars and runaway whitespace. */
export function cleanSpokenInput(raw: string, maxChars = 1200): string {
  return String(raw ?? "")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxChars);
}

export interface TwimlOptions {
  lang?: IvrLang;
  /** set to keep the line open and listen for the next utterance */
  gather?: { action: string; speechTimeout?: number; hint?: string };
}

/**
 * A single `<Response>` shape that both Twilio and Exotel accept (Exotel's XML is a compatible
 * subset). If a provider needs its own dialect, this is the one function to fork.
 */
export function ivrXml(line: IvrLine, opts: TwimlOptions = {}): string {
  const lang = opts.lang === "hi" ? "hi-IN" : "en-IN";
  const say = `<Say language="${lang}">${xmlEscape(line.say)}</Say>`;
  if (!opts.gather) return `<?xml version="1.0" encoding="UTF-8"?>\n<Response>${say}<Hangup/></Response>`;
  const g = opts.gather;
  return (
    `<?xml version="1.0" encoding="UTF-8"?>\n<Response>${say}` +
    `<Gather input="speech" language="${lang}" action="${xmlEscape(g.action)}" ` +
    `speechTimeout="${g.speechTimeout ?? 6}"${g.hint ? ` hints="${xmlEscape(g.hint)}"` : ""}/>` +
    `<Say language="${lang}">${xmlEscape(opts.lang === "hi" ? "कोई जवाब नहीं मिला।" : "We did not hear anything.")}</Say>` +
    `<Hangup/></Response>`
  );
}
