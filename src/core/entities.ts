/** Deterministic entity extraction + checks (UPI handle, caller series, money, returns). */

// Known NPCI-registered PSP handles (subset). Unknown handles are a soft signal, not proof.
const UPI_PSP = new Set([
  "okaxis", "oksbi", "okhdfcbank", "okicici", "ybl", "ibl", "axl", "paytm", "ptyes", "ptaxis", "pthdfc", "ptsbi", "apl", "yapl",
  "upi", "sbi", "icici", "hdfcbank", "axisbank", "kotak", "kmbl", "pnb", "boi", "barodampay", "unionbank", "cnrb", "idfcbank",
  "indus", "yesbank", "fbl", "rbl", "aubank", "jupiteraxis", "freecharge", "airtel", "jio", "slice", "naviaxis", "waicici", "wasbi",
]);

export interface UpiHit {
  id: string;
  handle: string;
  knownPsp: boolean;
}
export interface PhoneHit {
  raw: string;
  kind: "trai_1600" | "mobile" | "international" | "tollfree" | "other";
}
export interface MoneyHit {
  raw: string;
  rupees: number;
}
export interface ReturnClaim {
  raw: string;
  percent: number;
  period: "day" | "week" | "month" | "year" | "unknown";
  annualised: number; // compounded, in %
}

export interface Entities {
  upi: UpiHit[];
  phones: PhoneHit[];
  urls: string[];
  money: MoneyHit[];
  returns: ReturnClaim[];
  registrations: string[];
}

const UPI_RE = /\b([a-z0-9._-]{2,64})@([a-z]{2,32})\b/g;
const PHONE_RE = /(?:\+?\d[\d\s-]{8,15}\d)/g;
const URL_RE = /\b(?:https?:\/\/)?(?:[a-z0-9-]+\.)+(?:com|in|net|org|io|xyz|top|club|app|site|online|info|co|me|link|live)(?:\/[^\s]*)?/g;
const MONEY_RE = /(?:₹|rs\.?|inr|rupees?)\s?([\d,]+(?:\.\d+)?)\s?(lakh|lac|lakhs|crore|cr|k|thousand)?|([\d,]+(?:\.\d+)?)\s?(lakh|lac|lakhs|crore|cr)\b/g;
const RETURN_RE = /(\d{1,4}(?:\.\d+)?)\s?%\s*(?:return|profit|returns|interest|gain|munafa|मुनाफा|रिटर्न)?\s*(?:per|a|every|\/|in|प्रति)?\s*(day|daily|week|weekly|month|monthly|year|yearly|annum|annual|p\.a\.|pa|din|mahina|महीना|दिन|साल)?/g;
const REG_RE = /\b(in[ah]\d{9})\b/g;

const UNIT: Record<string, number> = { lakh: 1e5, lac: 1e5, lakhs: 1e5, crore: 1e7, cr: 1e7, k: 1e3, thousand: 1e3 };

function periodOf(p?: string): ReturnClaim["period"] {
  if (!p) return "unknown";
  if (/^(day|daily|din|दिन)/.test(p)) return "day";
  if (/^week/.test(p)) return "week";
  if (/^(month|mahina|महीना)/.test(p)) return "month";
  if (/^(year|annum|annual|p\.?a|साल)/.test(p)) return "year";
  return "unknown";
}

const PERIODS_PER_YEAR = { day: 365, week: 52, month: 12, year: 1, unknown: 1 } as const;

export function annualise(percent: number, period: ReturnClaim["period"]): number {
  const n = PERIODS_PER_YEAR[period];
  const v = (Math.pow(1 + percent / 100, n) - 1) * 100;
  return Number.isFinite(v) ? Math.min(v, 1e9) : 1e9;
}

export function extractEntities(text: string): Entities {
  const upi: UpiHit[] = [];
  for (const m of text.matchAll(UPI_RE)) {
    const handle = m[2];
    if (/^(gmail|yahoo|outlook|hotmail|icloud|proton)$/.test(handle)) continue; // e-mail, not UPI
    upi.push({ id: m[0], handle, knownPsp: UPI_PSP.has(handle) });
  }

  const phones: PhoneHit[] = [];
  for (const m of text.matchAll(PHONE_RE)) {
    const d = m[0].replace(/[^\d+]/g, "");
    const digits = d.replace(/^\+?91/, "");
    let kind: PhoneHit["kind"] = "other";
    if (/^1600\d{6}$/.test(digits)) kind = "trai_1600";
    else if (/^1800\d{6,7}$/.test(digits)) kind = "tollfree";
    else if (/^[6-9]\d{9}$/.test(digits)) kind = "mobile";
    else if (d.startsWith("+") && !d.startsWith("+91")) kind = "international";
    if (digits.length >= 10) phones.push({ raw: m[0].trim(), kind });
  }

  const urls = [...new Set([...text.matchAll(URL_RE)].map((m) => m[0]))].filter((u) => !u.includes("@"));

  const money: MoneyHit[] = [];
  for (const m of text.matchAll(MONEY_RE)) {
    const num = parseFloat((m[1] ?? m[3] ?? "0").replace(/,/g, ""));
    const unit = (m[2] ?? m[4] ?? "").toLowerCase();
    money.push({ raw: m[0].trim(), rupees: num * (UNIT[unit] ?? 1) });
  }

  const returns: ReturnClaim[] = [];
  for (const m of text.matchAll(RETURN_RE)) {
    const percent = parseFloat(m[1]);
    if (!(percent > 0)) continue;
    const period = periodOf(m[2]);
    // ignore bare percentages with no return/profit context and no period
    if (period === "unknown" && !/return|profit|interest|gain|munafa|मुनाफा|रिटर्न/.test(m[0])) continue;
    returns.push({ raw: m[0].trim(), percent, period, annualised: annualise(percent, period) });
  }

  const registrations = [...text.matchAll(REG_RE)].map((m) => m[1].toUpperCase());

  return { upi, phones, urls, money, returns, registrations };
}
