/**
 * Return-plausibility maths — framework-free, pure, unit-tested.
 *
 * Anchors (public, regulator-level facts — not advice):
 *  - Bank FDs / RBI repo-linked instruments: single-digit % per year.
 *  - Long-run diversified equity: roughly low-teens % per year, with large drawdowns.
 * Anything "guaranteed" above ~30%/yr has no real-asset precedent and matches Ponzi advertising.
 */

export type Period = "day" | "week" | "month" | "year" | "unknown";
export type PlausibilityBand = "plausible" | "stretch" | "implausible" | "impossible";

export const PLAUSIBLE_ANNUAL = 15;
export const IMPLAUSIBLE_ANNUAL = 30;
export const IMPOSSIBLE_ANNUAL = 200;

const PERIODS_PER_YEAR: Record<Period, number> = { day: 365, week: 52, month: 12, year: 1, unknown: 1 };

/** Compounded annual % for `percent` per `period`. Clamped so the UI never sees Infinity. */
export function annualise(percent: number, period: Period): number {
  if (!(percent > 0)) return 0;
  const n = PERIODS_PER_YEAR[period];
  const v = (Math.pow(1 + percent / 100, n) - 1) * 100;
  return Number.isFinite(v) ? Math.min(v, 1e9) : 1e9;
}

export function band(annualPct: number): PlausibilityBand {
  if (annualPct > IMPOSSIBLE_ANNUAL) return "impossible";
  if (annualPct > IMPLAUSIBLE_ANNUAL) return "implausible";
  if (annualPct > PLAUSIBLE_ANNUAL) return "stretch";
  return "plausible";
}

/** Years for money to double at a given annual % (exact, not rule-of-72). */
export function doublingYears(annualPct: number): number {
  if (!(annualPct > 0)) return Infinity;
  return Math.log(2) / Math.log(1 + annualPct / 100);
}

/**
 * Ponzi arithmetic: how many NEW depositors per month a scheme must recruit just to pay a
 * promised monthly return to existing depositors (everyone deposits the same ticket size),
 * assuming zero real investment income. Returns month-by-month required recruits.
 */
export function ponziRecruitment(monthlyPct: number, startInvestors: number, months: number): number[] {
  const out: number[] = [];
  let investors = Math.max(1, startInvestors);
  for (let m = 0; m < months; m++) {
    const need = Math.ceil(investors * (monthlyPct / 100)); // payouts must come from new money
    out.push(need);
    investors += need;
  }
  return out;
}

/** India's population ≈ 1.45 bn — the month when required recruits exceed it. */
export function ponziCollapseMonth(monthlyPct: number, startInvestors = 100, limit = 1.45e9): number | null {
  if (!(monthlyPct > 0)) return null;
  let investors = Math.max(1, startInvestors);
  for (let m = 1; m <= 1200; m++) {
    investors *= 1 + monthlyPct / 100;
    if (investors > limit) return m;
  }
  return null;
}

export function fmtPct(v: number): string {
  if (v >= 1e6) return "astronomically";
  if (v >= 1000) return `${Math.round(v).toLocaleString("en-IN")}%`;
  return `${v.toFixed(1)}%`;
}
