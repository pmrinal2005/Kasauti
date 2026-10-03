/**
 * Landing page — the first screen, and the place where the privacy claim has to be exact.
 *
 * Deliberately a **server component with no client JavaScript**: this page is the one everybody on
 * a slow connection sees first, and it should cost one HTML response. The dashboard (which does need
 * workers and canvases) is one tap away.
 *
 * Copy rules that are load-bearing:
 *   * the claim is "Kasauti's own servers never receive your audio" — not "no audio leaves the
 *     device", which would be false for Chrome and Safari;
 *   * the telephony exception is stated on the landing page, not buried;
 *   * no accuracy number appears anywhere without the eval-gate panel that produced it.
 */
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Kasauti — check any investment message before you pay",
  description:
    "Paste a WhatsApp forward, SMS or voice-note transcript and see the evidence behind a scam warning in about 20 ms. Checks run on your device; Kasauti's servers never receive your audio.",
};

const STEPS = [
  {
    n: "01",
    t: "Paste or speak the message",
    d: "A forwarded “guaranteed 3% daily profit”, a KYC-expiry SMS, a police threat, a tips-group invite. Typed, pasted, or dictated by your own browser's speech engine.",
  },
  {
    n: "02",
    t: "Deterministic checks run first",
    d: "Normalisation, multilingual lexicons, UPI/TRAI/SEBI entity checks and return-plausibility maths — a few milliseconds, entirely offline, and auditable line by line.",
  },
  {
    n: "03",
    t: "Laya reads what a keyword list cannot",
    d: "A quantized decision model runs inside a Web Worker: is this an offer, a threat, a payment request? One triage question, then only the follow-ups worth asking. Uncertain answers are shown as uncertain.",
  },
];

const HONEST = [
  ["Runs on your device", "The model is a static file fetched once from a public Hugging Face repo and cached in your browser. No inference server exists to send anything to."],
  ["Audio, precisely", "Chrome and Safari do their own speech recognition — their policy applies. Kasauti's own servers never receive your audio: there is no /api/asr on this deployment."],
  ["One documented exception", "A sponsored phone line has no browser, so its provider transcribes the call. That route is off by default and says so on /api/health."],
  ["Uncertainty is shown", "Below the calibrated threshold the answer is “unsettled”, and the app says the deterministic checks are all it has. Absence of evidence is not safety."],
  ["The one thing that sends text", "Tap “Why?” and the message goes to our explanation model (Groq) to be reworded in plain language. It is the only path that sends your text anywhere, it never sends audio, and everything else — the checks, the verdict — has already happened on your device without it."],
];

export default function Home() {
  return (
    <main className="landing">
      <header className="landing-top">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">क</div>
          <div className="brand-text">
            <div className="brand-name">Kasauti</div>
            <div className="brand-sub">Touchstone for every forward</div>
          </div>
        </div>
        <nav className="row" style={{ gap: 10 }}>
          <Link className="btn" href="/dashboard#checker">Check a message</Link>
          <Link className="btn ghost" href="/dashboard">Open the dashboard</Link>
        </nav>
      </header>

      <section className="landing-hero">
        <p className="pill"><span className="dot" />0 audio bytes sent · runs in your browser</p>
        <h1>
          Before you pay, <span className="grad">check it.</span>
        </h1>
        <p className="lead">
          Kasauti reads an investment message the way a careful friend would: first the mechanical
          checks — payment handles, regulator numbers, “3% daily” maths — then a small decision model
          that runs on your own phone. You get the evidence, not just a score.
        </p>
        <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
          <Link className="btn primary" href="/dashboard#checker">Check a message — no login</Link>
          <Link className="btn" href="/dashboard#lens">Promotion or education?</Link>
          <Link className="btn ghost" href="/dashboard#privacy">What leaves my device?</Link>
        </div>
        <p className="hint" style={{ marginTop: 12 }}>
          Works offline after the first load. Nothing to install, no account, and nothing about your
          check is stored.  Tap “Why?” and the message goes to our explanation model to be reworded — that
          is the only thing that ever sends text anywhere, and it is always your choice.
        </p>
      </section>

      <section className="landing-grid">
        {STEPS.map((s) => (
          <article className="card" key={s.n}>
            <div className="tier">
              <div className="tier-n" style={{ background: "var(--glass-2)" }}>{s.n}</div>
              <div>
                <b>{s.t}</b>
                <p className="hint" style={{ marginBottom: 0 }}>{s.d}</p>
              </div>
            </div>
          </article>
        ))}
      </section>

      <section className="landing-grid">
        <article className="card">
          <h2>What we will not pretend</h2>
          <ul className="list">
            {HONEST.map(([t, d]) => (
              <li key={t}><b>{t}</b> — <span className="muted">{d}</span></li>
            ))}
          </ul>
        </article>
        <article className="card">
          <h2>If you have already paid</h2>
          <p className="muted" style={{ marginTop: 0 }}>
            Speed matters more than anything else. Report within the first hours and a transaction can
            still be frozen.
          </p>
          <ol className="list">
            <li>Call your bank and ask them to freeze or flag the transfer.</li>
            <li>Report on the national cybercrime portal, or call the financial-fraud helpline.</li>
            <li>Keep the screenshots, UPI IDs, phone numbers and the app name — the portal asks for them.</li>
          </ol>
          <div className="row" style={{ gap: 10, marginTop: 10 }}>
            <a className="btn primary" href="tel:1930">Call 1930</a>
            <a className="btn" href="https://cybercrime.gov.in" target="_blank" rel="noreferrer">cybercrime.gov.in</a>
            <Link className="btn ghost" href="/dashboard#registry">Check a regulator number</Link>
          </div>
        </article>
      </section>

      <section className="card landing-close">
        <h2>Kasauti never recommends a stock, fund, broker or trade.</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          It is a check, not an adviser. Everything it shows you is either a deterministic rule you
          can read in the source, or a model answer with its confidence and its calibration shown next
          to it. Where the app cannot tell, it says so.
        </p>
        <Link className="btn primary" href="/dashboard">Open the dashboard</Link>
      </section>
    </main>
  );
}
