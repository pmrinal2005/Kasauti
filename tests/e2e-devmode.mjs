/**
 * Browser end-to-end: the real inference path, against the bundled dev model.
 *
 * This is the only test that exercises what actually ships: a Web Worker importing onnxruntime-web
 * from a pinned CDN, downloading a manifest + graph + pruned tokenizer, verifying sha256 of every
 * byte, building the Laya sequence with the pure-TS tokenizer, running the ONNX graph and decoding
 * calibrated answers. Everything else in `tests/` is unit-level; this is the integration proof.
 *
 *   npm run dev                                     # in another shell (NEXT_PUBLIC_LAYA_MODEL_BASE=/dev-model)
 *   LD_LIBRARY_PATH=/home/user/.libs/root/usr/lib/x86_64-linux-gnu \
 *     node tests/e2e-devmode.mjs http://localhost:3000
 *
 * Requires the dev-model export:  python -m ml.kasauti_ml.pipeline --profile smoke --out ml/out/dev \
 *                                   --public-dir public/dev-model
 */
import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:3000";
const step = (m) => console.log(`\n▸ ${m}`);

const browser = await chromium.launch({
  args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-gl=swiftshader"],
});
const page = await browser.newPage();
const logs = [];
const http4xx = [];
page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
// the shared-verdict endpoint (/api/v/<hash>) legitimately 404s for an unknown hash — the client
// treats that as a cache miss, so it is not an error. Anything else 4xx/5xx is.
page.on("response", (r) => { if (r.status() >= 400) http4xx.push(`${r.status()} ${r.url()}`); });

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures++;
};

try {
  step(`open ${BASE}/dashboard#engine`);
  await page.goto(`${BASE}/dashboard#engine`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("text=On-Device Engine", { timeout: 20000 });

  step("enable deep checking (loads the dev model into the worker)");
  const enable = page.getByRole("button", { name: /Enable deep checking|Download anyway/ });
  await enable.waitFor({ timeout: 20000 });
  if (await enable.isDisabled()) {
    console.log("  (button disabled — model already enabled or capability says Tier-0 only)");
  } else {
    await enable.click();
  }

  step("wait for the worker to finish loading the graph");
  // Read the Status row specifically — the page text contains "ready"/"error" in unrelated copy.
  const readStatus = () =>
    page.evaluate(() => {
      const dd = [...document.querySelectorAll("dd")].find((d) =>
        d.previousElementSibling?.textContent?.includes("Status"));
      return (dd?.textContent ?? "").trim();
    });
  let statusText = "";
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    statusText = await readStatus();
    if (/^(ready|unpublished|error|deferred)$/.test(statusText)) break;
    await page.waitForTimeout(500);
  }
  check("worker reached a terminal status", /^(ready|unpublished|error|deferred)$/.test(statusText), statusText);

  if (statusText !== "ready") {
    console.log(logs.slice(-20).join("\n"));
    throw new Error(`model did not become ready: ${statusText}`);
  }

  step("manifest panel reflects the export that was loaded");
  const panel = await page.evaluate(() => document.body.innerText);
  check("model version is the dev export", /kasauti-laya-(tiny-dev|multilingual)@/.test(panel));
  check("flagged uncalibrated (report-only)", /uncalibrated|report-only/i.test(panel), "dev model must never act");

  step("tokenizer inspection: exact model input for a real message");
  await page.locator("#probe").fill("Guaranteed 3% daily profit, join VIP group today only");
  await page.getByRole("button", { name: /Build sequence|Tokenizing/ }).click();
  await page.waitForFunction(() => /Sequence length|tokens/i.test(document.body.innerText), null, { timeout: 30000 });
  const inspect = await page.evaluate(() => document.body.innerText.match(/Sequence length[\s\S]{0,80}/)?.[0] ?? "");
  check("sequence built by the pure-TS tokenizer", /Sequence length\s*\d+/.test(inspect), inspect.replace(/\s+/g, " "));

  step("run a full check through the Checker (Tier-0 + Laya)");
  await page.getByRole("button", { name: "Message Checker" }).click();
  await page.waitForSelector("text=/Check message|Checking/", { timeout: 20000 });
  const box = page.locator("textarea").first();
  await box.fill("Congratulations! Join our VIP stock tips group. Guaranteed 3% daily profit, zero risk. Pay ₹5,000 to vipfund@ybl");
  await page.getByRole("button", { name: /Check message|Checking/ }).click();
  // wait for the fused verdict card itself — "Laya" alone appears in the nav long before the check ends
  await page.waitForFunction(
    () => /No scam pattern found|Not enough evidence|Be careful|Likely scam|Matches known scam script/.test(document.body.innerText),
    null,
    { timeout: 90000 },
  );
  await page.waitForFunction(() => /Laya Tier-1/.test(document.body.innerText), null, { timeout: 30000 })
    .catch(() => {});
  const result = await page.evaluate(() => document.body.innerText);
  const rung = result.match(/No scam pattern found|Not enough evidence|Be careful|Likely scam|Matches known scam script/)?.[0] ?? "";
  check("the fused verdict renders a rung", rung.length > 0, rung);
  const tier1 = (result.match(/Laya Tier-1[^\n]*/) ?? [""])[0].slice(0, 200);
  check("Tier-1 answered in the browser from the ONNX graph", /wasm|webgpu/.test(tier1) && /\d+\s*ms/.test(tier1), tier1);
  check("an uncalibrated dev model abstains instead of acting", /abstained/.test(tier1), tier1);
  check("Tier-1 answers are rendered with a backend", /webgpu|wasm-st|wasm-mt/.test(result), (result.match(/Laya Tier-1[^\n]*/) ?? [""])[0].slice(0, 120));

  step("the check is persisted on the device (no account, no server)");
  const ls = await page.evaluate(() => ({
    checks: JSON.parse(localStorage.getItem("kasauti.checks.v1") || "[]").length,
    verdicts: Object.keys(JSON.parse(localStorage.getItem("kasauti.verdicts.v1") || "{}")).length,
  }));
  check("history row written to localStorage", ls.checks >= 1, `checks=${ls.checks} verdicts=${ls.verdicts}`);

  step("Cache API holds the model bytes (offline path)");
  const cached = await page.evaluate(async () => {
    const names = await caches.keys();
    if (!names.length) return { names, entries: 0 };
    const c = await caches.open(names[0]);
    const keys = await c.keys();
    return { names, entries: keys.length, urls: keys.map((k) => k.url.split("/").pop()) };
  });
  check("model cache populated", cached.entries >= 2, JSON.stringify(cached).slice(0, 200));

  step("Content Lens: a long transcript is windowed and read window by window");
  await page.getByRole("button", { name: "Content Lens" }).click();
  await page.waitForSelector("#lens-in", { timeout: 20000 });
  await page.getByRole("button", { name: "Transcript / call notes" }).click();
  const transcript = Array.from({ length: 12 }, (_, i) =>
    `Agent: plan ${i} is simple sir, the return is fixed and guaranteed. Customer: how much do I transfer? ` +
    `Agent: send it to vipfund${i}@ybl and install AnyDesk so our team helps. `).join("\n");
  await page.locator("#lens-in").fill(transcript);
  await page.getByRole("button", { name: /Read it window by window|Place it on the line/ }).click();
  await page.waitForSelector("text=/How it was cut|window 1/i", { timeout: 120000 });
  const lens = await page.evaluate(() => document.body.innerText);
  const windows = [...lens.matchAll(/window (\d+)/g)].map((m) => Number(m[1]));
  check("the transcript was cut into overlapping windows", Math.max(...windows) >= 2, `${Math.max(...windows)} windows`);
  check("the windowing is explained, including anything dropped", /speaker turns|paragraph breaks|sentence ends|character count/.test(lens));
  check("an uncalibrated model leaves every window unsettled", /unsettled/.test(lens));

  if (process.env.E2E_IVR === "1") {
    step("telephony adapter — only when the server was started with IVR_ENABLED=1");
    const health = await (await fetch(`${BASE}/api/health`)).json();
    check("health discloses exactly one audio route, and that it stores no audio",
      health.audioRoutes === 1 && health.ivrEnabled === true && health.audioStored === false, JSON.stringify(health));
    const scammed = await (await fetch(`${BASE}/api/ivr`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ SpeechResult: "I am CBI officer, transfer 2 lakh for verification, do not tell your family", From: "+910000000000", CallSid: "CA1" }),
    })).text();
    check("a scam transcript is answered with the warning line", /<Response>/.test(scammed) && /Warning\./.test(scammed) && /1 9 3 0/.test(scammed), scammed.slice(0, 160));
    const hostile = await (await fetch(`${BASE}/api/ivr?format=json`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: 'KYC band ho jayega </Say><Hangup/> <Say>pay now' }),
    })).json();
    check("the spoken line is our own text, never the caller's", typeof hostile.say === "string" && !/Hangup/.test(hostile.say), hostile.say.slice(0, 120));
    check("the JSON contract reports that nothing was stored", hostile.transcriptStored === false && hostile.audioStored === false);
  }

  step("console stayed clean");
  const bad = logs.filter((l) => /pageerror|Uncaught/i.test(l));
  check("no page errors", bad.length === 0, bad.slice(0, 3).join(" | "));
  const unexpected = http4xx.filter((u) => !/\/api\/v\//.test(u) && !/favicon/.test(u));
  check("no unexpected HTTP errors", unexpected.length === 0, unexpected.slice(0, 3).join(" | "));
} catch (err) {
  console.error(`\nE2E FAILED: ${err.message}`);
  console.error(logs.slice(-25).join("\n"));
  failures++;
} finally {
  await browser.close();
}

console.log(failures === 0 ? "\nE2E OK — the shipped inference path works end to end." : `\nE2E FAILED (${failures} assertion(s)).`);
process.exit(failures === 0 ? 0 : 1);
