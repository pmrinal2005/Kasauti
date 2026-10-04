# Kasauti — Investor-Protection Dashboard

> *Kasauti* (कसौटी) = touchstone. A fully client-side scam checker for first-time Indian investors:
> **deterministic first, Laya second, LLM last** — and **zero audio bytes ever reach our servers**
> (the sponsored phone line is the one documented exception: its provider transcribes the call, and
> the route is off unless `IVR_ENABLED=1`).

**Stack:** Next.js 16 (App Router, `--webpack`) · React 19 · TypeScript strict · hand-written CSS + SVG/Canvas charts (no UI or chart libraries) · Web Workers · `onnxruntime-web` (CDN, inside the inference Worker only) and a **dependency-free pure-TS tokenizer** verified id-for-id against Rust `tokenizers`.
**Deploy target:** Vercel Hobby (free tier), region `bom1`. No Cloudflare, no Hono.

## What's built so far

| Area | Status | Where |
|---|---|---|
| Tier-0 core: NFKC + zero-width + homoglyph + Indic-digit normalisation, URL canonicalisation | ✅ | `src/core/normalize.ts` |
| SHA-256 + 64-bit SimHash fingerprints | ✅ | `src/core/fingerprint.ts` |
| Aho–Corasick multilingual lexicon (EN / Hinglish / हिन्दी / मराठी / தமிழ்), 10 families | ✅ | `src/core/lexicon.ts` |
| Entity checks: UPI handle vs PSP list, TRAI 1600-series, lakh/crore amounts, return claims, SEBI INA/INH numbers | ✅ | `src/core/entities.ts` |
| Plausibility maths: compounded annualisation, bands, doubling time, Ponzi-collapse month | ✅ | `src/core/plausibility.ts` |
| Evidence fusion (Tier-0) + **Tier-1 fusion rule table** (`fuseTier1`) | ✅ | `src/core/fusion.ts` |
| Laya browser contract: port of `rl_common.build_sequence`, **manifest v2** calibrated decode (per-bucket temperature, abstention, option un-permutation) | ✅ | `src/core/laya-client-browser.ts` |
| Question banks: one generated source of truth for trainer + runtime (`ml/banks.json` → `src/lib/banks.ts`), with a banks-sha drift check in the manifest | ✅ | `src/lib/banks.ts` |
| Pure-TS tokenizer (Metaspace + BPE + `byte_fallback` + added tokens), **id-exact vs Rust `tokenizers`** on 44 texts / 1 165 ids from the real multilingual checkpoint | ✅ | `src/core/tokenizer.ts`, `tests/tokenizer-parity.test.ts` |
| Tier-0 Worker; inference Worker (Cache API `kasauti-model-v2`, sha256 verification of every artifact, WebGPU/WASM EP, circuit breaker, Save-Data deferral, micro-benchmark, tokenizer dry-run) | ✅ | `src/workers/*`, `src/lib/engine.ts` |
| ML pipeline: banks → data → RLCD training (real DDP) → per-bucket calibration → blocking eval gates → vocab pruning → INT8 + Q4 ONNX → ORT agreement → parity vectors → manifest → staged publish | ✅ | `ml/` |
| Capability gating (WebGPU, COI, cores, `deviceMemory`, Save-Data, in-app browser) → backend + question budget | ✅ | `src/lib/capability.ts` |
| Browser-native voice: `SpeechRecognition` (`processLocally` first, auto-restart, hard timeout) + `speechSynthesis` fallback chain | ✅ | `src/core/voice.ts` |
| Dashboard: collapsible sidebar, header (⌘K search, language, theme, profile), KPI cards with sparklines, line/bar/donut charts, sortable/filterable/paginated table + CSV export | ✅ | `src/components/*` |
| Views: Overview · Message Checker · **Content Lens (promotion vs education)** · Voice Explainer · Registry & Calculators (crash simulator on Canvas) · On-Device Engine · Privacy Ledger | ✅ | `src/components/views/*` |
| Local on-device verdict cache (data-flow step 4) | ✅ | `src/lib/engine.ts` |
| API: shared verdict cache, health, text-only "Why?" (Groq with fallback chain + advice guardrail) | ✅ | `app/api/*` |
| Security/COI headers, PWA manifest, hand-rolled service worker | ✅ | `next.config.mjs`, `public/` |
| Tests: 68 on the Node built-in runner (core, tokenizer parity, manifest/calibration/gating, URL resolution, transcript windowing, IVR spoken contract) | ✅ | `tests/*.test.ts` |
| Browser E2E against a real quantized ONNX graph (worker loads, hashes, tokenizes, infers, decodes, abstains) | ✅ | `tests/e2e-devmode.mjs` |

## Entry points

| Path | Method | Purpose |
|---|---|---|
| `/` | GET | **Landing page** — server-rendered, no client JS: the pitch, the exact privacy claim, the evidence ladder, and the post-incident path (Call 1930 / cybercrime.gov.in) |
| `/dashboard#overview \| #checker \| #lens \| #voice \| #registry \| #engine \| #privacy` | GET | Dashboard views (hash-routed, works offline) |
| `/api/v/{sha256}` | GET | Shared verdict lookup. 404 = miss. CDN cacheable (`s-maxage=300`) |
| `/api/v/{sha256}` | POST | `{simhash, rung, codes[], modelVersion}`: hash + codes only, validated, rate-limited per IP |
| `/api/explain` | POST | `{text, codes[], lang}` → streamed plain-text explanation. **The only route that sends your message text to a third party (Groq), and only when you tap “Why?”**; without a key it answers from the evidence codes alone. Text only, never audio |
| `/api/health` | GET | `{store, groq, audioRoutes}` — `0` unless the IVR adapter is switched on |
| `/api/ivr` | POST | **Telephony adapter, off unless `IVR_ENABLED=1`.** Provider (Twilio/Exotel) transcribes; we run the same deterministic Tier-0 core and answer with a fixed spoken line (TwiML XML, or JSON on request). No audio is received or stored, no transcript is persisted, no language model is on the phone |

There is deliberately **no `/api/asr`**. `/api/ivr` is the *only* route that can touch speech, it is disabled by default, and it says so on `/api/health`.

## Data architecture

- **On device:** check history (`kasauti.checks.v1`), the hash → fused-verdict cache (`kasauti.verdicts.v1`), and the language and theme settings, all in localStorage. Model and tokenizer binaries go in the Cache API (`kasauti-model-v2`), keyed by URL and re-hash-verified on every load.
- **Server (optional):** Supabase table `verdicts` (`supabase/schema.sql`) holding the hash, SimHash, rung, evidence codes, model version and hit count, with **no raw text**. If Supabase isn't configured, it falls back to an in-memory LRU on each server instance.
- **Model delivery:** a public Hugging Face **model** repo (no Space, no compute, no Kasauti-operated inference anywhere). `ml/kasauti_ml/pipeline.py` produces the whole layout; `ml/README.md` has the step-by-step. **Until a calibrated export is published, the engine reports "unpublished" and the app runs on Tier-0 alone** — that is the circuit breaker, and it is what a fresh clone does. A tiny dev export ships in `public/dev-model/` for offline development and the E2E test (`NEXT_PUBLIC_LAYA_MODEL_BASE=/dev-model`); it is a *plumbing* model whose answers are deliberately report-only.

### Published export layout (manifest schema 2, `NEXT_PUBLIC_LAYA_MODEL_BASE`)
```jsonc
// <base>/manifest.json
{ "schema": 2, "modelVersion": "kasauti-laya-multilingual@<sha8>", "maxLen": 1024, "headMaxLen": 256,
  "files": { "wasm":   { "path": "kasauti-laya-int8.<sha8>.onnx", "bytes": 0, "sha256": "…" },
             "webgpu": { "path": "kasauti-laya-q4.<sha8>.onnx",   "bytes": 0, "sha256": "…" },
             "fp32":   { "path": "kasauti-laya-fp32.<sha8>.onnx", "bytes": 0, "sha256": "…" } },
  "temperature": [1.0, 1.0, 1.0], "temperatureByBucket": { "choice:3-5": 1.2, "noul:2": 1.4 },
  "abstain": { "noul:2": 0.35, "default": 0.35 },
  "tokenizer": { "path": "tokenizer.json", "sha256": "…", "vocab": 0, "pruned": true },
  "parity": { "path": "parity.json", "sha256": "…" },
  "banks": { "version": "1.0.0", "sha256": "…" },
  "evals": { "accuracy": 0.0, "ece": 0.0, "int8Agreement": 0.0 } }
```
Graph I/O matches `DecisionModel.forward`: inputs `input_ids, attention_mask` (int64 `[B,L]`), `marker_pos` (int64 `[B,K]`), `marker_mask` (bool `[B,K]`), `qtype` (int64 `[B]`) → outputs `logits` `[B,K]` (+ `act_logits`, never gated on). Every artifact's `sha256` is verified before use and again on every cache hit, and once the session is up the Worker **replays the published parity vectors on the device** — a graph that drifts past the manifest's tolerance is demoted to report-only rather than quietly answering; a mismatch between the manifest's `banks.sha256` and the compiled-in `BANKS_SHA256` is shown in the UI as *question-bank drift*.

## Laya facts the design depends on (from the model card and `rl_common.py`)
- It doesn't generate text. It answers `choice` / `score` / `noul` questions, scoring each option at its own `[MASK]` marker. In `laya-multilingual`, CLS is `<bos>`=2, SEP is `<eos>`=1 and MASK is `<mask>`=4.
- It's over-confident out of the box, so the plan is one temperature per `(type, option-count)` bucket: published ECE drops from 0.314 to 0.106 on multilingual. Decisions gate on `confidence`, **never** `act_probability` (issue #185).
- Accuracy falls apart above about 20 options (Banking77), so the triage question uses 5 options and follow-ups are `noul`.
- The base checkpoints are close to chance on typed decisions until fine-tuned, so production requires the Kaggle fine-tune.

## Run locally
```bash
npm install
npm test                      # 68 tests: core, tokenizer parity, manifest/calibration/gating
npm run typecheck
npm run build && npm start    # http://localhost:3000/dashboard

# ML side (CPU only, ~75 s — validates the whole export chain on a tiny model)
python -m ml.scripts.gen_banks_ts                  # regenerate src/lib/banks.ts from ml/banks.json
python -m ml.kasauti_ml.pipeline --profile smoke --out ml/out/dev --public-dir public/dev-model --reference-check
python tests/test_reference_parity.py              # our mirrors vs the installed `laya` package
                                                   # add KASAUTI_TOKENIZER=<real tokenizer.json> for the 256k one

# browser E2E against that dev model (real ONNX inference in a Worker)
NEXT_PUBLIC_LAYA_MODEL_BASE=/dev-model npm run dev
node tests/e2e-devmode.mjs http://localhost:3000

# …and the telephony route, only if the server was started with IVR_ENABLED=1
IVR_ENABLED=1 NEXT_PUBLIC_LAYA_MODEL_BASE=/dev-model npm run dev
E2E_IVR=1 node tests/e2e-devmode.mjs http://localhost:3000
```

## Deploy (Vercel, manual)
1. Import the repo into Vercel. The framework is detected as Next.js and `vercel.json` pins the `bom1` region.
2. Optional env vars (see `env.example`): `NEXT_PUBLIC_LAYA_MODEL_BASE`, `GROQ_API_KEY`, `GROQ_MODELS`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`. **Every variable is optional**; with none set the app still works fully on Tier-0.
3. If you're using Supabase, run `supabase/schema.sql` once.

## Not built yet / next steps
1. **Run the GPU fine-tune** — `notebooks/kasauti_laya_finetune_kaggle.ipynb` (2×T4), then set
   `NEXT_PUBLIC_LAYA_MODEL_BASE` to the published model repo. **Exact runbook:
   [`docs/kaggle-training-and-integration.md`](docs/kaggle-training-and-integration.md)** — Kaggle
   setup, the flags worth changing, the gate table, how to get the artifacts out, Vercel wiring,
   verification and rollback. The pipeline, client and gating are finished and tested against a dev
   model; what is missing is a trained checkpoint — the base checkpoints are near chance on typed
   decisions until fine-tuned.
2. Run the real-device benchmark matrix (Android Go, Safari iOS, WhatsApp and Telegram webviews) and tune the question-budget profiles.
3. Long-transcript windowing is in (`src/lib/transcript.ts` + Content Lens *Transcript* mode: speaker turns/paragraphs/sentences as cut points, overlapping windows, dropped-window count shown). What is still missing is a **published** `lens_v1` — the windows are read by the same export as everything else.
4. Pre-render IndicF5 clips for about 300 static concepts (fallback step 2 in the synthesizer chain).
5. Ship the full SEBI IA/RA registry snapshot (Brotli, content-hashed). Right now there's a 5-entry **fictional** demo snapshot.
6. Suraksha Mandali circles, Raksha Contact, community-report triage, district digests.
7. The telephony/IVR adapter exists (`/api/ivr`, off by default) but is **unverified against a real
   provider** — it has unit tests for the spoken contract and XML escaping, not a live call. Wire it
   to an Exotel/Twilio number, then re-check the copy in `docs/privacy-notice.md`.
8. ~~The landing page.~~ Built — see `/`.

**About the data:** the network charts, KPIs and the "network" rows in the table come from a seeded PRNG and are illustrative only. Only rows tagged "you" are real checks from this device.

## Docs
| Doc | What it covers |
|---|---|
| [`docs/kaggle-training-and-integration.md`](docs/kaggle-training-and-integration.md) | train → publish → integrate, start to finish |
| [`docs/privacy-notice.md`](docs/privacy-notice.md) | exactly what leaves the device, including the one IVR exception |
| [`docs/screenshots/`](docs/screenshots) | landing page and the Content Lens, as rendered |
| [`ml/README.md`](ml/README.md) | the 12 ML modules, the smoke run, regeneration commands |

## Privacy
See [`docs/privacy-notice.md`](docs/privacy-notice.md) and the in-app Privacy Ledger.

---
Kasauti never recommends a stock, fund, broker or trade. Fraud? Call **1930** · **cybercrime.gov.in**
