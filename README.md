# Kasauti — Investor-Protection Dashboard

> *Kasauti* (कसौटी) = touchstone. A fully client-side scam checker for first-time Indian investors:
> **deterministic first, Laya second, LLM last** — and **zero audio bytes ever reach our servers.**

**Stack:** Next.js 16 (App Router, `--webpack`) · React 19 · TypeScript strict · hand-written CSS + SVG/Canvas charts (no UI or chart libraries) · Web Workers · `onnxruntime-web` and the `transformers.js` tokenizer, both loaded from a CDN only inside the inference Worker.
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
| Laya browser contract: port of `rl_common.build_sequence`, calibrated decode, abstention, adaptive question tree | ✅ | `src/core/laya-client-browser.ts` |
| Tokenizer parity patch (transformers.js vs Rust `tokenizers`) — verified byte-identical | ✅ | `patchTokenizerJSON()` |
| Tier-0 Worker; inference Worker (Cache API, WebGPU/WASM EP, circuit breaker, micro-benchmark, tokenizer dry-run) | ✅ | `src/workers/*` |
| Capability gating (WebGPU, COI, cores, `deviceMemory`, Save-Data, in-app browser) → backend + question budget | ✅ | `src/lib/capability.ts` |
| Browser-native voice: `SpeechRecognition` (`processLocally` first, auto-restart, hard timeout) + `speechSynthesis` fallback chain | ✅ | `src/core/voice.ts` |
| Dashboard: collapsible sidebar, header (⌘K search, language, theme, profile), KPI cards with sparklines, line/bar/donut charts, sortable/filterable/paginated table + CSV export | ✅ | `src/components/*` |
| Views: Overview · Message Checker · Voice Explainer · Registry & Calculators (crash simulator on Canvas) · On-Device Engine · Privacy Ledger | ✅ | `src/components/views/*` |
| Local on-device verdict cache (data-flow step 4) | ✅ | `src/lib/engine.ts` |
| API: shared verdict cache, health, text-only "Why?" (Groq with fallback chain + advice guardrail) | ✅ | `app/api/*` |
| Security/COI headers, PWA manifest, hand-rolled service worker | ✅ | `next.config.mjs`, `public/` |
| Unit tests (18, Node built-in runner, no extra deps) | ✅ | `tests/core.test.ts` |

## Entry points

| Path | Method | Purpose |
|---|---|---|
| `/` | GET | Redirects to `/dashboard` (the landing page comes later) |
| `/dashboard#overview \| #checker \| #voice \| #registry \| #engine \| #privacy` | GET | Dashboard views (hash-routed, works offline) |
| `/api/v/{sha256}` | GET | Shared verdict lookup. 404 = miss. CDN cacheable (`s-maxage=300`) |
| `/api/v/{sha256}` | POST | `{simhash, rung, codes[], modelVersion}`: hash + codes only, validated, rate-limited per IP |
| `/api/explain` | POST | `{text, codes[], lang}` → streamed plain-text explanation (text only, never audio) |
| `/api/health` | GET | `{store, groq, audioRoutes: 0}` |

There is deliberately **no `/api/asr`**.

## Data architecture

- **On device:** check history (`kasauti.checks.v1`), the hash → fused-verdict cache (`kasauti.verdicts.v1`), and the language and theme settings, all in localStorage. Model and tokenizer binaries go in the Cache API (`kasauti-model-v1`), keyed by URL.
- **Server (optional):** Supabase table `verdicts` (`supabase/schema.sql`) holding the hash, SimHash, rung, evidence codes, model version and hit count, with **no raw text**. If Supabase isn't configured, it falls back to an in-memory LRU on each server instance.
- **Model delivery:** a public Hugging Face **model** repo (no Space, no compute). It has to contain `manifest.json` plus the quantized graphs (see below). The tokenizer is loaded straight from `convaiinnovations/laya/multilingual/tokenizer`.

### Expected ONNX export layout (`NEXT_PUBLIC_LAYA_MODEL_BASE`)
```json
// <base>/manifest.json
{ "modelVersion": "laya-multilingual-kasauti@<hash>", "maxLen": 1024, "headMaxLen": 256,
  "files": { "wasm": "model.int8.<hash>.onnx", "webgpu": "model.q4.<hash>.onnx" },
  "temperatures": { "noul:2": 1.4, "choice:3-5": 1.2 }, "abstain": { "noul:2": 0.35 } }
```
The graph I/O has to match `DecisionModel.forward`: inputs `input_ids, attention_mask` (int64 [B,L]), `marker_pos` (int64 [B,K]), `marker_mask` (bool [B,K]) and `qtype` (int64 [B]); output `logits` [B,K]. **Until that manifest exists, the engine reports "unpublished" and the app runs on Tier-0 alone** (that's the circuit breaker). The tokenizer dry-run on the Engine page still works.

## Laya facts the design depends on (from the model card and `rl_common.py`)
- It doesn't generate text. It answers `choice` / `score` / `noul` questions, scoring each option at its own `[MASK]` marker. In `laya-multilingual`, CLS is `<bos>`=2, SEP is `<eos>`=1 and MASK is `<mask>`=4.
- It's over-confident out of the box, so the plan is one temperature per `(type, option-count)` bucket: published ECE drops from 0.314 to 0.106 on multilingual. Decisions gate on `confidence`, **never** `act_probability` (issue #185).
- Accuracy falls apart above about 20 options (Banking77), so the triage question uses 5 options and follow-ups are `noul`.
- The base checkpoints are close to chance on typed decisions until fine-tuned, so production requires the Kaggle fine-tune.

## Run locally
```bash
npm install
npm test          # 18 unit tests for core modules
npm run typecheck
npm run build && npm start    # http://localhost:3000/dashboard
```

## Deploy (Vercel, manual)
1. Import the repo into Vercel. The framework is detected as Next.js and `vercel.json` pins the `bom1` region.
2. Optional env vars (see `env.example`): `NEXT_PUBLIC_LAYA_MODEL_BASE`, `GROQ_API_KEY`, `GROQ_MODELS`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`. **Every variable is optional**; with none set the app still works fully on Tier-0.
3. If you're using Supabase, run `supabase/schema.sql` once.

## Not built yet / next steps
1. **Fine-tune and export** `laya-multilingual` on Kaggle with the `claim_v1`, `lens_v1` and `intent_concepts_v1` banks, producing an INT8 WASM build and a 4-bit WebGPU build plus the calibration manifest, then publish to an HF model repo.
2. Run the real-device benchmark matrix (Android Go, Safari iOS, WhatsApp and Telegram webviews) and tune the question-budget profiles.
3. Add the Content Lens view (`lens_v1`, promotion vs education) and long-transcript windowing.
4. Pre-render IndicF5 clips for about 300 static concepts (fallback step 2 in the synthesizer chain).
5. Ship the full SEBI IA/RA registry snapshot (Brotli, content-hashed). Right now there's a 5-entry **fictional** demo snapshot.
6. Suraksha Mandali circles, Raksha Contact, community-report triage, district digests.
7. The telephony/IVR adapter: the one documented audio exception, still unbuilt.
8. The landing page.

**About the data:** the network charts, KPIs and the "network" rows in the table come from a seeded PRNG and are illustrative only. Only rows tagged "you" are real checks from this device.

## Privacy
See [`docs/privacy-notice.md`](docs/privacy-notice.md) and the in-app Privacy Ledger.

---
Kasauti never recommends a stock, fund, broker or trade. Fraud? Call **1930** · **cybercrime.gov.in**
