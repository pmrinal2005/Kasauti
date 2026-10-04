# Train Laya on Kaggle → ship it to Kasauti: the exact steps

The complete runbook for producing a **calibrated, quantized, reference-checked Laya checkpoint**
and switching the Kasauti dashboard onto it. No application code changes: the browser runtime
already speaks exactly the artifact layout the pipeline publishes.

Two things this document is not:

* it is **not** required to run Kasauti. With no model published the app works fully on Tier-0 (the
  deterministic checker) and says so, on purpose.
* it is **not** a "later" task in the code. The trainer, the calibrator, the eval gates, the ONNX
  exporter, the INT8/4-bit quantizers, the vocabulary pruner, the parity vectors, the manifest, the
  HF publisher, the Worker loader, the Cache-API store, the abstention gate and the report-only
  switch are all built and tested. Training is the one step that needs a GPU, which is why it lives
  on Kaggle.

**What has been executed and verified in this repo** (everything except the GPU fine-tune itself):

| check | command | result |
|---|---|---|
| Python↔TS↔runtime unit tests | `npm test` | 68/68 pass |
| typecheck + production build | `npm run typecheck && npm run build` | clean, 7 routes |
| whole pipeline on CPU (tiny model) | `python -m ml.kasauti_ml.pipeline --profile smoke --out ml/out/dev --public-dir public/dev-model --reference-check` | exit 0, ~75 s |
| export vs **reference Laya client** | `python -m ml.kasauti_ml.reference_check --staging ml/out/dev/staging` | mirrors / calibration / reference client / evals all pass |
| mirrors vs installed `laya` | `python tests/test_reference_parity.py` (add `KASAUTI_TOKENIZER=<checkpoint>/multilingual/tokenizer/tokenizer.json` for the real 256k tokenizer) | 7/7 checks pass, both with the fixture and the real tokenizer |
| real browser, real ONNX graph | `node tests/e2e-devmode.mjs http://localhost:3000` | E2E OK (worker → download → sha256 → TS tokenizer → WASM → decode) |
| parity self-test **on the device** | same run, *Self-test on this device* row | `replayed 6 logits + 10 tokenizer vectors · max \|Δlogit\| 0.0473 ≤ 0.35 · top-1 100%` |
| real 256 000-token tokenizer pruning | `export.prune_tokenizer(...)` on `multilingual/tokenizer.json` | 256 000 → 6 647 tokens, **0% segmentation drift**, 6.8 s |
| notebook itself | executed end-to-end in `SCOPE="smoke"` | 26/26 cells, zero errors |

---

## 0. What you are training, in one paragraph

Laya is a **non-generative typed-decision** model: you hand it a state (the message), a typed
question and its options, and it answers in **one forward pass** by scoring each option at its own
`[MASK]` marker. It never writes text. Upstream ships near-chance zero-shot on typed decisions,
over-confident (ECE ≈ 0.31 for multilingual) and with an `act_probability` field that carries no
usable signal (their issue #185) — so Kasauti fine-tunes it with RLCD, refits a temperature per
`(type, option-count)` bucket, and gates every answer on **calibrated confidence**, never on
`act_probability`. You are shipping `laya-multilingual` (mmBERT-base, 322 M params, 1 024 context
by default, 8 192 available, 100+ languages) plus a new decision head.

---

## 1. Prerequisites (5 minutes, one time)

| Need | Where | Notes |
|---|---|---|
| Kaggle account, phone-verified | kaggle.com | accelerators stay hidden until the phone is verified |
| (recommended) a fork of this repo | fork `pmrinal2005/Kasauti` | so the notebook clones *your* branch |
| (optional) a Hugging Face account | huggingface.co | only for the publish path; model repos are free and stay free |
| (optional) an HF **write** token | HF → Settings → Access Tokens → *Write* | store it as a Kaggle **Secret** named `HF_TOKEN`, never in a cell |

Kaggle quotas to plan around: a **12 h** accelerator-session cap and a weekly GPU allowance
(*Settings → Accelerator*). The default run needs roughly **20–40 min** wall clock on **T4 ×2**
(most of it the fine-tune); the notebook streams progress so you can see where you are.

---

## 2. Put the code where Kaggle can see it

Pick **one**. Option A unless you have a reason not to.

### Option A — clone from GitHub (Internet ON)

1. Kaggle → **Create → New Notebook**.
2. Sidebar → **Settings**:
   * *Accelerator* → **GPU T4 ×2**
   * *Internet* → **On** (the clone, the checkpoint download and the publish all need it)
   * *Persistence* → *Files only* (default; `/kaggle/working` survives the session)
3. **File → Import Notebook** → upload `notebooks/kasauti_laya_finetune_kaggle.ipynb`.
4. **Step 1** of the notebook: set `REPO_URL` to your fork if you have one.

### Option B — attach the repo as a Dataset (no clone)

Zip the repo (including `ml/` and `notebooks/`, excluding `node_modules/` and `.next/`), upload as a
Kaggle Dataset, *Add Input → Your Datasets* into the notebook, and in **Step 1** set
`PROJECT_SRC = "/kaggle/input/<dataset-slug>"`. Internet must still be **On** for the checkpoint
download.

---

## 3. Run the notebook, top to bottom

The notebook is generated from `ml/scripts/gen_notebook.py` (`python -m ml.scripts.gen_notebook
--check` fails if it is stale), so this table and the notebook cannot drift apart.

| step | what it does | expected |
|---|---|---|
| 0 — install | `pip install "laya[onnx]>=0.3.26" …`, then prints the versions that resolved | ~1 min |
| 1 — code + preflight | clone/mount, then print GPU count, RAM, free disk | 2× T4, ≥ 12 GB RAM |
| 2 — config | every knob in one cell (`SCOPE`, `VARIANTS`, `EPOCHS`, `TASKS`, `BASE_REVISION`, `PUBLISH_REPO`, `PUBLIC_DIR`) | instant |
| 3 — provenance | recompute `sha256(ml/banks.json)` and compare with `src/lib/banks.ts` | must print *in sync* |
| 4 — data | build the multi-task dataset, print split/language/template stats | ~10 000 rows at `VARIANTS=40`, **zero template overlap** |
| 5 — CPU smoke | the whole chain on a tiny random model, **including the reference check** | ~2 min, exit 0, gates advisory |
| 6 — fine-tune | `torchrun --nproc_per_node=2 -m ml.kasauti_ml.pipeline --profile kaggle … --reference-check` | the long one — see §4 |
| 7 — reports | `eval_report.json`, `calibration.json`, `summary.json` (incl. base-checkpoint provenance) | read the gate table first |
| 8 — independent verify | re-hash every published byte, replay the browser's parity vectors through both quantized graphs, re-check the graph I/O | every row `ok`; prints `ALL EXPORT CHECKS PASSED` only when the export cleared its gates |
| 9 — reference cross-check | `tests/test_reference_parity.py` + the pipeline's `reference_check.json` | `7/7 checks passed`, and `PASS` for mirrors · calibration · reference client (77 = `laya` not installed → `%pip install "laya[onnx]"`) |
| 10 — artifacts | copy `staging/` to `out-export/`, optionally push to a HF **model** repo | ~1 min |
| 11 — integration | the exact Vercel/browser wiring (also §6 below) | — |

**Always run step 5 first.** Two minutes, and it proves the dataset, tokenizer, trainer, calibrator,
exporter, quantizers, manifest and reference check all work *before* you spend GPU hours.

### The one cell that matters (step 6, default)

```bash
torchrun --standalone --nproc_per_node=2 -m ml.kasauti_ml.pipeline \
  --profile kaggle --out /kaggle/working/out \
  --model-id convaiinnovations/laya --subfolder multilingual \
  --variants 40 --epochs 3 \
  --micro-batch 8 --grad-accum 4 --group-size 4 \
  --lr-encoder 2.5e-5 --lr-head 1e-4 \
  --max-len 1024 --head-max-len 256 \
  --abstain-target-error 0.15 --log-every 25 \
  --reference-check
```

| flag | meaning | when to change it |
|---|---|---|
| `--variants 40` | surface variants per template | **the main coverage knob.** More data beats more epochs |
| `--epochs 3` | RLCD passes over the train split | raise only if accuracy is short (5–6); watch eval ECE rise after that |
| `--micro-batch` × `--grad-accum` | per-GPU microbatch × accumulation (8 × 4 = 32 effective, ×2 GPUs) | halve the microbatch, double the accumulation on CUDA OOM |
| `--group-size 4` | RLCD noise samples per item (the GRPO group) | 2 saves memory, 8 is smoother but slower |
| `--max-len` / `--head-max-len` | state budget / option-prompt budget | `--max-len 8192` for long transcripts (≈8× cost, ≤4 000 tokens measured-reliable) |
| `--abstain-target-error 0.15` | error rate allowed **among accepted answers** | 0.25 when too much abstains; 0.10 when you want only confident answers |
| `--reference-check` | run the export through the reference `ONNXAgent` before publishing | leave on. It is the only independent check of the artifact |
| `--revision <sha>` | pin the base checkpoint | it is already pinned to Laya's reviewed revision; set this only to reproduce an older export |

### The base checkpoint is pinned, and only the files you need are downloaded

`convaiinnovations/laya` is a live repository that bundles three checkpoints (~2.5 GB). The pipeline
resolves the revision through `laya.revisions.PINNED_REVISIONS` (commit
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` at the time of writing), downloads **only**
`multilingual/*` (~680 MB), verifies digests when `LAYA_SHA256_DIGESTS` is set, and records all of
it in `summary.json.base` and in the manifest:

```json
{"modelId": "convaiinnovations/laya", "subfolder": "multilingual",
 "revisionRequested": "55cf4c4…", "revisionResolved": "55cf4c4…", "pinned": true,
 "onlySubfolderFiles": true}
```

If two exports of "the same" model ever disagree, this field is the first thing to look at. Use
`--no-pin-base` deliberately (and expect a non-reproducible export) or `--all-files` to fetch the
whole repo.

---

## 4. What actually happens during the fine-tune

1. **Data**: `ml/banks.json` → template expansion → split **by template hash** (a template seen in
   training can never appear in val/test) → soft gold distributions, hard negatives, romanised
   code-mixing, option-order permutations.
2. **RLCD** (the reference loop): sample `GROUP_SIZE` noisy distributions per item with zero-mean
   projected Gaussian noise on the logits; score each with the strictly proper rule (log + spherical,
   plus a ranked probability score on `score` questions); REINFORCE against a group-mean baseline,
   with a soft cross-entropy anchor, rare-class re-weighting and a negative-class boost. Encoder and
   head have separate learning rates; the label-smoothed CE term keeps it from collapsing.
3. **Calibration**: one temperature per `(question type, option-count)` bucket, fitted on a held-out
   slice against `answer_confidence`; then an abstention map — the smallest cut per bucket that keeps
   accepted-answer error under `--abstain-target-error`.
4. **Gates**: accuracy, per-language floor, ECE (overall + worst bucket), `noul`/`choice` accuracy,
   pay-to-withdraw precision/recall, impersonation recall, negative FPR, abstention coverage,
   quantized agreement, tokenizer parity. **Blocking** on the `kaggle` profile: a failed gate exits
   `2` and publishes nothing.
5. **Export**: vocabulary pruning (complete merge closure — see §7) → embedding-row gather → dynamic
   fp32 ONNX → INT8 per-channel (WASM path) → 4-bit `MatMulNBits` (WebGPU path) → ORT agreement
   re-check → parity vectors → manifest → staging with sha256 verification.
6. **Reference check** (`--reference-check`): `laya.onnx_agent.ONNXAgent` is pointed at a scratch
   checkpoint dir holding *our* pruned tokenizer, *our* fitted temperatures and *our* exported graph;
   its answers are compared with our own decode. A failure exits `4` and publishes nothing, unless
   you explicitly pass `--allow-reference-check-skip`.

Exit codes: **2** gates failed · **3** INT8 drifted from fp32 · **4** reference client disagreed ·
**5** tokenizer drift. All four mean *nothing was published*.

---

## 5. The tokenizer is part of the model

The browser tokenizes with a from-scratch TypeScript port of the Rust `tokenizers` library, so the
published `tokenizer.json` must be a **pruned but faithful** copy: same ids, same merge ranks, same
segmentation. Two failure modes that cost real accuracy and produce no error:

* **Merge closure.** A BPE merge list is a multimap — in the multilingual tokenizer 177 155 of
  234 865 merge results have more than one producer (up to 30). Keeping only the first producer's
  parts made the pruned tokenizer re-segment **16.5 % of a 10 400-text corpus** differently from the
  tokenizer the model was trained with. `ml/kasauti_ml/export.py` now walks *every* producer of every
  kept token (verified: 0 % drift, 256 000 → 6 647 tokens, 6.8 s), and the pipeline **fails the
  export (exit 5)** above `--max-tokenizer-drift` (default 0.5 %) instead of only warning.
* **Test fixture.** `tests/fixtures/laya-tokenizer.mini.json` had the same problem plus a custom
  top-level key that the Rust parser rejects with a misleading *"expected `,` or `}`"*. It is now
  strictly upstream-shaped, merge-closed, and self-checked: the generator re-segments every parity
  text with the fixture it is about to write and refuses to write one that does not reproduce the
  reference ids exactly. Regenerate with:

  ```bash
  python -m ml.scripts.make_tokenizer_fixture --repo convaiinnovations/laya \
      --subfolder multilingual/tokenizer          # add --check in CI
  ```

---

## 6. Wiring the export into Kasauti (step 11, step by step)

**1. Publish.** Either keep `out-export/` as notebook output, or push it to a HF **model** repo
(step 10). A model repo is static LFS storage on the Hub CDN: no Space, no ZeroGPU quota, nothing to
keep warm, nothing that can be billed. Upload the **whole directory** — `manifest.json`, both
graphs, the pruned tokenizer, `parity.json`, `reference_check.json`.

**2. Point the build at it.** The value is inlined at build time, so it must exist *before* the
build. Vercel → Project → Settings → Environment Variables:

```
NEXT_PUBLIC_LAYA_MODEL_BASE = https://huggingface.co/<user>/kasauti-laya-multilingual-onnx/resolve/main
```

Leave `NEXT_PUBLIC_ORT_BASE` unset to use the pinned `onnxruntime-web@1.23.0` CDN default (or
self-host a copy of that `dist/` and point the variable at it). Then redeploy.

**3. Verify locally.**

```bash
npm run typecheck && npm test && npm run build
NEXT_PUBLIC_LAYA_MODEL_BASE=$PWD/public/dev-model npm start     # or npm run dev
node tests/e2e-devmode.mjs http://localhost:3000               # Worker + download + decode
```

**4. Verify in the browser** — `/dashboard` → *On-Device Engine*:

* **model version** reads `kasauti-laya-multilingual@<sha8>` (your export, not `tiny-dev`);
* **backend** is `webgpu`, or `wasm-st` / `wasm-mt` where WebGPU is absent (the page names it);
* **manifest** shows your fitted temperatures and abstention cuts; *Artefacts verified* names the
  file the browser actually downloaded and hashed; **Self-test on this device** reports the
  Worker's own replay of your parity vectors — if that row ever says `FAILED`, the export was
  demoted to report-only on the spot and the page will not act on it;
* **Eval gates** lists your numbers against the thresholds — every row green;
* **question-bank drift** is empty (manifest `banks.sha256` equals the compiled-in `BANKS_SHA256`);
* press **Build sequence**: sequence length and `[MASK]` marker count must match training.

**5. Check gate behaviour.** Run a known scam through the Checker. A confident answer acts; an
unsettled one shows *abstained* and the tier line says the verdict came from Tier-0. If *everything*
abstains, either the export is `reportOnly` (see §3 step 8) or the thresholds are too aggressive —
raise `--abstain-target-error` and re-run step 6.

**6. Transcripts.** `/dashboard#lens` → *Transcript / call notes*: a pasted transcript is cut into
overlapping windows, read per window, aggregated, with the dropped-window count shown. No retrain.

**7. Offline.** The first load caches graphs + tokenizer in the Cache API; afterwards checks work
offline. `Save-Data` / 2G defers the download by design, and the Engine page says so.

---

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `CUDA out of memory` (step 6) | microbatch × seq-len 1024 | halve `MICRO_BATCH`, double `GRAD_ACCUM` |
| exit `2` | eval gates failed | more `VARIANTS`/`EPOCHS`, or narrower `TASKS` |
| exit `3` | INT8 top-1 agreement < 0.95 | re-export with `--block-size 64`, keep `--quantize-embedding` |
| exit `4` | reference client ≠ our export | open `reference_check.json`; failures name the question and both answers |
| exit `5` | tokenizer drift > 0.5 % | widen the corpus (more `--variants`); `--allow-tokenizer-drift` only knowingly |
| step 9 reports the check **skipped** | `laya` or its `onnx` extra missing | `%pip install "laya[onnx]>=0.3.26"` |
| browser: *no calibrated export* | manifest 404 | wrong `NEXT_PUBLIC_LAYA_MODEL_BASE`, or the repo is private |
| browser: *report-only* | export did not clear the gates, or was built with `--profile smoke` | read the *Eval gates* panel — the manifest is telling the truth, not failing |
| browser: *bank drift* | trained on different wording | regenerate `src/lib/banks.ts` from the same `ml/banks.json`, retrain |
| everything abstains | thresholds fitted to a hard split | re-run with `--abstain-target-error 0.25` |
| tokenizer mismatch | graph and tokenizer from different runs | always ship the whole `staging/` directory together |

---

## 8. Honest limits (read before tuning)

* **Base checkpoints are near chance on typed decisions zero-shot.** The accuracy comes from this
  fine-tune and from the banks; that is why the gates exist.
* **The model ships over-confident.** Per-(type, option-count) temperature fitting is what makes the
  confidence numbers mean anything — never ship an export without calibration.
* **`choice` questions degrade past ~20 options** (options share a fixed `head_max_len` budget). Keep
  Kasauti's banks well below that, or raise `--head-max-len`.
* **`noul` can latch onto its own labels** on the English checkpoint; the multilingual checkpoint is
  much less prone, and the banks phrase `noul` as single yes/no statements for exactly this reason.
  Always check `noul` answers on your own data.
* **`score` is the weakest primitive.** Used only where an ordinal answer is genuinely needed.
* **Long documents**: measured reliability degrades past ~4 000 tokens even with `--max-len 8192`, so
  Kasauti windows long transcripts instead of sending them whole.
* **`act_probability` carries no usable signal** upstream; Kasauti gates on `answer_confidence`
  only. The browser never reads `act_logits`.

---

## 9. Adding coverage later

`ml/banks.json` is where coverage lives: add surface variants and new templates, not new *kinds* of
question. Keep questions flat (one typed question at a time — multi-label questions must be
decomposed into independent `noul` calls) and keep option counts modest.

Adding a bank means adding its templates to `ml/kasauti_ml/datagen.py` (`TEMPLATES_BY_TASK`), then
re-running the notebook. `python -m ml.scripts.gen_banks_ts` regenerates the browser mirror, the
manifest sha256 changes with it, and that is how the runtime notices drift.

## 10. Beyond the hackathon

* Publish the model card with the base revision, the banks sha256, the eval table and the reference
  check result — everything a reviewer needs to reproduce the numbers.
* Open the question banks and the gold evaluation sets as public-good datasets, alongside the ONNX
  weights themselves.
* Contribute anonymised scam-pattern clusters to official channels; the telephony/IVR channel is the
  one place that needs institutional backing (a real number), and it is documented as the single,
  bounded audio-handling exception in `docs/privacy-notice.md`.
