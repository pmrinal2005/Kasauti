# Train Laya on Kaggle → ship it to Kasauti: the exact steps

This is the complete, copy-pasteable runbook for producing a **calibrated, quantized Laya
checkpoint** and switching the Kasauti dashboard onto it. Nothing in the web app needs to change —
the runtime already speaks exactly the artifact layout the pipeline publishes.

Two things this document is not:

* it is **not** required to run Kasauti. With no model published the app works fully on Tier-0 (the
  deterministic checker) and says so, on purpose.
* it is **not** a "later" task in the code. The trainer, the calibrator, the eval gates, the ONNX
  exporter, the INT8/4-bit quantizers, the parity vectors, the manifest, the HF publisher, the
  Worker loader, the Cache-API store, the abstention gate and the report-only switch are all built
  and tested (`npm test` → 68 tests; `tests/e2e-devmode.mjs` runs a real ONNX graph in a real
  browser). Training is the one step that needs a GPU, which is why it lives on Kaggle.

---

## 0. What you are training, in one paragraph

Laya is a **non-generative typed-decision** model: you hand it a state (the message), a typed
question and its options, and it answers in **one forward pass** by scoring each option at its own
`[MASK]` marker. It never writes text. Upstream ships near-chance zero-shot on typed decisions
(0.362 on the typed-decisions benchmark), over-confident (ECE 0.314 for multilingual) and useless on
`act_probability` (issue #185) — so Kasauti fine-tunes it with RLCD, refits a temperature per
`(type, option-count)` bucket, and gates every answer on **calibrated confidence**, never on
`act_probability`. The thing you are shipping is `laya-multilingual` (mmBERT-base, 322 M params,
1024 context, 100+ languages) plus a new decision head: approximately 325 M trainable parameters.

---

## 1. Prerequisites (5 minutes, one time)

| Need | Where | Notes |
|---|---|---|
| Kaggle account with a phone-verified phone number | kaggle.com | Accelerators are hidden until the phone is verified |
| A GitHub fork of this repo | fork `pmrinal2005/Kasauti` | so the notebook clones *your* branch, not upstream |
| (optional) a Hugging Face account | huggingface.co | only for the publish path in §6; free, model repos are free |
| (optional) an HF **write** token | HF → Settings → Access Tokens → *Write* | store it as a Kaggle **Secret** named `HF_TOKEN`, never in a cell |

Kaggle quotas to plan around: the accelerator session cap is **12 h** and there is a weekly GPU
allowance (visible in *Settings → Accelerator*). The default run needs roughly **2–5 h** on
**T4 ×2**; the notebook streams progress so you can see where you are.

---

## 2. Put the code where Kaggle can see it

Pick **one** of these. Option A is the one to use unless you have a reason not to.

### Option A — clone from GitHub (Internet ON)

1. Kaggle → **Create → New Notebook**.
2. Right-hand sidebar → **Settings**:
   * *Accelerator* → **GPU T4 ×2**
   * *Internet* → **On** (required for `git clone`, the checkpoint download and the HF publish)
   * *Persistence* → *Files only* (default) — `/kaggle/working` survives the session
3. **File → Import Notebook** → upload `notebooks/kasauti_laya_finetune_kaggle.ipynb`.
4. In **Step 1** of the notebook set:

   ```python
   REPO_URL = "https://github.com/<your-user>/Kasauti.git"
   REPO_REF = "main"
   ```

   Leave the default if you have no fork — it clones upstream.

### Option B — upload the repo as a private Dataset (no Internet needed for the clone)

1. Zip the repo **including** `ml/` and `notebooks/` but excluding `node_modules/` and `.next/`.
2. Kaggle → **Datasets → New Dataset** → upload the zip → *Create*.
3. In the notebook: *Add Input → Your Datasets → the dataset*. Then in **Step 1** set
   `PROJECT_DIR` to the mounted path (`/kaggle/input/<dataset-slug>/Kasauti`) and skip the clone
   cell. You still need *Internet: On* to download the `convaiinnovations/laya` checkpoint.

---

## 3. Run the notebook, top to bottom

The notebook is generated from `ml/scripts/gen_notebook.py` and is the source of truth for the
commands; this section is the operator's view of it.

| Notebook step | What it does | Expected |
|---|---|---|
| 0 — install | `pip install -r ml/requirements.txt` (`laya`, `transformers`, `onnx`, `onnxruntime`, `onnx_ir`, `scikit-learn`) | ~1–2 min |
| 1 — code | clone/mount, `PROJECT_DIR`, `os.chdir` | instant |
| 2 — config | all knobs in one cell (`OUT_DIR`, `MODEL_ID`, `SUBFOLDER`, `VARIANTS`, `EPOCHS`, `MICRO_BATCH`, `GRAD_ACCUM`, `MAX_LEN`, `TASKS`, `PUBLIC_DIR`) | instant |
| 3 — provenance | recompute `sha256(ml/banks.json)` and compare with `src/lib/banks.ts` | must print **in sync**; if not, run `python -m ml.scripts.gen_banks_ts` and commit |
| 4 — data | build the multi-task dataset and print split/language/template stats | ~10 320 rows at `VARIANTS=40`, 6 languages, 43 templates, **zero template overlap** between splits |
| 5 — CPU smoke | the entire pipeline end-to-end on a tiny random model | ~20 s, gates advisory, **this is your dry run** |
| 6 — fine-tune | `torchrun --nproc_per_node=2 -m ml.kasauti_ml.pipeline --profile kaggle …` | the long one — see §4 |
| 7 — reports | `eval_report.json`, `train_report.json`, `calibration.json` | read the gate table before touching anything else |
| 8 — independent verify | re-loads the staged artifacts with `onnxruntime` and re-tokenizes with Rust `tokenizers` | must print **ALL SELECTED CELLS PASSED** |
| 9 — artifacts | copy `ml/out/run/staging/*` into a flat `out-export/` in the notebook output | tiny — see §6 |
| 10 — integrate | the exact Vercel wiring (also in §7 below) | — |

**Always run step 5 first.** It costs 20 seconds and it proves the dataset, the tokenizer, the
trainer, the calibrator, the exporter, the quantizers and the manifest all work *before* you spend
hours of GPU.

### The one cell that matters (step 6, by default)

```bash
torchrun --standalone --nproc_per_node=2 -m ml.kasauti_ml.pipeline \
  --profile kaggle --out /kaggle/working/out \
  --model-id convaiinnovations/laya --subfolder multilingual \
  --variants 40 --epochs 3 \
  --micro-batch 8 --grad-accum 4 --group-size 4 \
  --lr-encoder 2.5e-5 --lr-head 1e-4 \
  --max-len 1024 --head-max-len 256 \
  --abstain-target-error 0.15 --log-every 25
```

Everything you would want to change lives here:

| Flag | Meaning | When to change it |
|---|---|---|
| `--variants 40` | surface variants generated per template | **the main coverage knob.** More data beats more epochs |
| `--epochs 3` | RLCD passes over the train split | raise only if the loss is still falling (see `train_report.json`) |
| `--micro-batch` × `--grad-accum` | per-GPU microbatch × accumulation (8 × 4 = 32 effective, ×2 GPUs) | halve the microbatch and double the accumulation on OOM |
| `--group-size 4` | RLCD group size (reward baselines average within a group) | 2 for cheaper/noisier, 8 for a smoother gradient |
| `--tasks` | restrict to e.g. `claim_v1` | the fastest way to a *published* model you can react to: ~half the data |
| `--max-len 1024` / `--head-max-len 256` | context the sequence builder may use | Laya's documented budgets: 256 multilingual, 192 English |
| `--abstain-target-error 0.15` | target error on *accepted* answers — sets the abstention cuts | raise to 0.25 if the browser abstains on everything |
| `--public-dir` | also write the export somewhere local | leave empty on Kaggle; you download from *Output* |
| `--publish-repo` | push straight to an HF model repo (needs `HF_TOKEN`) | optional; step 9/21 in the notebook does the same thing explicitly |

The pipeline order is fixed and each stage is a gate for the next:

```
banks → data (split by template hash) → leak check → RLCD fine-tune (DDP)
      → per-bucket temperature fit → abstention cuts → eval gates
      → vocab pruning (lossless, drift-checked) → embedding gather
      → fp32 export → INT8 + 4-bit export → ORT agreement vs fp32
      → parity vectors → manifest v2 (sha256 of every byte) → verify → publish
```

Exit codes are meaningful: **0** published, **2** the accuracy gates failed (add data or epochs —
do *not* bypass), **3** the INT8 build disagrees with fp32 more than 5 % of the time (broken
quantization; the export is refused rather than shipped).

---

## 4. What to expect while it runs

* **Data**: ~7 920 train / 1 680 val / 720 test rows at `--variants 40`, expanded into
  (state, question, options) items with option-order augmentation.
* **Time**: roughly 1–2 h per epoch on T4 ×2 at `--max-len 1024`, so **2–5 h** for the default
  3 epochs, plus ~10–20 min for calibration, quantization and verification. `--tasks claim_v1`
  roughly halves it. Run it with **Save & Run All (Commit)** if you want it to survive your laptop
  closing.
* **Memory**: micro-batch 8 × 1024 tokens × 322 M params in bf16 on a 16 GB T4 is comfortable.
  `CUDA out of memory` means someone raised `MICRO_BATCH`/`MAX_LEN` — halve the batch, double the
  accumulation, keep the effective batch the same.
* **Noise**: two `TracerWarning`s from the ONNX export are expected and benign (`core.py:432`,
  `core.py:446` — a Python boolean that becomes a trace constant). Everything else in the log is
  either a gate line or a progress line.

---

## 5. Read the gate table (this is the point of the whole exercise)

`eval_report.json` holds the numbers and `ml/kasauti_ml/evals.py :: DEFAULT_GATES` holds the
thresholds the pipeline enforces. They also appear in the app, on **On-Device Engine → Eval gates**,
read straight from the manifest the browser downloaded.

| Gate | Threshold | Why it exists |
|---|---|---|
| `accuracy_overall` | ≥ 0.80 | the question is answered, not guessed |
| `accuracy_per_language_min` | ≥ 0.70 | the English-root checkpoint collapses on non-Latin scripts |
| `ece_overall_max` | ≤ 0.12 | Laya ships at ECE 0.314; a confidence you can't trust is decoration |
| `ece_per_bucket_max` | ≤ 0.20 | per (type, option-count) bucket, not just on average |
| `noul_accuracy_min` | ≥ 0.85 | follow-ups are `noul` (yes/no) — the weakest primitive |
| `choice_accuracy_min` | ≥ 0.75 | triage; ≤ 20 options, order-shuffled |
| `precision/recall_pay_to_withdraw` | ≥ 0.80 | the highest-harm script family |
| `recall_impersonation_min` | ≥ 0.85 | "digital arrest" style calls |
| `false_positive_rate_negatives_max` | ≤ 0.10 | measured on genuine SEBI advisories |
| `abstention_coverage_min` | ≥ 0.55 | a model that abstains on everything is useless |
| `quantized_agreement_min` | ≥ 0.95 | the INT8 graph must agree with fp32 top-1 |
| `tokenizer_parity_exact` | exact | browser ids must equal Rust `tokenizers` ids |

**A run that fails a gate publishes nothing.** You get exit code 2 and a printed table; the fix is
almost always more data (`--variants`) or more epochs — never a lowered threshold.

---

## 6. Get the artifacts out

The staging directory is what ships. It is small — the 322 M-parameter multilingual model in INT8
plus 4-bit plus fp32, the pruned tokenizer, the parity vectors and the manifest:

```
ml/out/run/staging/
├── manifest.json                 ← schema 2; the only file the browser reads first
├── kasauti-laya-int8.<sha8>.onnx ← WASM path (phones without WebGPU)
├── kasauti-laya-q4.<sha8>.onnx   ← WebGPU path (4-bit)
├── kasauti-laya-fp32.<sha8>.onnx ← reference only; the browser never fetches it
├── tokenizer.json                ← lossless vocab pruning, 0 segmentation drift
└── parity.json                   ← 10 tokenizer / 6 logit vectors for the in-browser self-test
```

Two ways out:

* **Publish to a Hugging Face model repo (recommended).** Free, CDN-backed, CORS-enabled,
  range-requests, no egress bill, and exactly what the blueprint requires — static bytes over
  `fetch()`, no Space, no compute we operate. In the notebook (step 9/21 or the pipeline's
  `--publish-repo`):

  ```python
  from huggingface_hub import HfApi
  HfApi().create_repo("kasauti-laya-multilingual-onnx", repo_type="model", private=False, exist_ok=True)
  HfApi().upload_folder(repo_id="<user>/kasauti-laya-multilingual-onnx", folder_path="/kaggle/working/out-export")
  ```

  The files must sit at the **root of the branch** (not in a subfolder) so that
  `https://huggingface.co/<user>/kasauti-laya-multilingual-onnx/resolve/main/manifest.json`
  resolves. Public repo — a private repo 404s for the browser and the engine will (correctly) report
  *no calibrated export published*.
* **Download from the notebook Output tab** and keep them as a build artifact. Do **not** commit a
  few hundred MB of ONNX into `public/`: it counts against Vercel's deployment size and every
  visitor would download it from your deployment instead of a CDN.

---

## 7. Integrate it (5 minutes)

1. **Vercel → your project → Settings → Environment Variables → Add**

   ```
   NEXT_PUBLIC_LAYA_MODEL_BASE = https://huggingface.co/<user>/kasauti-laya-multilingual-onnx/resolve/main
   ```

   Production + Preview, all environments you deploy. The value is inlined at **build** time, so it
   must exist *before* the build — add it first, then redeploy.

2. **Redeploy** (manual, as per this project's rule — Vercel CLI or the dashboard's *Redeploy*).

3. **Verify in the browser** — open `/dashboard` → **On-Device Engine**:
   * *Status* → `ready`, *Detail* → `kasauti-laya-multilingual@<sha8> · webgpu` (or `wasm-st` /
     `wasm-mt` on a browser without WebGPU). It must **not** say *report-only*;
   * the *Calibration manifest* panel lists your fitted temperatures and abstention cuts, and
     *Artefacts verified* names the INT8/Q4 file you actually uploaded;
   * **no** *question-bank drift* line — if there is one, the model was trained on different question
     wording than the runtime asks; regenerate `src/lib/banks.ts` from the *same* `ml/banks.json` and
     retrain;
   * **Eval gates** panel shows your numbers against the thresholds, all green;
   * press **Build sequence** — sequence length and `[MASK]` marker count must match the run.
4. **Verify behaviour** — Message Checker, sample *VIP tips*: the ladder renders, *Laya Tier-1* shows
   `triage = investment_offer` with a real confidence, and the follow-up lines carry probabilities.
   Paste a genuine SEBI advisory from the samples: it must **not** escalate.
5. **Verify offline + deferral** — reload with DevTools → Network → *Offline*: the check still works
   (Cache API). Then set chrome://flags *Save-Data* or throttle to 2G: the app offers Tier-0 and
   defers the download, with copy that says why.
6. **Rollback** is one step: delete or blank `NEXT_PUBLIC_LAYA_MODEL_BASE` and redeploy. The engine
   reports *unpublished* and the whole app runs on Tier-0 — no error, no dead end.

---

## 8. Local development with the same path

You do not need Kaggle to exercise the browser end-to-end. The repo ships a **plumbing-test** export
in `public/dev-model/` (a ~0.5 M-parameter random model, `reportOnly: true`, so it can never act):

```bash
python -m ml.scripts.gen_banks_ts
python -m ml.kasauti_ml.pipeline --profile smoke --out ml/out/dev --public-dir public/dev-model
NEXT_PUBLIC_LAYA_MODEL_BASE=/dev-model npm run dev
node tests/e2e-devmode.mjs http://localhost:3000
```

The E2E drives a real browser, loads the real graph, hashes it, tokenizes with the pure-TS
tokenizer, answers, abstains and fills the Cache API — the same code path your Kaggle model will
take, with numbers that mean nothing.

---

## 9. When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| pipeline exits **2** | accuracy gates failed | `--variants 80`, or `--epochs 5`, or narrow to `--tasks claim_v1` |
| pipeline exits **3** | INT8 ↔ fp32 top-1 agreement < 0.95 | keep `--quantize-embedding`, try `--block-size 64`, and check for a corrupted export |
| `CUDA out of memory` | microbatch × 1024 tokens | halve `--micro-batch`, double `--grad-accum` |
| browser: *no calibrated export published* | manifest 404 | wrong `NEXT_PUBLIC_LAYA_MODEL_BASE`, private repo, or files in a subfolder |
| browser: *report-only* | this export didn't clear the gates | read the gate table; it is telling the truth, not failing |
| browser: *question-bank drift* | trained on other wording | regenerate `src/lib/banks.ts` and retrain |
| everything abstains | cuts fitted to a hard split | re-run with `--abstain-target-error 0.25` |
| tokenizer mismatch at load | graph and tokenizer from different runs | always ship the whole `staging/` directory together |
| slow first check (> 10 s) | cold download of the INT8 graph | expected once; afterwards it is in the Cache API |
| `text-generation` style output | you are using the wrong repo | Kasauti uses the **encoder + decision head**; there is no generation anywhere |

---

## 10. Licences and honesty

Laya is Apache-2.0 (`github.com/NandhaKishorM/laya`, model cards at
`huggingface.co/convaiinnovations/laya`). A fine-tune is a derivative: keep the licence file, credit
the authors in your model repo card, and state that the numbers this pipeline publishes are measured
on **synthetic** data generated from `ml/banks.json` templates — not on real reported scams, which
you do not have. Say that in the model card, in the same sentence as the accuracy number. The app
already does the equivalent: every simulated figure in the dashboard is labelled, and the eval-gate
panel shows exactly what the manifest claims instead of asserting quality.

Speech never leaves the device, and with this architecture **nothing** leaves the device: the model
is fetched as static bytes and executed by `onnxruntime-web` inside the visitor's browser. The only
exception in the whole design is the telephony/IVR adapter, which is a separate, documented route.
