#!/usr/bin/env python3
"""Generate `notebooks/kasauti_laya_finetune_kaggle.ipynb`.

The notebook is GENERATED rather than hand-written so it cannot silently drift away from the code
it drives: every command in it is one this repo already runs locally (the smoke profile), and the
flags are the actual `ml/kasauti_ml/pipeline.py` flags.

Two properties this generator is responsible for:

* **Every cell runs.** `SCOPE = "full"` is the Kaggle path (2×T4, the real multilingual
  checkpoint). `SCOPE = "smoke"` skips only the GPU fine-tune and analyses the CPU smoke artifacts
  instead, so the whole notebook — not a subset — can be executed on any machine. The repo's own
  verification run uses that (`KASAUTI_NOTEBOOK_SCOPE=smoke`).
* **Nothing is unpinned.** The base checkpoint comes from Laya's reviewed revision
  (`laya.revisions.PINNED_REVISIONS`), the fine-tune is reproducible from the manifest, and the
  export is cross-checked against the reference client before it can be published.

Usage:
    python -m ml.scripts.gen_notebook                       # write the notebook
    python -m ml.scripts.gen_notebook --check               # exit 1 when stale
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "kasauti_laya_finetune_kaggle.ipynb")

MD = "markdown"
CODE = "code"


def _cell(kind: str, source: str, cell_id: str) -> Dict[str, Any]:
    lines = source.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    # nbformat 4.5 requires a unique cell id; without one every nbformat >= 5.1.4 warns and future
    # versions refuse to write the notebook at all
    cell: Dict[str, Any] = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": src}
    if kind == CODE:
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ title
    cells.append(_cell(MD, r"""
# Kasauti × Laya — fine-tune, verify, quantize and publish (Kaggle, 2×T4)

This notebook turns the **latest Laya checkpoint** into the exact artifact Kasauti's browser
runtime loads: a vocabulary-pruned, INT8 + 4-bit ONNX pair with fitted per-bucket temperatures,
abstention thresholds and parity vectors, published as **static bytes in a Hugging Face *model*
repo** (never a Space, never a server).

**What runs where.** Laya is Kasauti's Tier-1 *typed-decision* model: state + typed questions in,
typed answers with calibrated probabilities out, one forward pass, no text generation. It is
fine-tuned here with **RLCD** — REINFORCE on a group-mean baseline with zero-mean Gaussian logit
noise, scored by a strictly proper scoring rule — the same loop the reference `train_ddp.py`
implements, using the banks in `ml/banks.json` as the only source of question wording.

**Before you spend GPU time**

* Settings → Accelerator → **GPU T4 ×2** (`nvidiaTeslaT4` is selected here; Kaggle's `×2` toggle is
  a UI choice the metadata cannot carry), and **Internet on**.
* Add your Hugging Face **write** token as a Kaggle Secret named `HF_TOKEN` — only step 10 needs it.
* Run the cells in order. **Step 5 is a ~2-minute CPU smoke run** that exercises the whole chain
  (data → RLCD → calibration → INT8/Q4 export → ORT agreement → parity vectors → manifest →
  staging → reference cross-check) on a tiny randomly-initialised model. If it passes, the only
  things left that can fail are GPU memory and the model's actual accuracy.

**Expected wall clock on 2×T4:** deps ~1 min · data ~30 s · smoke ~2 min · fine-tune ~5–10 min ·
calibration + eval ~1 min · INT8/Q4 export + agreement ~1 min · reference check ~1 min ·
publish ~1 min.

**The one thing this notebook refuses to do:** ship a model it cannot reproduce. The base
checkpoint is pinned to Laya's reviewed revision, the manifest records which revision it came from,
and step 9 re-runs the export through the *reference* Laya client before step 10 is allowed to
publish it.

*Laya is Apache-2.0 (github.com/NandhaKishorM/laya). This notebook trains a derivative model and
publishes it under your own account; keep the upstream attribution in the model card, and keep
`docs/privacy-notice.md` with the app.*
""", "kasauti-laya-fine-tune-01"))

    # ------------------------------------------------------------------ install
    cells.append(_cell(MD, r"""
## Step 0 — install dependencies

Kaggle's image already ships PyTorch + CUDA. What is added here:

* **`laya[onnx]`** — the reference implementation *and* its ONNX agent. The `onnx` extra is what
  lets step 9 hand the exported graph to `laya.onnx_agent.ONNXAgent`; without it the notebook still
  trains and exports, but the independent cross-check is skipped (and reported as skipped).
* `tokenizers` / `safetensors` / `huggingface_hub` — the pipeline compares its own tokenizer
  handling against the Rust library the model was trained with.
* `onnx` / `onnxruntime` / `onnx_ir` — export and quantization. ORT ≥ 1.21 moved the 4-bit
  weight-only quantizer into `onnx_ir`, which `MatMulNBitsQuantizer` imports.

Floors are what `laya` itself declares (`transformers>=4.48`, `torch>=2.0`), so pip is free to take
a newer minor — the cell prints what actually resolved, and that is what the model card should say.
""", "step-0-install-dependencies-02"))
    cells.append(_cell(CODE, r"""
%pip install -q "laya[onnx]>=0.3.26" "transformers>=4.48" "tokenizers>=0.21" "safetensors>=0.4" \
    "onnx>=1.17" "onnxruntime>=1.20" "onnx_ir>=0.1" "huggingface_hub>=0.26" "scikit-learn>=1.5"

import importlib
for mod in ("laya", "torch", "transformers", "tokenizers", "onnx", "onnxruntime", "huggingface_hub"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<18} {getattr(m, '__version__', '?')}")
    except Exception as exc:                      # a missing optional is information, not a crash
        print(f"  {mod:<18} MISSING ({type(exc).__name__})")

import laya
assert tuple(int(x) for x in laya.__version__.split(".")[:3]) >= (0, 3, 20), \
    f"this notebook is written for laya >= 0.3.20 (found {laya.__version__})"
print("\nlaya.revisions.PINNED_REVISIONS:", laya.PINNED_REVISIONS)
""", "pip-install-q-laya-onnx-03"))

    # ------------------------------------------------------------------ code + preflight
    cells.append(_cell(MD, r"""
## Step 1 — get the project code, and check the machine

Two ways to get the pipeline here, pick one:

1. **Clone the repo** (default; the clone also brings `ml/banks.json`, the question banks, which are
   the single source of truth for the wording the browser will ask).
2. **Attach the repo as a Kaggle Dataset** and set `PROJECT_SRC` to its mount path
   (`/kaggle/input/<dataset>`), so each run is reproducible without network.

The pipeline imports as `ml.kasauti_ml.*`, so every later cell runs from the directory *containing*
`ml/`. The environment lines below are the preflight that decides whether step 6 can run at all:
two T4s, enough system RAM for the 647 MB checkpoint plus activations, and a free disk for the
~1.3 GB fp32 export that the quantizers read.
""", "step-1-get-the-project-code-and--04"))
    cells.append(_cell(CODE, r"""
import os, shutil, subprocess, sys

REPO_URL     = "https://github.com/pmrinal2005/Kasauti.git"   # change if you forked it
WORK         = os.environ.get("KASAUTI_WORK", "/kaggle/working")
PROJECT_DIR  = os.environ.get("KASAUTI_PROJECT_DIR", os.path.join(WORK, "Kasauti"))
PROJECT_SRC  = ""            # e.g. "/kaggle/input/kasauti-ml" to use an attached dataset instead

if PROJECT_SRC:
    os.makedirs(PROJECT_DIR, exist_ok=True)
    subprocess.run(["cp", "-r", f"{PROJECT_SRC.rstrip('/')}/.", PROJECT_DIR], check=True)
elif not os.path.isdir(os.path.join(PROJECT_DIR, "ml")):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, PROJECT_DIR], check=True)

os.chdir(PROJECT_DIR)
print("project:", PROJECT_DIR)
for rel in ("ml/banks.json", "ml/kasauti_ml/pipeline.py", "ml/kasauti_ml/reference_check.py",
            "src/lib/banks.ts", "notebooks/kasauti_laya_finetune_kaggle.ipynb"):
    print(f"  {'ok ' if os.path.exists(rel) else 'MISSING'} {rel}")

import torch
os.makedirs(WORK, exist_ok=True)              # Kaggle pre-creates /kaggle/working; a bare box may not
gpus = torch.cuda.device_count()
free_gb = shutil.disk_usage(WORK).free / 1e9
ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
print(f"\ntorch {torch.__version__} · cuda {torch.cuda.is_available()} · {gpus} device(s) · "
      f"{ram_gb:.1f} GB RAM · {free_gb:.1f} GB free in {WORK}")
for i in range(gpus):
    print("  gpu%d %s %.1f GB" % (i, torch.cuda.get_device_name(i),
                                  torch.cuda.get_device_properties(i).total_memory / 1e9))
if gpus < 2:
    print("\nNOTE: one GPU works, but the DDP cell below launches two processes — either switch the\n"
          "      accelerator to 'GPU T4 ×2' or run the pipeline without torchrun:\n"
          "      python -m ml.kasauti_ml.pipeline --profile kaggle --out /kaggle/working/out …")
""", "import-os-shutil-subprocess-05"))

    # ------------------------------------------------------------------ config
    cells.append(_cell(MD, r"""
## Step 2 — configuration

Everything the run depends on, in one place. Defaults are the reference setup (2×T4, all six
banks); the knobs that matter, and when to change them:

| flag | default | change it when |
|---|---|---|
| `SCOPE` | `"full"` | leave it. `"smoke"` (or `KASAUTI_NOTEBOOK_SCOPE=smoke`) skips the GPU fine-tune and analyses the CPU smoke artifacts — that is how this notebook is executed without a GPU |
| `EPOCHS` | 3 | gates fail on accuracy → 5–6, but watch eval ECE rise after that |
| `MICRO_BATCH` / `GRAD_ACCUM` | 8 / 4 | CUDA OOM → halve `MICRO_BATCH`, double `GRAD_ACCUM` (effective batch stays 32) |
| `VARIANTS` | 40 | you want more data per template; more data beats more epochs here |
| `TASKS` | all 6 banks | you want a narrower model (e.g. only `claim_v1`) — smaller, faster, sharper |
| `LR_ENCODER` / `LR_HEAD` | 2.5e-5 / 1e-4 | rarely. The encoder must move slowly; 1e-4 on it destroys the pretrained features |
| `MAX_LEN` | 1024 | the multilingual checkpoint's default; `laya-multilingual` accepts up to 8192 (`--max-len 8192`) for long transcript states, at ~8× the cost |
| `BASE_REVISION` | `""` = reviewed pin | only to reproduce an older export; the pipeline resolves `laya.PINNED_REVISIONS` for `MODEL_ID` |
| `PUBLISH_REPO` | empty | set to `you/kasauti-laya-multilingual-onnx` to let step 10 publish |
| `PUBLIC_DIR` | empty | set to `f"{PROJECT_DIR}/public/dev-model"` to also write the browser's dev model and run the Playwright E2E afterwards |

`ABSTAIN_TARGET_ERROR` is the point of the whole abstention map: it is the error rate the runtime is
allowed to have **among the answers it accepts**. 0.15 means "of the answers we do give, ~85% are
right"; everything below that confidence abstains and the UI says so instead of guessing.
""", "step-2-configuration-06"))
    cells.append(_cell(CODE, r"""
import os

SCOPE = os.environ.get("KASAUTI_NOTEBOOK_SCOPE", "full")     # "full" (2×T4) | "smoke" (CPU)
assert SCOPE in ("full", "smoke")

REAL_DIR     = os.path.join(WORK, "out")          # everything the real run writes
SMOKE_DIR    = os.path.join(WORK, "smoke")        # tiny end-to-end validation run
OUT_DIR      = REAL_DIR if SCOPE == "full" else SMOKE_DIR

MODEL_ID     = "convaiinnovations/laya"        # root repo: also holds `multilingual/`
SUBFOLDER    = "multilingual"                  # 100+ languages, mmBERT-base, 1024 ctx — what Kasauti ships
BASE_REVISION = ""                             # "" = the reviewed pin for MODEL_ID
TASKS        = ""                              # "" = all banks, else "claim_v1,lens_v1"
VARIANTS     = 40                              # surface variants per template
EPOCHS       = 3
MICRO_BATCH  = 8
GRAD_ACCUM   = 4
GROUP_SIZE   = 4                               # RLCD noise samples per item (reference default)
LR_ENCODER   = 2.5e-5
LR_HEAD      = 1e-4
MAX_LEN      = 1024
HEAD_MAX_LEN = 256
ABSTAIN_TARGET_ERROR = 0.15                    # accepted-answer error the abstention map must hold

PUBLISH_REPO = ""                              # e.g. "yourname/kasauti-laya-multilingual-onnx"
PUBLIC_DIR   = ""                              # e.g. f"{PROJECT_DIR}/public/dev-model"

HF_TOKEN = ""
try:                                           # Kaggle secret; harmless if absent
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as exc:
    print("no HF_TOKEN secret (fine unless you publish in step 10):", type(exc).__name__)

for d in (REAL_DIR, SMOKE_DIR):
    os.makedirs(d, exist_ok=True)
print("scope:", SCOPE, "· the cells that follow read/write", OUT_DIR)
print("config:", dict(TASKS=TASKS or "all", VARIANTS=VARIANTS, EPOCHS=EPOCHS, MICRO_BATCH=MICRO_BATCH,
                      GRAD_ACCUM=GRAD_ACCUM, GROUP_SIZE=GROUP_SIZE, MAX_LEN=MAX_LEN,
                      HEAD_MAX_LEN=HEAD_MAX_LEN, BASE_REVISION=BASE_REVISION or "(reviewed pin)"))
""", "import-os-scope-os-environ-07"))

    # ------------------------------------------------------------------ provenance
    cells.append(_cell(MD, r"""
## Step 3 — provenance check (question-bank drift)

The most expensive failure in this pipeline is silent: the model gets fine-tuned on one wording of a
question and the browser asks a slightly different one. Nothing errors, the answers are just worse.
`ml/banks.json` is the single source of truth; `src/lib/banks.ts` is generated from it, and the
manifest carries the same sha256 so the runtime can detect the drift at load time.

This cell fails loudly if the two halves disagree — before any GPU minute is spent.
""", "step-3-provenance-check-question-08"))
    cells.append(_cell(CODE, r"""
import hashlib, json, subprocess, sys

banks = json.load(open("ml/banks.json", encoding="utf-8"))
sha = hashlib.sha256(json.dumps(banks, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8")).hexdigest()
print(f"banks version {banks['version']} · sha256 {sha}")
print(f"{'bank':<24}{'triage options':>15}{'max questions':>15}  title")
for bid, b in banks["tasks"].items():
    tri = b["triage"].get("criteria") or {}
    fol = max([len(v) for v in (b.get("followups") or {"": []}).values()] or [0])
    print(f"{bid:<24}{len(tri) if not isinstance(tri, list) else len(tri):>15}{1 + fol:>15}  {b['title']}")

r = subprocess.run([sys.executable, "-m", "ml.scripts.gen_banks_ts", "--check"],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
assert r.returncode == 0, "src/lib/banks.ts is stale — run: python -m ml.scripts.gen_banks_ts"
""", "import-hashlib-json-subproc-09"))

    # ------------------------------------------------------------------ data
    cells.append(_cell(MD, r"""
## Step 4 — inspect the training data

Rows are expanded from templates and split **by template hash**, not by row: a template that
appears in training can never appear in validation or test, so a high eval score cannot come from
having memorised the surface form. `template_overlap` must be `0` in every pair — the pipeline
aborts otherwise.

`gold.probabilities` is a distribution, not a label. Where a case is genuinely ambiguous the
distribution is soft on purpose: the RLCD reward is a proper scoring rule, so the model is trained
to be *right about its uncertainty* rather than confident and wrong.
""", "step-4-inspect-the-training-data-10"))
    cells.append(_cell(CODE, r"""
import sys
sys.path.insert(0, ".")
from ml.kasauti_ml import datagen

tasks = datagen.load_tasks("ml/banks.json")
if TASKS:
    want = {t.strip() for t in TASKS.split(",") if t.strip()}
    tasks = {k: v for k, v in tasks.items() if k in want}
templates = [t for name in tasks for t in datagen.TEMPLATES_BY_TASK[name]]
rows = datagen.build_rows(tasks, templates, n_variants=VARIANTS, seed=7, romanized=True)
stats = datagen.dataset_stats(rows)
print(json.dumps(stats, indent=1)[:1200])
assert not any(stats["template_overlap"].values()), f"template leakage: {stats['template_overlap']}"

print("\nexample row (state truncated):")
ex = rows[0]
print(json.dumps({"task": ex["task"], "lang": ex["lang"], "split": ex["split"],
                  "state": ex["state"][:180], "triage": ex["questions"]["triage"]["instructions"],
                  "gold": ex["gold"]["triage"]}, ensure_ascii=False, indent=1))
""", "import-sys-sys-path-insert-0-11"))

    # ------------------------------------------------------------------ smoke
    cells.append(_cell(MD, r"""
## Step 5 — CPU smoke run (always; do this before burning GPU time)

A tiny randomly-initialised model is pushed through **every** stage on CPU: data → RLCD training →
calibration → gates → vocabulary pruning → fp32 export → INT8 + Q4 export → ORT agreement → parity
vectors → manifest → staging → **reference cross-check**.

What it proves is the plumbing, and only the plumbing. What it does *not* prove is accuracy: a
0.5 M-parameter model trained for two epochs on synthetic data cannot learn six decision banks, so
the accuracy gates *will* fail here — the pipeline says so and stages the export as **report-only**
(the runtime then abstains on everything rather than acting on an unproven model). Accuracy is the
GPU run's job.

`--reference-check` is the part that matters even when it passes: it loads the export with Laya's
own `ONNXAgent`, replays real questions through the reference client and compares the answers with
our decode. A failure there stops the run (exit 4) instead of publishing.
""", "step-5-cpu-smoke-run-always-do-t-12"))
    cells.append(_cell(CODE, r"""
import json, os, subprocess, sys

cmd = [sys.executable, "-m", "ml.kasauti_ml.pipeline", "--profile", "smoke",
       "--out", SMOKE_DIR, "--variants", "4", "--epochs", "2", "--log-every", "5000",
       "--reference-check"]
print(" ".join(cmd), "\n", flush=True)
r = subprocess.run(cmd, capture_output=True, text=True)
print((r.stdout or r.stderr)[-5000:])
assert r.returncode == 0, "smoke run failed — fix this before the GPU run"

smoke = json.load(open(f"{SMOKE_DIR}/summary.json", encoding="utf-8"))
print("\nsmoke artifacts MB:", smoke["sizes_mb"], "· modelVersion:", smoke["modelVersion"])
print("staged:", sorted(os.listdir(f"{SMOKE_DIR}/staging")))
ref = json.load(open(f"{SMOKE_DIR}/reference_check.json", encoding="utf-8"))
print("reference check:", {k: (v.get("ok") if isinstance(v, dict) else v)
                           for k, v in ref.items() if k in
                           ("mirrors", "calibration", "referenceClient", "referenceEvals")})
""", "import-json-os-subprocess-13"))

    # ------------------------------------------------------------------ train
    cells.append(_cell(MD, r"""
## Step 6 — the real fine-tune (2×T4, DDP)

`torchrun` launches one process per GPU; the pipeline wraps the model in `DistributedDataParallel`,
shards the rows by rank and averages gradients, so both T4s train **one** model. Rank 0 does the
evaluation, calibration, export and publish — ranks 1+ stop after training.

The loop is RLCD:

1. **Sample** `GROUP_SIZE` noisy distributions per item with the zero-mean projected Gaussian
   perturbation `eps - mean(eps)`, so the noise cannot change the ranking the model already has.
2. **Score** each sample with the strictly proper rule (log-score + spherical, minus a ranked
   probability score on `score` questions) against the soft gold distribution.
3. **REINFORCE** on the group-mean baseline plus a soft cross-entropy anchor, with rare-class
   weights and a negative-class boost so the "normal message" class is not drowned out.

Watch `reward` trend upward and `sigma` anneal 0.4 → 0.1. `loss` itself is not a progress metric
(the RL term is a surrogate).

**The base checkpoint is pinned.** `--revision` (filled from `laya.revisions.PINNED_REVISIONS` when
`BASE_REVISION` is empty) and only the `multilingual/` files are downloaded — five files, ~680 MB,
instead of the repo's ~2.5 GB. Without the pin, the same command run next month would silently
fine-tune a different base and the export could never be reproduced from its own manifest.

Exit codes: `2` = eval gates failed (add data/epochs, or narrow `TASKS`), `3` = INT8 drifted from
fp32, `4` = the reference client disagreed with our export. All three mean *nothing was published*.
""", "step-6-the-real-fine-tune-2-t4-d-14"))
    cells.append(_cell(CODE, r"""
import json, subprocess, sys, time

if SCOPE != "full":
    print(f"SCOPE={SCOPE}: skipping the GPU fine-tune — steps 7-10 read the smoke artifacts in\n"
          f"{SMOKE_DIR} instead. Set SCOPE='full' (Accelerator: GPU T4 ×2) to run it.")
else:
    cmd = ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "ml.kasauti_ml.pipeline",
           "--profile", "kaggle", "--out", OUT_DIR,
           "--model-id", MODEL_ID, "--subfolder", SUBFOLDER,
           "--variants", str(VARIANTS), "--epochs", str(EPOCHS),
           "--micro-batch", str(MICRO_BATCH), "--grad-accum", str(GRAD_ACCUM),
           "--group-size", str(GROUP_SIZE),
           "--lr-encoder", str(LR_ENCODER), "--lr-head", str(LR_HEAD),
           "--max-len", str(MAX_LEN), "--head-max-len", str(HEAD_MAX_LEN),
           "--abstain-target-error", str(ABSTAIN_TARGET_ERROR), "--log-every", "25",
           "--reference-check"]
    if BASE_REVISION:
        cmd += ["--revision", BASE_REVISION]
    if TASKS:
        cmd += ["--tasks", TASKS]
    if PUBLIC_DIR:
        cmd += ["--public-dir", PUBLIC_DIR]
    print(" ".join(cmd), "\n", flush=True)

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:                   # stream, so a 10-minute run is not a black box
        print(line, end="")
    proc.wait()
    print(f"\nexit={proc.returncode} in {(time.time()-t0)/60:.1f} min")
    assert proc.returncode == 0, (
        "pipeline failed. exit 2 = accuracy gates failed (more VARIANTS/EPOCHS, or narrower TASKS); "
        "exit 3 = INT8 drifted from fp32 (re-export with --block-size 64); exit 4 = the reference "
        "client disagreed with our export (read reference_check.json); exit 5 = the pruned tokenizer "
        "re-segments the corpus differently (widen the corpus with more VARIANTS). Gating is the "
        "point — do not skip it.")

    summary = json.load(open(f"{OUT_DIR}/summary.json", encoding="utf-8"))
    print("\nbase checkpoint this model was trained from:", json.dumps(summary["base"], indent=1))
""", "import-json-subprocess-sys-15"))

    # ------------------------------------------------------------------ reports
    cells.append(_cell(MD, r"""
## Step 7 — read the reports

Three files decide whether this model ships:

* `eval_report.json` — every gate with the threshold it had to clear, plus the metrics behind it.
* `calibration.json` — the fitted temperature per bucket, the abstention cut that holds
  accepted-answer error under `ABSTAIN_TARGET_ERROR`, and the coverage that implies.
* `summary.json` — artifact sizes, vocabulary pruning, quantized agreement, and the base-checkpoint
  provenance (which repo, which subfolder, which commit).

A note on reading ECE: the base Laya multilingual checkpoint ships **over-confident** (ECE ≈ 0.31 on
the card's own measurement). If the fitted ECE is meaningfully below the raw ECE, calibration is
doing real work, and the abstention map is then meaningful rather than decorative. In the smoke
profile the numbers are meaningless by construction — the model is random; only the *file shapes*
are being tested there.
""", "step-7-read-the-reports-16"))
    cells.append(_cell(CODE, r"""
import json, os

ev   = json.load(open(f"{OUT_DIR}/eval_report.json", encoding="utf-8"))
cal  = json.load(open(f"{OUT_DIR}/calibration.json", encoding="utf-8"))
summ = json.load(open(f"{OUT_DIR}/summary.json", encoding="utf-8"))

print(f"gates ({SCOPE} scope — {'advisory: smoke gates are expected to fail' if SCOPE != 'full' else 'blocking'})")
for c in ev["gates"]["checks"]:
    print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']:<34} value={c['value']}  threshold={c['threshold']}")
print("\nmetrics:", json.dumps({k: v for k, v in ev["metrics"].items()
                                if k in ("n", "accuracy", "choice_accuracy", "noul_accuracy",
                                         "score_mae", "ece", "brier", "per_language")}, indent=1))
print("\ncalibration: ECE raw", cal.get("ece_raw", {}).get("overall"), "→ fitted",
      cal.get("ece_fitted", {}).get("overall"))
print("temperatures per type:", [round(t, 3) for t in cal["temperature"]])
print("temperature per bucket:", {k: round(v, 3) for k, v in cal["temperature_by_options"].items()})
print("abstention map:", {k: round(v, 3) for k, v in cal.get("abstain", {}).items()})
print("coverage at that map:", cal.get("coverage"))
print("\nbase:", json.dumps(summ.get("base", {})))
print("sizes MB:", summ["sizes_mb"], "· agreement:", summ["agreement"])
""", "import-json-os-ev-json-17"))

    # ------------------------------------------------------------------ verify
    cells.append(_cell(MD, r"""
## Step 8 — verify the staged artifacts (independent of the trainer)

This re-checks the export the way the **browser** will load it, not the way the trainer produced it:

1. every file's sha256 against the manifest (a manifest that lies is worse than no manifest);
2. both quantized graphs replayed on the parity vectors and compared with the fp32 reference,
   through the same ONNX Runtime engine family the browser uses;
3. the graph still exposes `input_ids / attention_mask / marker_pos / marker_mask / qtype`;
4. the manifest is honest about what it is — `reportOnly: true` means the runtime will abstain on
   everything by construction.

The numbers that matter here are the ones the **Worker** will enforce at load time: the tokenizer
vectors must replay id-for-id, and each quantized graph must stay inside the logits tolerance the
manifest publishes for it. A 4-bit graph that drifted further than that is not "a bit less accurate"
— the runtime refuses it, and this cell fails the same way. The trainer's own INT8 top-1 agreement
over the eval split is a separate, harder gate (`>= 0.95`, exit `3`) and is printed alongside, so
the two numbers can be read together. If either is short, re-export with `--block-size 64` and keep
`--quantize-embedding`.
""", "step-8-verify-the-staged-artifac-18"))
    cells.append(_cell(CODE, r"""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, ".")
from ml.kasauti_ml import export as mlx

STAGE = f"{OUT_DIR}/staging"
man = json.load(open(f"{STAGE}/manifest.json", encoding="utf-8"))
summary_agreement = json.load(open(f"{OUT_DIR}/summary.json", encoding="utf-8"))["agreement"]

ok = True
for key, spec in man["files"].items():
    p = os.path.join(STAGE, spec["path"])
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    good = got == spec["sha256"] and os.path.getsize(p) == spec["bytes"]
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} {key:<7} {spec['path']:<38} {spec['bytes']/1048576:6.1f} MB")
p = os.path.join(STAGE, man["tokenizer"]["path"])
good = hashlib.sha256(open(p, "rb").read()).hexdigest() == man["tokenizer"]["sha256"]
ok &= good
print(f"  {'ok ' if good else 'BAD'} tokenizer  vocab={man['tokenizer']['vocab']} pruned={man['tokenizer']['pruned']}")
assert ok, "artifact hashes do not match the manifest"

# 2) the browser's own self-test, replayed here on the staged bytes: tokenizer ids must match
#    exactly, and every quantized graph must reproduce the published fp32 logits within the
#    tolerance the manifest carries (that tolerance is what the Worker compares against on load).
parity = json.load(open(os.path.join(STAGE, man["parity"]["path"]), encoding="utf-8"))
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(STAGE, man["tokenizer"]["path"]))
bad = [v["text"][:40] for v in parity["tokenizer"]["vectors"]
       if list(tok.encode(v["text"], add_special_tokens=False).ids) != list(v["ids"])]
assert not bad, f"tokenizer parity vectors do not replay: {bad}"
print(f"\ntokenizer parity: {len(parity['tokenizer']['vectors'])} texts replay exactly")

vecs = parity["logits"]["vectors"]
tol = float(parity["logits"]["tolerance"])
items = [{"ids": v["ids"], "markers": v["markers"], "qtype": v["qtype"],
          "target": [0.0] * len(v["markers"])} for v in vecs]
ref = [np.asarray(v["logits"], dtype=np.float64) for v in vecs]
print(f"logits parity: {len(vecs)} vectors · published tolerance {tol}")
for key, label in (("fp32", "fp32 (reference)"), ("wasm", "INT8 (WASM EP)"), ("webgpu", "4-bit (WebGPU EP)")):
    sess = mlx.ort_session(os.path.join(STAGE, man["files"][key]["path"]))
    got = [np.asarray(l, dtype=np.float64) for _qt, l, _t in mlx.ort_predict(sess, items)]
    deltas = [float(np.abs(g - r).max()) for g, r in zip(got, ref)]
    top1 = float(np.mean([float(np.argmax(g) == np.argmax(r)) for g, r in zip(got, ref)]))
    within = sum(1 for d in deltas if d <= tol)
    print(f"  {label:<20} max |Δlogit| {max(deltas):.4f} · top-1 {top1:.3f} · "
          f"within tolerance {within}/{len(vecs)}")
    assert within == len(vecs), (f"{label} exceeds the published tolerance ({tol}) on "
                                 f"{len(vecs) - within} vector(s) — the Worker would refuse this "
                                 f"graph. Re-export with --block-size 64.")

# 3) the big-sample quantization number the trainer measured on the eval split (the pipeline gates
#    INT8 at >= 0.95 and exits 3 otherwise), printed here so both numbers sit side by side
print(f"\nquantization agreement over the eval split (trainer): {json.dumps(summary_agreement)}")

io = mlx.graph_io_summary(os.path.join(STAGE, man["files"]["wasm"]["path"]))
print("\ngraph inputs:", [i["name"] for i in io["inputs"]], "→ outputs:", [o["name"] for o in io["outputs"]])
assert {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"} <= {i["name"] for i in io["inputs"]}
print("manifest:", man["modelVersion"], "· eval at export:", man["evals"])

if man.get("reportOnly"):
    print("\nNOTE: this export is REPORT-ONLY — it did not clear the eval gates (or it was built with"
          "\n      --profile smoke), and the runtime abstains on every answer and says so. Inspect it,"
          "\n      do not ship it: raise VARIANTS/EPOCHS (or narrow TASKS) and re-run step 6.")
else:
    print("\nALL EXPORT CHECKS PASSED — calibrated and gate-passing.")
""", "import-hashlib-json-os-sys-19"))

    # ------------------------------------------------------------------ reference parity
    cells.append(_cell(MD, r"""
## Step 9 — independent cross-check against the reference Laya client

Everything Kasauti ships is a *reimplementation* of something in `laya`: the browser cannot run
PyTorch, so the tensor construction, the calibration fit and the reward were reimplemented in Python
and then in TypeScript. Reimplementations drift silently — a wrong token id is not an exception, it
is a slightly worse answer.

This step therefore stops treating our mirrors as the source of truth and compares them with the
installed package:

* **`mirrors`** — `build_sequence` / `build_head` / `render_options` / `temp_bucket` /
  `clamp_temperature` / `ece_score` / `proper_reward` against `laya.common`, on every question in
  the banks and a spread of real states, including permuted option orders.
* **`calibration`** — our fitters and `laya.calibrate` on *identical* records, so a drift in the
  numbers the manifest publishes is caught here.
* **`reference client`** — `laya.onnx_agent.ONNXAgent` pointed at a scratch checkpoint holding our
  pruned tokenizer, our fitted temperatures and our exported graph; its answers are compared with
  our own decode. If the reference client and the browser disagree about the same bytes, one of them
  is wrong.
* **`evals`** — `laya.evals.evaluate` with the reference client as the runner (informational).

`tests/test_reference_parity.py` is the same contract in test form and runs without a GPU; it is
what CI should run on every commit that touches `ml/`.
""", "step-9-independent-cross-check-a-20"))
    cells.append(_cell(CODE, r"""
import json, subprocess, sys

r = subprocess.run([sys.executable, "tests/test_reference_parity.py"], capture_output=True, text=True)
print(r.stdout or r.stderr)
# 77 = the `laya` package is not importable; 0 = every check passed
assert r.returncode in (0, 77), "reference parity checks FAILED — see the output above"

ref = json.load(open(f"{OUT_DIR}/reference_check.json", encoding="utf-8"))
print("pipeline reference check (laya " + str(ref.get("layaVersion")) + "):")
for section in ("mirrors", "calibration", "referenceClient", "referenceEvals"):
    body = ref.get(section) or {}
    detail = body.get("skipped") or (f"{body.get('compared', body.get('n_examples', ''))} items"
                                     if body.get("ok") else body.get("failures"))
    print(f"  {'PASS' if body.get('ok') else 'FAIL'}  {section:<16} {detail}")
print("\nreference client max probability delta:", (ref.get("referenceClient") or {}).get("max_probability_delta"))
print("reference evals overall:", json.dumps((ref.get("referenceEvals") or {}).get("overall"), indent=1))
assert ref.get("passed"), "the pipeline's reference check did not pass — do not publish this export"
""", "import-json-subprocess-sys-21"))

    # ------------------------------------------------------------------ publish
    cells.append(_cell(MD, r"""
## Step 10 — get the artifacts out

**Either** keep them as notebook output (they appear under *Output* → `out/` and can be attached as
a Kaggle dataset), **or** publish them to a Hugging Face **model** repo — which is what the browser
fetches. A model repo is static LFS storage on the Hub's CDN: no Space, no GPU quota, no Kasauti
server in the path, and nothing that can be paused or billed.

The staging directory is uploaded **as a whole directory**: `manifest.json`, the two graphs, the
pruned tokenizer, `parity.json` and `reference_check.json`. Publishing a subset is how a client ends
up with a tokenizer whose ids do not match the graph it downloaded — the manifest's sha256 fields are
what let the browser notice, and the service worker refuses a mismatched artifact rather than
loading it.
""", "step-10-get-the-artifacts-out-22"))
    cells.append(_cell(CODE, r"""
import json, os, shutil

KEEP = os.path.join(WORK, "out-export")           # /kaggle/working/out-export → in the notebook output
shutil.rmtree(KEEP, ignore_errors=True)
# a flat copy next to (not inside) the repo clone, so Kaggle's Output panel shows it directly
shutil.copytree(f"{OUT_DIR}/staging", KEEP)
print("copied →", KEEP)
print(sorted(os.listdir(KEEP)))
man = json.load(open(f"{KEEP}/manifest.json", encoding="utf-8"))
print("\nfiles:", {k: v["path"] for k, v in man["files"].items()})
print("base:", json.dumps(man.get("base", {})))
""", "import-json-os-shutil-keep-23"))
    cells.append(_cell(CODE, r"""
# Publish to a Hugging Face MODEL repo (static bytes; no compute is ever requested from HF).
from huggingface_hub import HfApi, create_repo

if PUBLISH_REPO and HF_TOKEN:
    assert not man.get("reportOnly"), \
        "refusing to publish a report-only export (it would abstain on everything). Re-run step 6."
    create_repo(PUBLISH_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(repo_id=PUBLISH_REPO, repo_type="model", folder_path=f"{OUT_DIR}/staging",
                      commit_message=f"kasauti: {man['modelVersion']}")
    base_url = f"https://huggingface.co/{PUBLISH_REPO}/resolve/main"
    print("published", man["modelVersion"], "→", base_url)
    print(f"\nNEXT_PUBLIC_LAYA_MODEL_BASE={base_url}")
else:
    print("skipped publish (set PUBLISH_REPO in step 2 and add the HF_TOKEN secret to enable)")
    print("the export is still available locally at", KEEP)
""", "publish-to-a-hugging-face-m-24"))

    # ------------------------------------------------------------------ integrate
    cells.append(_cell(MD, r"""
## Step 11 — integrate the export into Kasauti, step by step

The runtime is already built for exactly this artifact layout: **no application code changes** are
needed to go from the bundled dev model to your fine-tune. The same steps with more context are in
`docs/kaggle-training-and-integration.md` — keep that file open while doing this.

**1. Publish** (step 10) or copy `out-export/*` anywhere the browser can `fetch()`.

**2. Point the build at it.** The value is inlined at build time, so it must exist *before* the
build. Vercel → Project → Settings → Environment Variables:

```
NEXT_PUBLIC_LAYA_MODEL_BASE = https://huggingface.co/<user>/kasauti-laya-multilingual-onnx/resolve/main
```

Leave `NEXT_PUBLIC_ORT_BASE` unset to use the pinned CDN default, or set it to a self-hosted copy of
`onnxruntime-web@1.23.0/dist/` if you must avoid third-party origins. Redeploy.

**3. Verify locally first** (same commands the repo's own verification uses):

```bash
npm run typecheck && npm test && npm run build
NEXT_PUBLIC_LAYA_MODEL_BASE=$PWD/public/dev-model npm start   # or `npm run dev`
node tests/e2e-devmode.mjs http://localhost:3000              # Worker + download + decode
```

**4. Verify in the browser** — `/dashboard` → *On-Device Engine*:

* **model version** reads `kasauti-laya-multilingual@<sha8>` — your export, not `tiny-dev`;
* **backend** is `webgpu`, or `wasm-st` / `wasm-mt` where WebGPU is absent (both are supported; the
  page names which one is live);
* the **manifest** panel shows your fitted temperatures and abstention cuts, and *Artefacts verified*
  names the file the browser actually downloaded and hashed;
* **Eval gates** lists your numbers against the thresholds — every row green;
* **question-bank drift** is empty (the manifest's `banks.sha256` equals the compiled-in
  `BANKS_SHA256`);
* press **Build sequence**: the sequence length and `[MASK]` marker count must match what you
  trained on. This is the browser's own tensor builder being compared with the trainer's.

**5. Check the gate behaviour.** Run a known scam through the Checker. A confident answer acts; an
unsettled one shows *abstained* and the tier line says the verdict came from the deterministic
layer. If *everything* abstains, either the export is `reportOnly` (see step 8) or the fitted
thresholds are too aggressive — raise `ABSTAIN_TARGET_ERROR` and re-run step 6.

**6. Transcripts.** `/dashboard#lens` → *Transcript / call notes*: a pasted call transcript is cut
into overlapping windows, read per window and aggregated, with the dropped-window count shown.
Nothing about that needs a retrain.

**7. Offline behaviour.** The first load caches the graphs and tokenizer in the browser's Cache API;
afterwards a check works with the tab offline. `Save-Data` / 2G defers the download by design, and
the Engine page says so.

### Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `CUDA out of memory` in step 6 | microbatch × seq-len 1024 | halve `MICRO_BATCH`, double `GRAD_ACCUM` |
| pipeline exits `2` | accuracy gates failed | more `VARIANTS`/`EPOCHS`, or narrower `TASKS` |
| exit `3` | INT8 top-1 agreement < 0.95 | re-export with `--block-size 64`, keep `--quantize-embedding` |
| exit `4` | reference client ≠ our export | read `reference_check.json`; the failures name the question and both answers |
| exit `5` | the pruned tokenizer re-segmented > 0.5% of the corpus | widen the corpus (more `VARIANTS`); `--allow-tokenizer-drift` only knowingly |
| step 9 says the check was **skipped** | `laya` (or its `onnx` extra) missing | `%pip install "laya[onnx]>=0.3.26"` |
| browser says *no calibrated export* | manifest 404 | wrong `NEXT_PUBLIC_LAYA_MODEL_BASE`, or the repo is private |
| browser says *report-only* | this export did not clear the gates, or was built with `--profile smoke` | read the *Eval gates* panel; the manifest is telling the truth, not failing |
| browser says *bank drift* | trained on other wording | regenerate `src/lib/banks.ts` from the same `ml/banks.json`, then retrain |
| everything abstains | thresholds fitted to a hard split | re-run with `--abstain-target-error 0.25` |
| tokenizer mismatch | graph/tokenizer from different runs | always ship the whole `staging/` directory together |

### Re-running with more data

Coverage lives in `ml/banks.json`: add surface variants and new templates, not new *kinds* of
question. Keep `choice` questions under ~20 options (Laya's accuracy falls off sharply above that
unless `--head-max-len` is raised) and keep questions flat — the model answers one typed question at
a time, so multi-label questions must be decomposed into independent `noul` calls.

Adding a bank means adding its `templates` to `ml/kasauti_ml/datagen.py` (`TEMPLATES_BY_TASK`), then
re-running this notebook. `python -m ml.scripts.gen_banks_ts` regenerates the browser mirror, the
manifest sha256 changes with it, and that is how the runtime notices drift.

### Where the questions and answers come from (the honest limits)

* Base checkpoints are near chance on typed decisions zero-shot — the accuracy comes from this
  fine-tune, which is why the gates exist.
* The model ships **over-confident**; the per-(type, option-count) temperature fit is what makes the
  confidence numbers mean something. Never ship an export without calibration.
* `noul` can latch onto its own option labels on the English checkpoint; the multilingual checkpoint
  is much less prone, and the banks phrase `noul` questions as single yes/no statements for exactly
  this reason. Check `noul` answers on your own data.
* `score` is the weakest primitive. Kasauti uses it only where an ordinal answer is genuinely
  needed.
* Long documents: `laya-multilingual` reads up to 8192 tokens with `--max-len 8192`, but the
  measured reliability degrades past ~4000 tokens. Kasauti windows long transcripts instead of
  sending them whole.
""", "step-11-integrate-the-export-int-25"))

    # ------------------------------------------------------------------ roadmap
    cells.append(_cell(MD, r"""
## Roadmap — from this notebook to the live site

| # | step | command / where | what proves it worked |
|---|---|---|---|
| 1 | smoke run (step 5) | this notebook | `smoke/summary.json` exists, reference check passed |
| 2 | fine-tune (step 6) | this notebook, 2×T4 | exit 0 and `out/summary.json.base.revisionResolved` non-null |
| 3 | read the gates (step 7) | this notebook | every blocking gate green; `reportOnly` false |
| 4 | verify artifacts (step 8) | this notebook | hashes match, tokenizer replays exactly, both graphs inside the published tolerance (INT8 trainer agreement ≥ 0.95) |
| 5 | reference cross-check (step 9) | this notebook or `python tests/test_reference_parity.py` | mirrors + calibration + reference client all PASS |
| 6 | publish (step 10) | HF **model** repo | `manifest.json` reachable at `<base>/manifest.json` |
| 7 | point Vercel at it | `NEXT_PUBLIC_LAYA_MODEL_BASE` env var, then redeploy | `/dashboard#engine` shows your model version |
| 8 | browser E2E | `node tests/e2e-devmode.mjs http://localhost:3000` | Worker reaches `ready`; decode returns calibrated answers |
| 9 | real-device check | a mid-range Android phone, Chrome | first load downloads once, work offline afterwards, p50 latency acceptable |
| 10 | tune the question budget | `src/lib/capability.ts` profiles | follow-up count matches what the device can actually run |

Steps 1–5 are cheap and catch almost everything. Step 9 is the one that cannot be done from a
notebook: `onnxruntime-web` throughput on real phones is the single unknown in the whole design, and
the answer decides how many follow-up questions the adaptive tree is allowed to ask.
""", "roadmap-from-this-notebook-to-th-26"))

    nb: Dict[str, Any] = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isGpuEnabled": True,
                       "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="generate the Kaggle fine-tune notebook")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    nb = build()
    text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    if args.check:
        current = open(args.out, encoding="utf-8").read() if os.path.exists(args.out) else ""
        if current != text:
            print(f"[gen_notebook] STALE: {os.path.relpath(args.out, REPO)}", file=sys.stderr)
            return 1
        print("[gen_notebook] up to date")
        return 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"[gen_notebook] wrote {os.path.relpath(args.out, REPO)} "
          f"({len(nb['cells'])} cells, {n_code} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
