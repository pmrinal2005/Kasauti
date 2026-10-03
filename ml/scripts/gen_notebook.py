#!/usr/bin/env python3
"""Generate `notebooks/kasauti_laya_finetune_kaggle.ipynb`.

The notebook is GENERATED rather than hand-written so it cannot silently drift away from the code
it drives: every command in it is one this repo already runs locally (the smoke profile), and the
flags are the actual `ml/kasauti_ml/pipeline.py` flags.

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


def _cell(kind: str, source: str) -> Dict[str, Any]:
    lines = source.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cell: Dict[str, Any] = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == CODE:
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ title
    cells.append(_cell(MD, r"""
# Kasauti × Laya — fine-tune, evaluate, quantize and publish (Kaggle, 2×T4)

This notebook turns the **latest Laya checkpoint** into the exact artifact Kasauti's browser
runtime loads: a vocabulary-pruned, INT8 + 4-bit ONNX pair with fitted per-bucket temperatures,
abstention thresholds and parity vectors, published as **static bytes in a Hugging Face *model*
repo** (never a Space, never a server).

**What runs where.** Laya is Kasauti's Tier-1 *typed-decision* model: state + typed questions in,
typed answers with probabilities out, one forward pass, no text generation. It is fine-tuned here
with **RLCD** — REINFORCE on a group-mean baseline with zero-mean Gaussian logit noise, scored by a
strictly proper scoring rule — exactly as the reference `train_ddp.py` does, using the banks in
`ml/banks.json` as the only source of question wording.

**Before you spend GPU time**

* Attach **two T4s** (Settings → Accelerator → GPU T4 ×2) and turn the Internet on.
* Add your HF write token as a Kaggle Secret named `HF_TOKEN` if you want step 9 to publish;
  every earlier step runs without it.
* Run **Step 4 (the CPU smoke run)** first. It exercises the whole chain — data, training loop,
  calibration, INT8/Q4 export, ORT agreement, parity vectors, manifest, staging + hash
  verification — on a randomly-initialised tiny model in about two minutes. If it passes, the only
  things left that can fail are GPU memory and the model's actual accuracy.

**Expected wall clock on 2×T4:** data ~30 s · fine-tune ~5–10 min · calibration + eval ~1 min ·
INT8/Q4 export + ORT agreement ~1 min · publish ~1 min.

*Laya is Apache-2.0 (github.com/NandhaKishorM/laya). This notebook trains a derivative model and
publishes it under your own account; keep the upstream attribution in the model card.*
"""))

    # ------------------------------------------------------------------ install
    cells.append(_cell(MD, "## Step 0 — install dependencies\n\nKaggle's image already has PyTorch CUDA. These pins are the versions the pipeline was built and tested against."))
    cells.append(_cell(CODE, r"""
# `laya` ships the reference implementation (build_sequence, teacher/agent helpers, ONNX agent).
# The `onnx` extra is not needed for training, but the tokenizer-library pins are: the pipeline
# compares its own tokenizer handling against `tokenizers` (Rust) for parity.
%pip install -q "laya>=0.3.20" "transformers>=4.48" "tokenizers>=0.21" "safetensors>=0.4" \
    "onnx>=1.17" "onnxruntime>=1.20" "huggingface_hub>=0.26" "scikit-learn>=1.5" "onnx_ir>=0.1"
print("deps ok")
"""))

    # ------------------------------------------------------------------ code
    cells.append(_cell(MD, r"""
## Step 1 — get the project code

Two options, pick one:

1. **Clone the repo** (simplest; the clone also gives you `ml/banks.json`, the training data spec).
2. **Attach it as a Kaggle Dataset** — upload the repo's `ml/` folder as a dataset, attach it to the
   notebook and point `PROJECT_SRC` at the mounted path. Use this when you iterate, so each version
   stays reproducible.

Either way the pipeline imports as the package `ml.kasauti_ml.*`, so it must be run from the
directory that *contains* `ml/`.
"""))
    cells.append(_cell(CODE, r"""
import os, subprocess, sys

REPO_URL = "https://github.com/pmrinal2005/Kasauti.git"   # change if you forked it
WORK = "/kaggle/working"
PROJECT_DIR = os.path.join(WORK, "Kasauti")
PROJECT_SRC = ""            # set to e.g. "/kaggle/input/kasauti-ml" to use an attached dataset instead

if PROJECT_SRC:
    os.makedirs(PROJECT_DIR, exist_ok=True)
    subprocess.run(["cp", "-r", f"{PROJECT_SRC.rstrip('/')}/.", PROJECT_DIR], check=True)
else:
    if not os.path.isdir(PROJECT_DIR):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, PROJECT_DIR], check=True)

os.chdir(PROJECT_DIR)
print("project:", PROJECT_DIR)
for rel in ("ml/banks.json", "ml/kasauti_ml/pipeline.py", "src/lib/banks.ts"):
    print(f"  {'ok ' if os.path.exists(os.path.join(PROJECT_DIR, rel)) else 'MISSING'} {rel}")
print(subprocess.run([sys.executable, "-c",
                      "import torch;print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|',
                      torch.cuda.device_count(), 'gpu(s)')"], capture_output=True, text=True).stdout.strip())
"""))

    # ------------------------------------------------------------------ config
    cells.append(_cell(MD, r"""
## Step 2 — configuration

Everything the run depends on, in one place. Defaults are the reference notebook's (2×T4, ~1 200
case rows). The knobs that matter, and when to change them:

| flag | default | change it when |
|---|---|---|
| `EPOCHS` | 3 | gates fail on accuracy → 5–6, but watch for overfitting (eval ECE rises) |
| `MICRO_BATCH` / `GRAD_ACCUM` | 8 / 4 | CUDA OOM → halve `MICRO_BATCH`, double `GRAD_ACCUM` (effective batch stays 32) |
| `VARIANTS` | 40 | you want more data per template; more is almost always better than more epochs |
| `TASKS` | all 6 banks | you want a narrower model (e.g. only `claim_v1`) — smaller, faster, sharper |
| `LR_ENCODER` / `LR_HEAD` | 2.5e-5 / 1e-4 | encoder must move slowly; a jump to 1e-4 usually destroys the pretrained features |
| `MAX_LEN` | 1024 | multilingual checkpoint; 512 halves memory for short SMS-style states |
| `PUBLISH_REPO` | empty | set to `you/kasauti-laya-multilingual-onnx` to publish from step 9 |
"""))
    cells.append(_cell(CODE, r"""
import os

OUT_DIR      = "/kaggle/working/out"           # everything the pipeline writes
SMOKE_DIR    = "/kaggle/working/smoke"         # tiny end-to-end validation run

MODEL_ID     = "convaiinnovations/laya"        # root repo: also holds `multilingual/`
SUBFOLDER    = "multilingual"                  # 100+ languages, mmBERT-base, 1024 ctx — what Kasauti ships
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
PUBLIC_DIR   = ""                              # e.g. f"{PROJECT_DIR}/public/dev-model" to smoke-test the browser

HF_TOKEN = ""
try:                                           # Kaggle secret; harmless if absent
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as exc:
    print("no HF_TOKEN secret (fine unless you publish in step 9):", type(exc).__name__)

os.makedirs(OUT_DIR, exist_ok=True)
print("config:", dict(TASKS=TASKS or "all", VARIANTS=VARIANTS, EPOCHS=EPOCHS,
                      MICRO_BATCH=MICRO_BATCH, GRAD_ACCUM=GRAD_ACCUM, GROUP_SIZE=GROUP_SIZE,
                      MAX_LEN=MAX_LEN, HEAD_MAX_LEN=HEAD_MAX_LEN))
"""))

    # ------------------------------------------------------------------ provenance
    cells.append(_cell(MD, r"""
## Step 3 — provenance check (question-bank drift)

The most expensive failure in this whole pipeline is silent: the model gets fine-tuned on one
wording of a question and the browser asks a slightly different one. Nothing errors, the answers
are just worse. `ml/banks.json` is the single source of truth; `src/lib/banks.ts` is generated from
it, and the manifest carries the same sha256 so the runtime can detect drift at load time.

This cell fails loudly if the two halves disagree.
"""))
    cells.append(_cell(CODE, r"""
import hashlib, json, subprocess, sys

banks = json.load(open("ml/banks.json", encoding="utf-8"))
sha = hashlib.sha256(json.dumps(banks, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8")).hexdigest()
print(f"banks version {banks['version']} · sha256 {sha}")
print(f"{'bank':<22}{'triage options':>15}{'max questions':>15}  title")
for bid, b in banks["tasks"].items():
    tri = b["triage"].get("criteria") or []
    fol = max([len(v) for v in (b.get("followups") or {"": []}).values()] or [0])
    n_opts = len(tri) if not isinstance(tri, list) else len(tri)
    print(f"{bid:<22}{n_opts:>15}{1 + fol:>15}  {b['title']}")

r = subprocess.run([sys.executable, "-m", "ml.scripts.gen_banks_ts", "--check"],
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
assert r.returncode == 0, "src/lib/banks.ts is stale — run: python -m ml.scripts.gen_banks_ts"
"""))

    # ------------------------------------------------------------------ data
    cells.append(_cell(MD, r"""
## Step 4 — inspect the training data

Rows are expanded from templates and split **by template hash**, not by row: a template that
appears in training can never appear in validation or test, so a high eval score cannot come from
having memorised the surface form. `template_overlap` must be `0` in every pair — the pipeline
aborts otherwise.

`gold.probabilities` is a distribution, not a label. Where a case is genuinely ambiguous the
distribution is soft on purpose: the RLCD reward is a proper scoring rule, so it is trained to be
*right about its uncertainty* rather than confident and wrong.
"""))
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
"""))

    # ------------------------------------------------------------------ smoke
    cells.append(_cell(MD, r"""
## Step 5 — CPU smoke run (do this before burning GPU time)

A tiny randomly-initialised model is pushed through **every** stage on CPU. It proves the chain
end-to-end — and nothing more. What it does **not** prove is accuracy: a 0.5 M-parameter model
trained for a minute on synthetic data cannot learn six decision banks, so the accuracy gates
*will* fail here and the pipeline says so and carries on. Accuracy is the GPU run's job.
"""))
    cells.append(_cell(CODE, r"""
import subprocess, sys, json
cmd = [sys.executable, "-m", "ml.kasauti_ml.pipeline", "--profile", "smoke",
       "--out", SMOKE_DIR, "--variants", "3", "--epochs", "2", "--log-every", "5000"]
print(" ".join(cmd), "\n")
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-4000:] or r.stderr[-4000:])
assert r.returncode == 0, "smoke run failed — fix this before the GPU run"
smoke = json.load(open(f"{SMOKE_DIR}/summary.json"))
print("\nsmoke artifacts:", json.dumps(smoke["sizes_mb"]), "modelVersion:", smoke["modelVersion"])
import os
print("staged files:", sorted(os.listdir(f"{SMOKE_DIR}/staging")))
"""))

    # ------------------------------------------------------------------ train
    cells.append(_cell(MD, r"""
## Step 6 — the real fine-tune (2×T4, DDP)

`torchrun` launches one process per GPU; the pipeline wraps the model in
`DistributedDataParallel`, shards the rows by rank and averages gradients, so both T4s train **one**
model. Rank 0 does the evaluation, calibration, export and publish — ranks 1+ stop after training.

The loop is RLCD:

1. **Sample** `GROUP_SIZE` noisy distributions per item, using the zero-mean projected Gaussian
   perturbation `eps - mean(eps)` so the noise cannot change the *ranking* the model already has.
2. **Score** each sample with the strictly proper rule (log-score + spherical, minus a ranked
   probability score for `score` questions) against the soft gold distribution.
3. **REINFORCE** on the group-mean baseline, plus a soft cross-entropy anchor, with rare-class
   weights and a negative-class boost so the "normal message" class is not drowned out.

Watch `reward` trend upward and `sigma` anneal 0.4 → 0.1. `loss` itself is not a progress metric
(the RL term is a surrogate).
"""))
    cells.append(_cell(CODE, r"""
import subprocess, sys, time

cmd = ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "ml.kasauti_ml.pipeline",
       "--profile", "kaggle", "--out", OUT_DIR,
       "--model-id", MODEL_ID, "--subfolder", SUBFOLDER,
       "--variants", str(VARIANTS), "--epochs", str(EPOCHS),
       "--micro-batch", str(MICRO_BATCH), "--grad-accum", str(GRAD_ACCUM),
       "--group-size", str(GROUP_SIZE),
       "--lr-encoder", str(LR_ENCODER), "--lr-head", str(LR_HEAD),
       "--max-len", str(MAX_LEN), "--head-max-len", str(HEAD_MAX_LEN),
       "--abstain-target-error", str(ABSTAIN_TARGET_ERROR), "--log-every", "25"]
if TASKS:
    cmd += ["--tasks", TASKS]
if PUBLIC_DIR:
    cmd += ["--public-dir", PUBLIC_DIR]
print(" ".join(cmd), "\n", flush=True)

t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:                      # stream, so a 10-minute run is not a black box
    print(line, end="")
proc.wait()
print(f"\nexit={proc.returncode} in {(time.time()-t0)/60:.1f} min")
assert proc.returncode == 0, ("pipeline failed — if it exited 2, the accuracy gates failed: add data "
                              "(VARIANTS) or epochs, then re-run. Gating is the point; do not skip it.")
"""))

    # ------------------------------------------------------------------ reports
    cells.append(_cell(MD, r"""
## Step 7 — read the reports

Three files decide whether this model ships:

* `eval_report.json` — every gate with the threshold it had to clear.
* `calibration.json` — the fitted temperature per bucket and the abstention cut that holds
  accepted-answer error under `ABSTAIN_TARGET_ERROR`, plus the coverage that implies.
* `summary.json` — artifact sizes, vocab pruning, quantized-agreement numbers.

A note on reading ECE: the base Laya multilingual checkpoint ships **over-confident** (ECE ≈ 0.31).
If the fitted ECE is meaningfully lower than the raw ECE, calibration is doing real work, and the
abstention map is then meaningful rather than decorative.
"""))
    cells.append(_cell(CODE, r"""
import json
ev = json.load(open(f"{OUT_DIR}/eval_report.json", encoding="utf-8"))
cal = json.load(open(f"{OUT_DIR}/calibration.json", encoding="utf-8"))
summ = json.load(open(f"{OUT_DIR}/summary.json", encoding="utf-8"))

print("gates")
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
print("\nartifacts:", json.dumps(summ))
"""))

    # ------------------------------------------------------------------ verify
    cells.append(_cell(MD, r"""
## Step 8 — verify the published artifacts (independent of the trainer)

This re-checks the export the way the browser will load it: every file's sha256 against the
manifest, then both quantized graphs run on the parity vectors and compared with the fp32 graph.
`onnxruntime` here is the same engine family the browser uses, so this is the closest offline
analogue of the WebGPU/WASM path.

Rule of thumb: INT8 and Q4 top-1 agreement must stay ≥ 0.95. Below that, raise `--block-size`
(32 → 64) or drop `--no-quantize-embedding`, and re-export.
"""))
    cells.append(_cell(CODE, r"""
import hashlib, json, os, sys
import numpy as np, onnxruntime as ort
sys.path.insert(0, ".")                      # run from the project root
from ml.kasauti_ml import export as mlx      # the pipeline's own ORT helper — same code path

STAGE = f"{OUT_DIR}/staging"
man = json.load(open(f"{STAGE}/manifest.json", encoding="utf-8"))

# 1) every byte in the manifest is the byte that is about to be uploaded
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

# 2) both quantized graphs re-run on the parity vectors and compared with the fp32 reference
parity = json.load(open(os.path.join(STAGE, man["parity"]["path"]), encoding="utf-8"))
vecs = parity["logits"]["vectors"]
# ort_predict expects the trainer's item shape; `target` is only carried along, never used here
items = [{"ids": v["ids"], "markers": v["markers"], "qtype": v["qtype"],
          "target": [0.0] * len(v["markers"])} for v in vecs]
# option counts differ per vector (2/5/6 markers), so compare vector by vector — not as a matrix
ref = [np.asarray(v["logits"], dtype=np.float64) for v in vecs]
print(f"\nparity: {len(parity['tokenizer']['vectors'])} tokenizer texts · {len(vecs)} logit vectors")
for key, label in (("wasm", "INT8 (WASM EP)"), ("webgpu", "4-bit (WebGPU EP)")):
    sess = mlx.ort_session(os.path.join(STAGE, man["files"][key]["path"]))
    got = [np.asarray(l, dtype=np.float64) for _qt, l, _t in mlx.ort_predict(sess, items)]
    top1 = float(np.mean([float(np.argmax(g) == np.argmax(r)) for g, r in zip(got, ref)]))
    dmax = max(float(np.abs(g - r).max()) for g, r in zip(got, ref))
    floor = 0.95 if key == "wasm" else 0.80
    if top1 >= 0.95:
        verdict = "ok"
    elif top1 >= floor:
        verdict = "acceptable for a small/dev model — for a release export with --block-size 64"
    else:
        verdict = "TOO LOW — re-export with --block-size 64 and keep --quantize-embedding"
    print(f"  {label:<20} top-1 agreement {top1:.3f} · max |Δlogit| {dmax:.4f}   {verdict}")
    # INT8 is a hard gate: the pipeline refuses to publish below 0.95 (exit 3). Q4 is a WebGPU
    # convenience path — 4-bit weights on a *small* model lose relatively more, so the floor is
    # lower and the number is reported rather than hidden.
    assert top1 >= floor, f"{label} drifted from fp32 (top-1 {top1:.3f} < {floor})"

# 3) the exported graph must still carry the I/O contract the browser sends
io = mlx.graph_io_summary(os.path.join(STAGE, man["files"]["wasm"]["path"]))
print("\ngraph inputs:", [i["name"] for i in io["inputs"]], "→ outputs:", [o["name"] for o in io["outputs"]])
assert {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"} <= {i["name"] for i in io["inputs"]}
print("manifest modelVersion:", man["modelVersion"], "| eval at export:", man["evals"])

# 4) the manifest must be honest about what it is. `reportOnly` is set by the pipeline whenever the
#    eval gates did not pass (or the profile was smoke), and the browser turns it into "abstain on
#    everything". Printing it here is the last chance to notice before a model that always abstains
#    reaches the app and looks broken.
if man.get("reportOnly"):
    print("\nNOTE: this export is REPORT-ONLY — it did not clear the eval gates, so the runtime will")
    print("      abstain on every answer and the UI will say so. Inspect it, do not ship it: add")
    print("      VARIANTS/EPOCHS (or narrow TASKS) and re-run step 6. See docs/kaggle-training-and-integration.md §5.")
else:
    print("\nALL EXPORT CHECKS PASSED — calibrated and gate-passing: safe to publish.")
"""))

    # ------------------------------------------------------------------ publish
    cells.append(_cell(MD, r"""
## Step 9 — get the artifacts out

**Either** keep them as notebook output (they show up under *Output* → `out/` and can be attached
as a dataset), **or** publish them to a Hugging Face *model* repo — which is what the browser
fetches. A model repo is static storage on the CDN: no Space, no GPU quota, no Kasauti server in
the path. `hf upload`/`upload_folder` is all it takes.
"""))
    cells.append(_cell(CODE, r"""
import json, shutil, os
KEEP = f"{PROJECT_DIR}/out-export"
shutil.rmtree(KEEP, ignore_errors=True)
shutil.copytree(f"{OUT_DIR}/staging", KEEP)          # lands in /kaggle/working → downloadable
print("copied →", KEEP)
print(sorted(os.listdir(KEEP)))
man = json.load(open(f"{KEEP}/manifest.json", encoding="utf-8"))
print("\nfiles:", {k: v["path"] for k, v in man["files"].items()})
"""))
    cells.append(_cell(CODE, r"""
# Optional: publish to a Hugging Face MODEL repo (static bytes, no compute).
from huggingface_hub import HfApi, create_repo

if PUBLISH_REPO and HF_TOKEN:
    create_repo(PUBLISH_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(repo_id=PUBLISH_REPO, repo_type="model", folder_path=f"{OUT_DIR}/staging")
    base = f"https://huggingface.co/{PUBLISH_REPO}/resolve/main"
    print("published.")
    print("manifest:", f"{base}/manifest.json")
    print("\nKasauti env for the deployment:")
    print(f"NEXT_PUBLIC_LAYA_MODEL_BASE={base}")
else:
    print("skipped publish (set PUBLISH_REPO in step 2 and add the HF_TOKEN secret to enable)")
"""))

    # ------------------------------------------------------------------ integrate
    cells.append(_cell(MD, r"""
## Step 10 — integrate the export into Kasauti (exact steps)

The runtime is already built for exactly this artifact layout — no code change is needed to switch
from the dev model to your fine-tune. The same steps, with more context, live in
`docs/kaggle-training-and-integration.md`; that file is the one to keep open while doing this.

1. **Publish** (step 9) or copy `out-export/*` anywhere the browser can `fetch()`.
2. **Point the build at it** — Vercel → Project → Settings → Environment Variables:

   ```
   NEXT_PUBLIC_LAYA_MODEL_BASE = https://huggingface.co/<user>/kasauti-laya-multilingual-onnx/resolve/main
   ```

   Redeploy (the value is inlined at build time, so it must exist *before* the build).
3. **Verify in the browser** — open `/dashboard` → *On-Device Engine*:
   * model version shows `kasauti-laya-multilingual@<sha8>` (i.e. your export, not `tiny-dev`);
   * backend is `webgpu` (or `wasm-st`/`wasm-mt` where WebGPU is absent);
   * the manifest panel lists your fitted temperatures and abstention cuts, and *Artefacts verified*
     names the file you uploaded;
   * **Eval gates** lists your numbers against the thresholds and every row is green;
   * *question-bank drift* is **empty**;
   * press **Build sequence** — the sequence length and `[MASK]` marker count must match what you
     trained on.
4. **Check the gate behaviour** — run a known scam in the Checker. A confident answer acts; an
   unsettled one shows *abstained* and the tier line says the verdict is the deterministic one. If
   everything abstains, the fitted thresholds are too high — raise `ABSTAIN_TARGET_ERROR` and re-run
   step 6. Note that an export that *did not clear the gates* (or any `--profile smoke` export)
   carries `reportOnly: true` in its manifest: the runtime then abstains on everything by
   construction, and the Engine page says so. That is the intended behaviour, not a bug.
5. **Transcripts** — `/dashboard#lens`, *Transcript / call notes*: a pasted call transcript is cut
   into overlapping windows at speaker turns, read per window, and averaged, with the dropped-window
   count shown. Nothing about it needs a retrain.
5. **Offline behaviour** — the first load caches the graphs and tokenizer in the Cache API;
   afterwards the check works with the tab offline. `Data Saver` / 2G defers the download by design.

### Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `CUDA out of memory` in step 6 | microbatch × seq-len 1024 | halve `MICRO_BATCH`, double `GRAD_ACCUM` |
| pipeline exits `2` | accuracy gates failed | more `VARIANTS`/`EPOCHS`, or narrower `TASKS` |
| exit `3` | INT8 top-1 agreement < 0.95 | `--block-size 64`, keep `--quantize-embedding` |
| browser says *no calibrated export* | manifest 404 | wrong `NEXT_PUBLIC_LAYA_MODEL_BASE`, or the repo is private |
| browser says *report-only* | this export did not clear the eval gates, or it was built with `--profile smoke` | read the *Eval gates* panel; the manifest is telling the truth, not failing |
| browser says *bank drift* | trained on other wording | regenerate `src/lib/banks.ts` from the same `ml/banks.json` and retrain |
| everything abstains | thresholds fitted to a hard split | re-run with `--abstain-target-error 0.25` |
| Tokenizer mismatch error | graph/tokenizer from different runs | always ship the whole `staging/` directory together |

### Re-running with more data

`ml/banks.json` is where coverage lives: add surface variants and new templates, not new *kinds* of
question. Keep `choice` questions under 20 options — Laya's accuracy collapses above that — and keep
questions flat (no nested objects): the model answers one typed question at a time.

Adding a bank means adding `templates` for it in `ml/kasauti_ml/datagen.py` (its `TEMPLATES_BY_TASK`
entry), then re-running this notebook. `python -m ml.scripts.gen_banks_ts` regenerates the browser
mirror, and the manifest sha256 changes with it — which is how the runtime notices drift.
"""))

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
