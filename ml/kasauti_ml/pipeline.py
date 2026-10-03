"""End-to-end pipeline: banks → data → RLCD train → calibrate → gate → export → parity → publish.

Two profiles:

``smoke``   CPU, ~1–2 minutes, tiny randomly-initialised model + synthetic tokenizer. Proves the
            whole chain runs, produces a **dev model** in ``public/dev-model`` that the browser E2E
            test loads, and lets the client be developed and tested before any GPU run exists.
``kaggle``  the real thing: downloads ``convaiinnovations/laya`` (multilingual/mmBERT-base),
            fine-tunes with RLCD on the generated multi-task data, calibrates, gates, exports the
            INT8 and 4-bit browser builds and publishes them.

Run from the repo root:

    python -m ml.kasauti_ml.pipeline --profile smoke --out ml/out/smoke --public-dir public/dev-model
    python -m ml.kasauti_ml.pipeline --profile kaggle --out /kaggle/working/kasauti \
        --publish-repo <hf-user>/kasauti-laya-onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(HERE)
DEFAULT_BANKS = os.path.join(ML_ROOT, "banks.json")

from . import calibrate as calib  # noqa: E402
from . import core, datagen, evals, export, parity, publish, train as trainer  # noqa: E402
from .tok import FastTokenizerAdapter, build_tiny_tokenizer, load_fast_tokenizer, load_tiny_tokenizer  # noqa: E402


def log(msg: str) -> None:
    print(f"[kasauti-ml] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kasauti × Laya fine-tune / export pipeline")
    p.add_argument("--profile", choices=["smoke", "kaggle"], default="smoke")
    p.add_argument("--banks", default=DEFAULT_BANKS)
    p.add_argument("--out", default="ml/out/run")
    p.add_argument("--public-dir", default=None,
                   help="copy the staged model here (e.g. public/dev-model) for the browser E2E")
    p.add_argument("--model-id", default="convaiinnovations/laya")
    p.add_argument("--subfolder", default="multilingual",
                   help="checkpoint subfolder: multilingual (100+ languages) is what Kasauti ships")
    p.add_argument("--model-dir", default=None, help="local checkpoint dir (skips the Hub download)")
    p.add_argument("--resume-from", default=None, help="start from an already fine-tuned dir")
    p.add_argument("--variants", type=int, default=None, help="surface variants per template")
    # None = "let the profile decide"; an explicit flag always wins (see PROFILES below)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--lr-encoder", type=float, default=None)
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--sigma-start", type=float, default=0.4)
    p.add_argument("--sigma-end", type=float, default=0.1)
    p.add_argument("--w-sph", type=float, default=0.75)
    p.add_argument("--w-rps", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--negative-boost", type=float, default=1.6)
    p.add_argument("--calib-max", type=int, default=400)
    p.add_argument("--abstain-target-error", type=float, default=0.15)
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--batch-eval", type=int, default=16)
    p.add_argument("--skip-train", action="store_true", help="reuse --resume-from / --model-dir")
    p.add_argument("--export-only", action="store_true", help="skip data/train/eval, just re-export")
    p.add_argument("--quantize-embedding", dest="quantize_embedding", action="store_true", default=True)
    p.add_argument("--no-quantize-embedding", dest="quantize_embedding", action="store_false")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--head-max-len", type=int, default=None)
    p.add_argument("--tasks", default=None, help="comma-separated task subset (default: all)")
    p.add_argument("--publish-repo", default=None, help="HF model repo id to publish to")
    p.add_argument("--hf-token", default=None, help="HF write token (else HF_TOKEN env)")
    p.add_argument("--gates", default=None, help="JSON file overriding eval gate thresholds")
    return p


# Per-profile defaults. `kaggle` numbers mirror the reference notebook (2×T4, ~4-6 min on 1 200
# cases); `smoke` is a ~30 s plumbing test on a randomly-initialised tiny model, and its accuracy
# gates are advisory — they must never be read as a statement about the fine-tune.
PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {"epochs": 6, "micro_batch": 4, "grad_accum": 1, "group_size": 2,
              "lr_encoder": 2e-3, "lr_head": 2e-3, "max_len": 192, "head_max_len": 64,
              "variants": 6},
    "kaggle": {"epochs": 3, "micro_batch": 8, "grad_accum": 4, "group_size": 4,
               "lr_encoder": 2.5e-5, "lr_head": 1e-4, "max_len": 1024, "head_max_len": 256,
               "variants": 40},
}


def apply_profile_defaults(args: Any) -> None:
    """Fill unset hyper-parameters from the profile. Explicit CLI values are never touched."""
    prof = PROFILES[args.profile]
    for key, val in prof.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)


def pick_device(explicit: Optional[str] = None) -> str:
    import torch

    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------------------
# decode / eval helpers
# --------------------------------------------------------------------------------------
def decode_predictions(rows: Sequence[Dict[str, Any]], logits_by_key: Dict[Tuple[str, str], Sequence[float]],
                       temperature: Sequence[float], temp_by_bucket: Dict[str, float],
                       abstain: Dict[str, float]) -> List[Dict[str, Any]]:
    """Turn raw slot logits into the exact answer object the browser produces.

    Order matters and mirrors ``onnx_agent._decode_answers``: softmax over the first k logits with
    the bucket temperature, *then* invert the option permutation, *then* read `answer_confidence`
    (max p) and compare it with the bucket's abstention cut.
    """
    import numpy as np

    out: List[Dict[str, Any]] = []
    for row in rows:
        for qid, q in row["questions"].items():
            key = (row["id"], qid)
            if key not in logits_by_key:
                continue
            qi = core.to_internal(q)
            keys = core.option_keys(qi)
            k = len(keys)
            z = logits_by_key[key][:k]
            bucket = core.temp_bucket(core.QTYPES[q["type"]], k)
            t = temp_by_bucket.get(bucket, temperature[core.QTYPES[q["type"]]])
            p = core.softmax_rows(np.asarray(z).reshape(1, -1), t)[0]
            order = row.get("option_orders", {}).get(qid) or list(range(k))
            p = core.unpermute_probs(p, order)
            conf = float(np.max(p))
            out.append({
                "row_id": row["id"], "qid": qid, "task": row["task"], "lang": row["lang"],
                "type": q["type"],
                "probabilities": {key: float(v) for key, v in zip(keys, p)},
                "confidence": conf,
                "abstained": conf < abstain.get(bucket, abstain.get("default", 0.0)),
                "bucket": bucket,
            })
    return out


# --------------------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import torch

    t_start = time.time()
    apply_profile_defaults(args)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    tasks = datagen.load_tasks(args.banks)
    with open(args.banks, "r", encoding="utf-8") as f:
        banks_raw = json.load(f)
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = {k: v for k, v in tasks.items() if k in wanted}
    log(f"profile={args.profile} device={device} tasks={sorted(tasks)}")

    profile = args.profile
    n_variants = args.variants

    # ---------------------------------------------------------------- data
    templates = [t for task_name in tasks for t in datagen.TEMPLATES_BY_TASK[task_name]]
    rows = datagen.build_rows(tasks, templates, n_variants=n_variants, seed=7,
                             romanized=profile != "smoke")
    stats = datagen.dataset_stats(rows)
    if any(v for v in stats["template_overlap"].values()):
        raise SystemExit(f"template leakage between splits: {stats['template_overlap']}")
    data_path = os.path.join(args.out, "data.jsonl")
    datagen.write_jsonl(rows, data_path)
    log(f"rows={stats['rows']} splits={stats['by_split']} languages={len(stats['by_lang'])} "
        f"templates={stats['templates']} (no template overlap, verified)")
    calib.write_json(stats, os.path.join(args.out, "data_stats.json"))

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    # ---------------------------------------------------------------- tokenizer + model
    cfg: Dict[str, Any] = {"max_len": args.max_len, "head_max_len": args.head_max_len,
                           "head_layers": 2, "act_costs": {"escalate": 1.0}}
    ckpt_dir = args.resume_from or args.model_dir
    if profile == "smoke":
        tok_json = os.path.join(args.out, "tokenizer_full.json")
        tmeta = build_tiny_tokenizer([r["state"] for r in rows], tok_json)
        tok = load_tiny_tokenizer(tok_json)
        cfg.update({"hidden": 64, "layers": 2, "head_layers": 1})
        model = core.build_tiny_model(tok.vocab_size, hidden=cfg["hidden"], layers=cfg["layers"],
                                      head_layers=1, max_len=cfg["max_len"])
        log(f"tiny tokenizer: vocab={tmeta['vocab_size']} merges={tmeta['merges']}; "
            f"tiny model: {sum(p.numel() for p in model.parameters())/1e3:.1f}k params")
    else:
        if not ckpt_dir:
            from huggingface_hub import snapshot_download
            from laya.agent import _fix_tokenizer_config

            log(f"downloading {args.model_id} (subfolder={args.subfolder})…")
            ckpt_dir = snapshot_download(args.model_id, allow_patterns=None)
            sub = os.path.join(ckpt_dir, args.subfolder)
            if os.path.isdir(sub):
                ckpt_dir = sub
            _fix_tokenizer_config(ckpt_dir)
        with open(os.path.join(ckpt_dir, "rl_agent_config.json")) as f:
            run_cfg = json.load(f)
        cfg.update({k: v for k, v in run_cfg.items() if k in ("max_len", "head_max_len", "head_layers",
                                                              "act_costs", "temperature",
                                                              "temperature_by_options", "encoder")})
        if args.max_len:
            cfg["max_len"] = args.max_len
        if args.head_max_len:
            cfg["head_max_len"] = args.head_max_len
        tok = load_fast_tokenizer(ckpt_dir)
        log(f"encoder={cfg.get('encoder')} vocab={tok.vocab_size} max_len={cfg['max_len']} "
            f"head_max_len={cfg['head_max_len']}")
        if args.export_only:
            model = None
        else:
            from safetensors.torch import load_file

            model = core.build_model_from_cfg(cfg, encoder_dir=os.path.join(ckpt_dir, "encoder"))
            state = load_file(os.path.join(ckpt_dir, "model.safetensors"))
            model.load_state_dict(state, strict=True)
            if hasattr(model.encoder, "gradient_checkpointing_enable"):
                model.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            model.head_checkpointing = True
            trainable = sum(p.numel() for p in model.parameters())
            log(f"model loaded: {trainable/1e6:.1f}M params")

    # ---------------------------------------------------------------- train
    report: Dict[str, Any] = {}
    if not args.export_only:
        items = trainer.rows_to_items(tok, rows, tasks, cfg["max_len"], cfg["head_max_len"],
                                      include_splits=["train"])
        log(f"training items: {len(items)}")
        if not items:
            raise SystemExit("no training items — check the banks/splits")
        args.device = device
        report = trainer.train(items, model, tok, cfg, args.out, args)
        log(f"trained in {report['seconds']}s | loss curve {[round(h['avg_loss'],3) for h in report['history']]}")
        calib.write_json(report, os.path.join(args.out, "train_report.json"))

    # ---------------------------------------------------------------- calibrate + gate
    eval_splits = ("val", "test")
    eval_rows = [r for r in rows if r["split"] in eval_splits]
    eval_items = _keyed_eval_items(tok, eval_rows, tasks, cfg, eval_splits)

    if args.export_only:
        metrics: Dict[str, Any] = {}
        calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}, "abstain": {}}
        coverage: Dict[str, Any] = {}
    else:
        args.device = device
        preds_raw = trainer._predict_items(model, tok, eval_items, device, args.batch_eval)  # noqa: SLF001
        recs = calib._records(preds_raw)  # noqa: SLF001
        calibration = calib.fit_temperature_map(recs, compute_ece=True)
        abstain = calib.fit_abstention_thresholds(
            recs, calibration["temperature"], calibration["temperature_by_options"],
            target_error=args.abstain_target_error)
        calibration["abstain"] = abstain
        coverage = calib.coverage_at(recs, calibration["temperature"],
                                     calibration["temperature_by_options"], abstain)
        calibration["coverage"] = coverage
        calib.write_json(calibration, os.path.join(args.out, "calibration.json"))
        log(f"temperatures type={ [round(t,3) for t in calibration['temperature']] } "
            f"buckets={ {k: round(v,3) for k,v in calibration['temperature_by_options'].items()} }")
        log(f"abstention={ {k: round(v,3) for k,v in abstain.items()} } coverage={coverage}")

        # logits keyed by (row_id, qid)
        logits_by_key = {}
        for it, (qt, logits, _t) in zip(eval_items, preds_raw):
            logits_by_key[(it["row_id"], it["qid"])] = logits
        decoded = decode_predictions(eval_rows, logits_by_key, calibration["temperature"],
                                     calibration["temperature_by_options"], abstain)
        metrics = evals.evaluate_predictions(eval_rows, decoded)
        gates_override = {}
        if args.gates:
            with open(args.gates) as f:
                gates_override = json.load(f)
        gate_report = evals.run_gates(metrics, gates_override,
                                      extra={"coverage": coverage.get("coverage", 0.0)})
        log(f"eval accuracy={metrics['accuracy']} ece={metrics['ece']} "
            f"noul={metrics['noul_accuracy']} choice={metrics['choice_accuracy']}")
        calib.write_json({"metrics": metrics, "gates": gate_report},
                         os.path.join(args.out, "eval_report.json"))
        for c in gate_report["checks"]:
            log(f"  gate {c['name']:<34} value={c['value']} threshold={c['threshold']} "
                f"{'PASS' if c['ok'] else 'FAIL'}")
        gates_passed = bool(gate_report["passed"])
        if not gates_passed:
            log("GATES FAILED — publishing nothing. Fix the fine-tune and re-run.")
            if profile != "smoke":
                raise SystemExit(2)
            log("(smoke profile: gates are advisory, so the export is staged as REPORT-ONLY)")
        else:
            log("all eval gates passed")

    # ---------------------------------------------------------------- export
    exp_dir = os.path.join(args.out, "export")
    os.makedirs(exp_dir, exist_ok=True)

    # 1. prune the vocabulary against the corpus the app actually sees
    corpus = [r["state"] for r in rows] + list(parity.PARITY_TEXTS)
    corpus += [t["instructions"] for task in tasks.values() for t in [task["triage"]]]
    for task in tasks.values():
        corpus += [q["instructions"] for group in task["followups"].values() for q in group.values()]
    prune_stats = export.prune_tokenizer(os.path.join(args.out, "tokenizer_full.json") if profile == "smoke"
                                        else os.path.join(ckpt_dir, "tokenizer", "tokenizer.json"),
                                        corpus, exp_dir)
    log(f"vocab pruned {prune_stats['vocab_before']} → {prune_stats['vocab_after']} "
        f"(drift={prune_stats['segmentation_drift']}/{prune_stats['texts']})")
    if prune_stats["segmentation_drift"]:
        log("WARNING: pruned tokenizer segments the corpus differently — widening the corpus is safer")
    tok_pruned = FastTokenizerAdapter(_rust_tok(os.path.join(exp_dir, "tokenizer.json")))

    # 2. gather embedding rows into the pruned order, then export fp32
    if not args.export_only:
        remap = json.load(open(os.path.join(exp_dir, "vocab_remap.json")))["remap"]
        with torch.no_grad():
            sd = model.state_dict()
            pruned_sd = export.gather_embeddings(sd, remap)
            model.load_state_dict(pruned_sd, strict=True)
            if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
                setattr(model.encoder.config, "vocab_size", prune_stats["vocab_after"])
        log(f"embeddings gathered to {prune_stats['vocab_after']} rows")
        torch.save({"state_dict": {k: v for k, v in model.state_dict().items()},
                    "cfg": cfg}, os.path.join(exp_dir, "model_pruned.pt"))
        fp32_path = os.path.join(exp_dir, "kasauti-laya-fp32.onnx")
        export.export_fp32(model, fp32_path)
    else:
        fp32_path = os.path.join(exp_dir, "kasauti-laya-fp32.onnx")
        if not os.path.exists(fp32_path):
            raise SystemExit("--export-only needs an existing fp32 export (or run without it once)")

    int8_path = export.quantize_int8(fp32_path, os.path.join(exp_dir, "kasauti-laya-int8.onnx"),
                                     quantize_embedding=args.quantize_embedding)
    q4_path = export.quantize_q4(fp32_path, os.path.join(exp_dir, "kasauti-laya-q4.onnx"),
                                 block_size=args.block_size)
    log(f"exported: fp32={os.path.getsize(fp32_path)/1e6:.1f} MB  "
        f"int8={os.path.getsize(int8_path)/1e6:.1f} MB  q4={os.path.getsize(q4_path)/1e6:.1f} MB")

    # 3. re-validate both quantized builds against the fp32 graph
    ort_fp32 = export.ort_session(fp32_path)
    ort_int8 = export.ort_session(int8_path)
    ort_q4 = export.ort_session(q4_path)
    sample = [{k: v for k, v in it.items() if k in ("ids", "markers", "qtype", "target")}
              for it in eval_items[: min(48, len(eval_items))]]
    p_fp32 = export.ort_predict(ort_fp32, sample)
    agree_int8 = export.agreement(p_fp32, export.ort_predict(ort_int8, sample))
    # MatMulNBits is a WebGPU-EP graph: the CPU EP can still run it, but if the kernel is missing
    # the agreement is meaningless — detect that and report it instead of pretending.
    try:
        agree_q4 = export.agreement(p_fp32, export.ort_predict(ort_q4, sample))
    except Exception as e:  # pragma: no cover - depends on the installed ORT build
        agree_q4 = {"n": 0, "top1_agreement": None, "error": str(e)[:200]}
    log(f"quantization agreement: int8={agree_int8['top1_agreement']} "
        f"(Δlogit {agree_int8['mean_abs_logit_delta']})  q4={agree_q4.get('top1_agreement')}")

    parity_int8 = evals.DEFAULT_GATES["quantized_agreement_min"]
    if (agree_int8["top1_agreement"] or 0) < parity_int8 and profile != "smoke":
        log("GATE FAILED: INT8 build disagrees with fp32 too often — refusing to publish")
        raise SystemExit(3)

    # ---------------------------------------------------------------- parity vectors
    tvec = parity.tokenizer_vectors(os.path.join(exp_dir, "tokenizer.json"))
    # the browser loads the INT8 graph on WASM and the Q4 graph on WebGPU: vectors come from fp32,
    # and the tolerance absorbs the quantization delta measured above
    tol = 0.35 if (agree_int8["top1_agreement"] or 1) < 0.995 else 0.08
    lvec = parity.logits_vectors(ort_fp32, sample, tolerance=tol)
    parity_path = os.path.join(exp_dir, "parity.json")
    parity.write_parity(parity_path, tvec, lvec)
    log(f"parity vectors: {len(tvec['vectors'])} tokenizer / {len(lvec['vectors'])} logits")

    if args.world_size > 1 and args.rank != 0:
        # Only rank 0 writes artifacts (all ranks writing identical files would race). Every DDP
        # collective happens inside the training loop and the ranks trim to the same micro-batch
        # count, so no rank can still be inside a collective here — the others just leave, and a
        # barrier would only add a way to hang when rank 0 later fails a gate.
        log("rank %d: training complete — evaluation, export and publish are rank-0 work; exiting" % args.rank)
        return {"profile": profile, "rank": args.rank, "worldSize": args.world_size,
                "skipped": "evaluation + export + publish (rank-0 work)"}

    # ---------------------------------------------------------------- manifest + staging
    # `gates_passed` is only defined when the eval ran; an --export-only run re-packages an existing
    # checkpoint whose gates were checked (or deliberately skipped) earlier, so it declares itself
    # report-only unless the caller says otherwise.
    gates_passed = bool(locals().get("gates_passed", False))
    tok_meta = dict(export.file_meta(os.path.join(exp_dir, "tokenizer.json")))
    tok_meta.update({"vocab": prune_stats["vocab_after"], "pruned": True,
                     "specials": tvec["specials"]})
    model_version = f"kasauti-laya-{'tiny-dev' if profile == 'smoke' else 'multilingual'}@{export.sha256_file(int8_path)[:8]}"
    manifest = calib.build_manifest(
        model_version=model_version,
        max_len=cfg["max_len"], head_max_len=cfg["head_max_len"],
        temperature=calibration["temperature"],
        temperature_by_options=calibration["temperature_by_options"],
        abstain={**calibration.get("abstain", {}), "default": round(float(max(calibration.get("abstain", {}).values() or [0.0])), 4)},
        files={},
        tokenizer={"src": os.path.join(exp_dir, "tokenizer.json"), **tok_meta},
        parity={"path": parity_path, **export.file_meta(parity_path)},
        evals={"accuracy": (metrics.get("accuracy") if not args.export_only else None),
               "ece": (metrics.get("ece") if not args.export_only else None),
               "int8Agreement": agree_int8["top1_agreement"],
               "q4Agreement": agree_q4.get("top1_agreement"),
               "dataset": f"kasauti-synthetic-{os.environ.get('KASAUTI_DATA_VERSION', '1.0.0')}"},
        extras={"profile": profile, "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # Honesty switch. The browser turns `reportOnly` into "abstain on everything", so an
                # export that did not clear the eval gates (or a smoke/dev export, whose gates are
                # advisory) can never move the verdict — it can only be inspected.
                "reportOnly": bool(profile == "smoke" or not gates_passed),
                # Drift guard: the browser compares this against `BANKS_SHA256` in src/lib/banks.ts.
                # A mismatch means the model was trained on different question wording than the
                # runtime asks — the one failure mode that is both silent and expensive.
                "banks": {"version": banks_raw.get("version"),
                          "sha256": hashlib.sha256(
                              json.dumps(banks_raw, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode("utf-8")).hexdigest()},
                "vocabPruning": {"before": prune_stats["vocab_before"], "after": prune_stats["vocab_after"],
                                 "drift": prune_stats["segmentation_drift"]},
                "graphIO": {"int8": export.graph_io_summary(int8_path)}},
    )
    staged_dir = os.path.join(args.out, "staging")
    manifest = publish.stage(staged_dir, {
        "wasm": int8_path,
        "webgpu": q4_path,
        "fp32": fp32_path,
    }, manifest, extra_files={"parity": parity_path})
    problems = publish.verify_staged(staged_dir, manifest)
    if problems:
        raise SystemExit("staged artifacts failed verification: " + "; ".join(problems))
    log(f"staged {len(manifest['files'])} artifacts in {staged_dir} (all sha256 verified)")

    published: Dict[str, Any] = {"staging": staged_dir, "modelVersion": model_version}
    if args.public_dir:
        publish.publish_local(staged_dir, args.public_dir)
        log(f"dev model copied to {args.public_dir}")
        published["local"] = args.public_dir
    if args.publish_repo:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        res = publish.publish_hf(staged_dir, args.publish_repo, token=token,
                                 commit_message=f"kasauti: {model_version}")
        log(f"HF publish: {res}")
        published["hf"] = res
        if res.get("ok"):
            print("\n" + publish.print_next_steps(res["url"], manifest))

    summary = {
        "profile": profile, "device": device, "modelVersion": model_version,
        "seconds": round(time.time() - t_start, 1), "rows": stats["rows"],
        "vocab": {"before": prune_stats["vocab_before"], "after": prune_stats["vocab_after"]},
        "sizes_mb": {"fp32": round(os.path.getsize(fp32_path) / 1e6, 1),
                     "int8": round(os.path.getsize(int8_path) / 1e6, 1),
                     "q4": round(os.path.getsize(q4_path) / 1e6, 1)},
        "agreement": {"int8": agree_int8, "q4": agree_q4},
        "published": published,
    }
    calib.write_json(summary, os.path.join(args.out, "summary.json"))
    log(f"done in {summary['seconds']}s → {os.path.join(args.out, 'summary.json')}")
    return summary


def _rust_tok(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def _keyed_eval_items(tok, eval_rows, tasks, cfg, splits) -> List[Dict[str, Any]]:
    """Eval items annotated with ``row_id``/``qid`` so ORT/torch predictions can be attributed."""
    out: List[Dict[str, Any]] = []
    for row in eval_rows:
        if row["split"] not in splits:
            continue
        if row["task"] not in tasks:
            continue
        for qid, q in row["questions"].items():
            gold = row["gold"].get(qid)
            if not gold:
                continue
            qi = core.to_internal(q)
            keys = core.option_keys(qi)
            order = row.get("option_orders", {}).get(qid) or list(range(len(keys)))
            ids, markers = core.build_sequence(tok, row["state"], qi, cfg["max_len"],
                                               cfg["head_max_len"], option_order=order)
            if len(markers) != len(keys):
                continue
            target = [float(gold["probabilities"].get(k, 0.0)) for k in keys]
            s = sum(target) or 1.0
            target = [t / s for t in target]
            out.append({"ids": ids, "markers": markers, "qtype": core.QTYPES[q["type"]],
                        "target": [target[opt] for opt in order],
                        "row_id": row["id"], "qid": qid, "task": row["task"], "lang": row["lang"],
                        "option_order": order})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv).parse_args(argv)
    # `torchrun --nproc_per_node=2` sets these; honouring them removes the need for hand-passed ranks
    env_world = int(os.environ.get("WORLD_SIZE", "0") or 0)
    if env_world > 1:
        args.world_size = env_world
        args.rank = int(os.environ.get("RANK", "0") or 0)
    if args.world_size > 1:
        import torch
        import torch.distributed as dist

        # gloo on CPU so the whole DDP path is testable without a GPU; nccl on Kaggle
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        if backend == "gloo" and not args.device:
            args.device = "cpu"
        try:
            run(args)
        finally:
            try:
                dist.destroy_process_group()
            except Exception as exc:  # peers may already have exited; shutdown noise is not a failure
                log("process group teardown: %s" % exc)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
