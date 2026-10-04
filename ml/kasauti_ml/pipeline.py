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
import shutil
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
    p.add_argument("--revision", default=None,
                   help="Hub revision (commit SHA / tag) to train from; default: the reviewed pin "
                        "laya.revisions.PINNED_REVISIONS carries for --model-id")
    p.add_argument("--no-pin-base", dest="pin_base", action="store_false", default=True,
                   help="train from the Hub default branch instead of the reviewed pin "
                        "(not reproducible — the manifest records that it was not)")
    p.add_argument("--all-files", dest="only_subfolder", action="store_false", default=True,
                   help="download every file in the repo (~2.5 GB) instead of just --subfolder (~680 MB)")
    p.add_argument("--max-tokenizer-drift", type=float, default=0.005,
                   help="fail the export when the pruned tokenizer reassembles more than this "
                        "fraction of the corpus differently (default 0.5%%)")
    p.add_argument("--allow-tokenizer-drift", action="store_true",
                   help="ship even when the drift gate fails (recorded in the manifest)")
    p.add_argument("--reference-check", action="store_true",
                   help="after staging, cross-check the export against the installed `laya` "
                        "reference client (mirrors / calibration / ONNXAgent answers)")
    p.add_argument("--allow-reference-check-skip", action="store_true",
                   help="do not fail the run when `laya` is missing for --reference-check")
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


def _resolve_base_revision(model_id: str, requested: Optional[str], pin: bool) -> Optional[str]:
    """The Hub revision the fine-tune starts from.

    `convaiinnovations/laya` is a live repository — its root `config.json` was added after the
    commit Laya's own `PINNED_REVISIONS` names — so "download whatever main is today" makes the
    exported model unreproducible from its own manifest: the same command, run a month later,
    silently fine-tunes a different base. Laya ships the pin for exactly this reason, and this
    resolves it the same way the reference does (`resolve_revision(mid, "reviewed")`), with a local
    fallback so the pipeline stays runnable without `laya` installed.
    """
    if not pin:
        return requested
    try:
        from laya.revisions import PINNED_REVISIONS, resolve_revision

        return resolve_revision(model_id, requested or "reviewed")
    except Exception:
        pass
    if requested:
        return requested
    try:  # `laya` not installed: use its published table rather than the moving branch
        _PINNED = {"convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"}
        return _PINNED.get(model_id.lower()) or _PINNED.get(model_id)
    except Exception:  # pragma: no cover
        return None


def _download_base(args: Any) -> Tuple[str, Dict[str, Any]]:
    """Download the checkpoint the fine-tune starts from: pinned, minimal, and with provenance.

    Two things the reference implementation does that a fine-tune must not skip:

    * **Pin the revision** (see `_resolve_base_revision`).
    * **Fetch only the subfolder.** The repo bundles all three checkpoints (~2.5 GB); with
      ``--subfolder multilingual`` the fine-tune needs five files (~680 MB). Laya's own
      `ONNXAgent` narrows ``allow_patterns`` to the subfolder for the same reason.
    """
    from huggingface_hub import snapshot_download

    revision = _resolve_base_revision(args.model_id, args.revision, args.pin_base)
    prefix = f"{args.subfolder}/" if args.subfolder else ""
    patterns = ([prefix + "*"] if args.subfolder and args.only_subfolder else None)
    log(f"downloading {args.model_id} (subfolder={args.subfolder or '-'}, "
        f"revision={revision or 'DEFAULT BRANCH'}{', subfolder files only' if patterns else ''})…")
    path = snapshot_download(args.model_id, revision=revision, allow_patterns=patterns)
    if args.subfolder:
        sub = os.path.join(path, args.subfolder)
        if os.path.isdir(sub):
            path = sub
    provenance: Dict[str, Any] = {
        "modelId": args.model_id,
        "subfolder": args.subfolder or None,
        "revisionRequested": revision,
        "pinned": bool(args.pin_base and revision),
        "onlySubfolderFiles": bool(patterns),
    }
    try:  # the commit the snapshot actually resolved to (None for a plain directory)
        from laya.revisions import snapshot_revision

        provenance["revisionResolved"] = snapshot_revision(path)
    except Exception:
        provenance["revisionResolved"] = None
    try:  # honours LAYA_SHA256_DIGESTS; a no-op when that variable is unset
        from laya.revisions import verify_digests

        verify_digests(path)
        provenance["digestsVerified"] = True
    except Exception as e:
        provenance["digestsVerified"] = False
        provenance["digestError"] = str(e)[:200]
        raise SystemExit(f"base checkpoint failed digest verification: {e}")
    return path, provenance


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
    base: Dict[str, Any] = {"modelId": None, "subfolder": None, "revisionRequested": None,
                            "revisionResolved": None, "pinned": False,
                            "onlySubfolderFiles": False, "kind": "tiny-synthetic"}
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
            from laya.agent import _fix_tokenizer_config

            ckpt_dir, base = _download_base(args)
            base["kind"] = "hub-snapshot"
            _fix_tokenizer_config(ckpt_dir)
        else:
            base = {"modelId": args.model_id, "subfolder": args.subfolder,
                    "revisionRequested": None, "revisionResolved": None, "pinned": False,
                    "onlySubfolderFiles": False, "kind": "local-dir", "path": ckpt_dir}
            try:
                from laya.revisions import snapshot_revision

                base["revisionResolved"] = snapshot_revision(ckpt_dir)
            except Exception:
                pass
            if args.resume_from:
                # carry the pinned-base provenance through a resume, so a re-export's manifest still
                # names the Hub commit the weights originally came from
                try:
                    with open(os.path.join(ckpt_dir, "rl_agent_config.json")) as f:
                        prev = (json.load(f).get("training") or {}).get("base")
                    if isinstance(prev, dict) and prev.get("modelId"):
                        base = {**prev, "kind": "resumed-finetune", "path": ckpt_dir}
                except Exception as e:
                    log(f"resume: no base provenance in rl_agent_config.json ({type(e).__name__})")
            if not args.resume_from:  # a resume dir is a previous *Kasauti* export, not the base
                try:
                    from laya.agent import _fix_tokenizer_config

                    _fix_tokenizer_config(ckpt_dir)
                except Exception as e:
                    log(f"tokenizer-config repair skipped: {type(e).__name__}: {e}")
        log(f"base: {base['kind']} {base.get('modelId') or base.get('path')} "
            f"revision={base.get('revisionResolved') or base.get('revisionRequested') or '-'} "
            f"pinned={'yes' if base.get('pinned') else 'no'}")
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
        if args.skip_train:
            log("--skip-train: using the loaded weights as-is (no RLCD updates)")
            report = {"skipped": True, "seconds": 0.0, "history": []}
        else:
            report = trainer.train(items, model, tok, cfg, args.out, args)
            log(f"trained in {report['seconds']}s | loss curve "
                f"{[round(h['avg_loss'], 3) for h in report['history']]}")
        if args.world_size > 1 and args.rank != 0:
            # Evaluation, export and publish are rank-0 work. Leaving *here* (not after the export,
            # as before) stops rank 1 from writing the same ~1.3 GB fp32 graph and quantized files
            # into the same directory while rank 0 is reading them.
            log("rank %d: training complete — eval/export/publish are rank-0 work; exiting" % args.rank)
            return {"profile": profile, "rank": args.rank, "worldSize": args.world_size,
                    "skipped": "evaluation + export + publish (rank-0 work)"}
        calib.write_json(report, os.path.join(args.out, "train_report.json"))
        if profile != "smoke" and not args.skip_train:
            # A loadable Laya checkpoint of the fine-tune, BEFORE vocab pruning. It is what
            # `--resume-from` / `--skip-train` re-export from (no second GPU run when only the export
            # step needs a retry), and `laya.load(<dir>)` can run it directly for spot checks.
            ft_dir = os.path.join(args.out, "finetuned")
            unwrapped = model.module if hasattr(model, "module") else model
            trainer.save_checkpoint(unwrapped, tok, cfg, ft_dir, extra={
                "temperature": report.get("temperatures_type_level", [1.0, 1.0, 1.0]),
                "training": {"kasauti": True, "epochs": args.epochs,
                             "world_size": args.world_size, "base": base}})
            log(f"fine-tuned checkpoint saved → {ft_dir}")

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
        gates_override = _load_gates(args.gates)
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
    # The question HEAD is tokenized separately from the state — "<type> question: <ins>" and each
    # " <rendered option>" — and its tokens ("▁choice", "▁question", "level", option keys, …) appear in
    # every sequence the model sees. Leaving them out of the prune corpus dropped them from the
    # pruned vocab, so the browser re-spelled them with byte-fallback tokens the model was never
    # trained on. Add the exact strings `build_head` encodes, for every bank question.
    for row in rows:
        for q in row["questions"].values():
            qi = core.to_internal(q)
            corpus.append("%s question: %s" % (qi["t"], qi["ins"]))
            corpus += [" " + o for o in core.render_options(qi)]
    corpus = list(dict.fromkeys(corpus))  # order-preserving de-dup (the drift check runs per text)
    prune_stats = export.prune_tokenizer(os.path.join(args.out, "tokenizer_full.json") if profile == "smoke"
                                        else os.path.join(ckpt_dir, "tokenizer", "tokenizer.json"),
                                        corpus, exp_dir)
    log(f"vocab pruned {prune_stats['vocab_before']} → {prune_stats['vocab_after']} "
        f"(drift={prune_stats['segmentation_drift']}/{prune_stats['texts']})")
    # A pruned tokenizer that reassembles text differently from the one the model was trained with
    # silently degrades every prediction for the affected inputs — there is no error, just worse
    # answers. With the complete merge closure (see `export._merge_parts`) this is 0 on both the
    # synthetic and the real multilingual tokenizer, so the gate exists to catch a regression, not
    # to be routinely overridden.
    drift_rate = prune_stats["segmentation_drift"] / max(1, prune_stats["texts"])
    if prune_stats["segmentation_drift"]:
        log(f"WARNING: pruned tokenizer segments {prune_stats['segmentation_drift']}/"
            f"{prune_stats['texts']} corpus texts differently ({drift_rate:.3%})")
    if drift_rate > args.max_tokenizer_drift:
        if not args.allow_tokenizer_drift:
            log(f"GATE FAILED: tokenizer drift {drift_rate:.3%} > {args.max_tokenizer_drift:.3%} "
                f"— the graph and the tokenizer would disagree at inference. Widen the corpus "
                f"(more --variants) or pass --allow-tokenizer-drift to ship anyway.")
            raise SystemExit(5)
        log("tokenizer drift gate overridden by --allow-tokenizer-drift (recorded in the manifest)")
    tok_pruned = FastTokenizerAdapter(_rust_tok(os.path.join(exp_dir, "tokenizer.json")))

    # 2. resize the embedding table to the pruned vocabulary, then export fp32
    with open(os.path.join(exp_dir, "vocab_remap.json")) as f:
        remap = json.load(f)["remap"]
    if not args.export_only:
        # Export runs on CPU in fp32: the dummies `export_fp32` traces with are CPU tensors, and a
        # model left on CUDA after training fails the trace with a device mismatch.
        model = model.to("cpu").float().eval()
        resized = export.resize_vocab(model, remap)
        log(f"embeddings resized to {prune_stats['vocab_after']} rows ({', '.join(resized)})")
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
    # eval_items were tokenized with the FULL tokenizer; the exported graph (and the browser) index
    # the PRUNED table, so the sample has to be in pruned ids or every agreement/parity number is
    # measured on the wrong embedding rows.
    sample = [{"ids": export.remap_ids(it["ids"], remap), "markers": it["markers"],
               "qtype": it["qtype"], "target": it["target"]}
              for it in eval_items[: min(48, len(eval_items))]]
    p_fp32 = export.ort_predict(ort_fp32, sample)
    p_int8 = export.ort_predict(ort_int8, sample)
    agree_int8 = export.agreement(p_fp32, p_int8)
    # MatMulNBits is a WebGPU-EP graph: the CPU EP can still run it, but if the kernel is missing
    # the agreement is meaningless — detect that and report it instead of pretending.
    try:
        p_q4 = export.ort_predict(ort_q4, sample)
        agree_q4 = export.agreement(p_fp32, p_q4)
    except Exception as e:  # pragma: no cover - depends on the installed ORT build
        agree_q4 = {"n": 0, "top1_agreement": None, "error": str(e)[:200]}
    log(f"quantization agreement: int8={agree_int8['top1_agreement']} "
        f"(decisive {agree_int8.get('decisive_top1_agreement')} on {agree_int8.get('n_decisive')}/"
        f"{agree_int8['n']}, Δlogit {agree_int8['mean_abs_logit_delta']})  "
        f"q4={agree_q4.get('top1_agreement')} (decisive {agree_q4.get('decisive_top1_agreement')})")

    parity_int8 = evals.DEFAULT_GATES["quantized_agreement_min"]
    if args.gates:  # the same override file the eval gates read (previously ignored here)
        parity_int8 = float(_load_gates(args.gates).get("quantized_agreement_min", parity_int8))
    # Gate on the DECISIVE agreement: raw top-1 over near-tied items measures noise, not
    # quantization quality (a random mini checkpoint scores ~0.3-0.5 raw with |Δlogit| ≈ 0.002).
    # If nothing in the sample is decisive (an untrained model), fall back to the raw number.
    int8_gate_value = agree_int8.get("decisive_top1_agreement")
    if int8_gate_value is None or (agree_int8.get("n_decisive") or 0) < 8:
        int8_gate_value = agree_int8["top1_agreement"]
    if (int8_gate_value or 0) < parity_int8 and profile != "smoke":
        log(f"GATE FAILED: INT8 build disagrees with fp32 on decisive items ({int8_gate_value} < "
            f"{parity_int8}) — refusing to publish")
        raise SystemExit(3)

    # ---------------------------------------------------------------- parity vectors
    tvec = parity.tokenizer_vectors(os.path.join(exp_dir, "tokenizer.json"))
    # the browser loads the INT8 graph on WASM and the Q4 graph on WebGPU: vectors come from fp32,
    # and the tolerance absorbs the quantization delta measured above
    tol = 0.35 if (int8_gate_value or 1) < 0.995 else 0.08
    # The fixed 0.35 / 0.08 were tuned on the 0.5 M-parameter smoke model. A 4-bit mmBERT-base drifts
    # further than that on some sequences, and the Worker would then demote a perfectly good export
    # on every device. Widen to the *measured* worst case (+25 %), never past a hard cap: a graph
    # that drifts more than the cap is a broken quantization, not a tolerance problem.
    measured = 0.0
    for other in (p_int8, locals().get("p_q4") or []):
        for (_qa, za, _ta), (_qb, zb, _tb) in zip(p_fp32, other):
            measured = max(measured, max((abs(a - b) for a, b in zip(za, zb)), default=0.0))
    tol_cap = float(os.environ.get("KASAUTI_PARITY_TOL_CAP", "1.5"))
    if measured > tol:
        if measured > tol_cap and profile != "smoke":
            log(f"GATE FAILED: quantized logits drift up to {measured:.3f} > cap {tol_cap} — "
                f"re-export with --block-size 64 (or --no-quantize-embedding)")
            raise SystemExit(3)
        tol = round(min(tol_cap, measured * 1.25), 3)
    log(f"parity tolerance {tol} (measured worst |Δlogit| {measured:.4f} over int8 + q4)")
    lvec = parity.logits_vectors(ort_fp32, sample, tolerance=tol)
    parity_path = os.path.join(exp_dir, "parity.json")
    parity.write_parity(parity_path, tvec, lvec)
    log(f"parity vectors: {len(tvec['vectors'])} tokenizer / {len(lvec['vectors'])} logits")

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
               "int8DecisiveAgreement": agree_int8.get("decisive_top1_agreement"),
               "q4DecisiveAgreement": agree_q4.get("decisive_top1_agreement"),
               "dataset": f"kasauti-synthetic-{os.environ.get('KASAUTI_DATA_VERSION', '1.0.0')}"},
        extras={"profile": profile, "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # Which checkpoint bytes this model was fine-tuned from, and whether that base was
                # a reviewed pin or a moving branch: the first question anyone asks when two
                # exports of "the same" model disagree.
                "base": base,
                # Honesty switch. The browser turns `reportOnly` into "abstain on everything", so an
                # export that did not clear the eval gates (or a smoke/dev export, whose gates are
                # advisory) can never move the verdict — it can only be inspected.
                # A `plumbing: true` gates file (ml/scripts/gates_plumbing.json) relaxes every
                # threshold so the real-architecture path can be exercised on a random mini
                # checkpoint; such an export is report-only no matter what its gates said.
                "reportOnly": bool(profile == "smoke" or not gates_passed
                                   or _load_gates(args.gates).get("plumbing")),
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

    # ---------------------------------------------------------------- reference check
    # Runs on the *staged bytes*, before anything is published: if the reference Laya client and
    # this export disagree about the same graph, nobody should be able to download it under
    # Kasauti's name. `laya` missing is a skip (CI without the extra), not a pass — it is reported
    # as skipped either way so the summary never implies a check that did not run.
    reference: Dict[str, Any] = {"ran": False}
    if args.reference_check:
        from . import reference_check as refcheck

        ref_out = os.path.join(args.out, "reference_check.json")
        argv = ["--staging", staged_dir, "--banks", args.banks, "--out", ref_out, "--report-only"]
        if profile != "smoke" and ckpt_dir:
            # the base checkpoint carries the tokenizer_config.json / encoder/ the scratch agent
            # dir wants; the smoke profile has no such directory
            argv += ["--checkpoint-dir", ckpt_dir]
        code = refcheck.main(argv)
        reference = {"ran": True, "exit": code, "report": ref_out}
        if os.path.exists(ref_out):
            with open(ref_out, "r", encoding="utf-8") as f:
                passed = bool(json.load(f).get("passed"))
            reference["passed"] = passed
            if code == 2:
                log("reference check SKIPPED: the `laya` package is not importable")
            elif passed:
                # travels with the model so an auditor can see it was checked, and by what
                shutil.copyfile(ref_out, os.path.join(staged_dir, "reference_check.json"))
                log("reference check passed (mirrors + calibration + reference client + evals)")
            else:
                log("reference check FAILED — refusing to publish")
                if not args.allow_reference_check_skip:
                    raise SystemExit(4)

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
        "profile": profile, "device": device, "modelVersion": model_version, "base": base,
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


def _load_gates(path: Optional[str]) -> Dict[str, Any]:
    """Gate overrides from a JSON file; keys starting with `_` are comments."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


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
        local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        if backend == "nccl":
            # Without this every rank resolves "cuda" to cuda:0: both processes load the model onto
            # the same T4 (OOM at mmBERT-base x 2) and DDP receives device_ids=[None]. The reference
            # train_ddp.py does exactly this before init_process_group.
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
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
