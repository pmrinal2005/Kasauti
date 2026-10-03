"""RLCD fine-tuning for Kasauti's Laya checkpoints.

Mirrors the reference ``train_ddp.py`` from the official Laya Kaggle notebook — same RLCD math
(zero-mean Gaussian exploration on logits, strictly-proper-scoring reward, GRPO-style group-mean
baseline, soft cross-entropy guidance), same split of encoder/head learning rates, same
calibration-holdout rule, same rolling checkpoint — with three Kasauti-specific additions:

1. **Multi-task**: `claim_v1`, `lens_v1`, `intent_concepts_v1`, `report_v1`, `impersonation_v1`,
   `resolution_v1` are trained in one model (they share a head; the question text distinguishes
   them), with per-task sampling weights so the small banks are not drowned out.
2. **Hard-negative up-weighting** and optional **rare-class re-weighting**.
3. **CPU/DDP-agnostic**: the identical loop runs single-process on CPU for the smoke profile and
   under ``torchrun --nproc_per_node=2`` on Kaggle's free 2×T4.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import core
from .datagen import fill  # noqa: F401  (kept for callers importing from here)
from .tok import FastTokenizerAdapter

QTYPES = core.QTYPES


# --------------------------------------------------------------------------------------
# dataset → tokenized training items
# --------------------------------------------------------------------------------------
def task_of(question_group: Optional[str]) -> str:
    return question_group or "unknown"


def rows_to_items(tok: FastTokenizerAdapter, rows: Sequence[Dict[str, Any]], tasks: Dict[str, Any],
                  max_len: int, head_max_len: int, tasks_filter: Optional[Sequence[str]] = None,
                  include_splits: Sequence[str] = ("train",)) -> List[Dict[str, Any]]:
    """Tokenize rows into RLCD items: ``{ids, markers, qtype, target, label, task, lang, split}``.

    ``target`` is placed in **marker (slot) order**, i.e. permuted by the row's ``option_orders``
    so it always lines up with what ``build_sequence`` put at each `[MASK]`.
    """
    items: List[Dict[str, Any]] = []
    for row in rows:
        if row["split"] not in include_splits:
            continue
        if tasks_filter and row["task"] not in tasks_filter:
            continue
        spec = tasks[row["task"]]
        for qid, q in row["questions"].items():
            gold = row["gold"].get(qid)
            if not gold:
                continue
            q_internal = core.to_internal(q)
            keys = core.option_keys(q_internal)
            probs = [float(gold["probabilities"].get(k, 0.0)) for k in keys]
            s = sum(probs)
            probs = [p / s for p in probs] if s > 0 else [1.0 / len(probs)] * len(probs)
            order = row.get("option_orders", {}).get(qid) or list(range(len(keys)))
            ids, markers = core.build_sequence(tok, row["state"], q_internal, max_len, head_max_len,
                                               option_order=order)
            if len(markers) != len(keys):
                continue  # options got clamped away — the reference trainer drops these too
            target_slots = [probs[opt] for opt in order]
            items.append({
                "ids": ids,
                "markers": markers,
                "qtype": QTYPES[q["type"]],
                "target": target_slots,
                "label": target_slots.index(max(target_slots)),
                "task": row["task"],
                "lang": row["lang"],
                "split": row["split"],
                "negative": bool(row.get("negative")),
                "template": row.get("template"),
                "qid": qid,
            })
    return items


def rare_class_weights(items: Sequence[Dict[str, Any]], power: float = 0.35) -> Dict[str, float]:
    """Per-(task,label) weights ~ 1/freq**power, normalised to mean 1 (reference re-weighting)."""
    counts: Dict[str, int] = {}
    for it in items:
        k = f"{it['task']}:{it['qtype']}:{it['label']}"
        counts[k] = counts.get(k, 0) + 1
    n = len(items) or 1
    raw = {k: (n / max(1, c)) ** power for k, c in counts.items()}
    mean = sum(raw.values()) / max(1, len(raw))
    return {k: v / mean for k, v in raw.items()}


def item_weight(it: Dict[str, Any], weights: Dict[str, float], negative_boost: float) -> float:
    w = weights.get(f"{it['task']}:{it['qtype']}:{it['label']}", 1.0)
    if it.get("negative"):
        w *= negative_boost
    return w


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def fit_one_temperature(pairs: Sequence[Tuple[Sequence[float], Sequence[float]]],
                        min_n: int = 10) -> float:
    """Reference ``fit_one_temp``: LBFGS on a single log-temperature, clamped to [TEMP_MIN, TEMP_MAX]."""
    import torch

    if len(pairs) < min_n:
        return 1.0
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, :len(z)] = torch.tensor(list(z), dtype=torch.float32)
        T[i, :len(t)] = torch.tensor(list(t), dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return core.clamp_temperature(float(log_t.exp().item()))


def train(train_items: Sequence[Dict[str, Any]], model, tok: FastTokenizerAdapter,
          cfg: Dict[str, Any], out_dir: str, args: Any) -> Dict[str, Any]:
    """The RLCD loop. Returns a small report dict (loss curve, sigma schedule, timings)."""
    import torch

    device = torch.device(args.device)
    model.to(device)
    model.train()

    world = max(1, int(getattr(args, "world_size", 1) or 1))
    rank = int(getattr(args, "rank", 0) or 0)
    is_main = rank == 0
    # Real DDP: without this each rank trains its own copy on its own shard and the two models
    # diverge, while both write the same checkpoint files. Mirrors the reference `train_ddp.py`.
    ddp = None
    if world > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            ddp = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                # the encoder's pooler (when the checkpoint has one) is never used by the decision
                # head, so DDP has to tolerate unused parameters rather than error out on them
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
            model = ddp

    rng = random.Random(20260922)
    order = list(range(len(train_items)))
    rng.shuffle(order)
    n_calib = min(args.calib_max, len(train_items) // 10)
    calib_items = [train_items[i] for i in sorted(order[:n_calib])]
    fit_items = [train_items[i] for i in sorted(order[n_calib:])]
    # trim so every rank sees the same number of micro-batches (reference issue #678)
    fit_items = fit_items[: len(fit_items) // world * world]
    my_items = fit_items[rank::world]

    weights = rare_class_weights(fit_items)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.01)
    updates = max(1, (len(my_items) // (args.micro_batch * args.grad_accum)) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    pad_id = tok.pad_token_id

    history: List[Dict[str, Any]] = []
    t0 = time.time()
    for epoch in range(args.epochs):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        for b_idx in range(0, len(my_items), args.micro_batch):
            chunk = my_items[b_idx:b_idx + args.micro_batch]
            if not chunk:
                continue
            batch = core.collate(chunk, pad_id)
            w = torch.tensor([item_weight(it, weights, args.negative_boost) for it in chunk])
            ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else _nullcontext()
            # accumulate without syncing; the last micro-batch of the accumulation window syncs
            accumulating = accum % args.grad_accum != 0 and (b_idx + args.micro_batch) < len(my_items)
            sync_ctx = model.no_sync() if (ddp is not None and accumulating) else _nullcontext()
            with ctx, sync_ctx:
                logits, _act = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            w = w.to(device)

            # 1) sample GROUP_SIZE noisy distributions with the zero-mean projection
            eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2) strictly proper scoring reward, group-mean baseline
            with torch.no_grad():
                r = core.proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                                       w_sph=args.w_sph, w_rps=args.w_rps)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3) policy-gradient on the logit perturbation + soft CE guidance, weighted per item
            # logp is [G, N]: the Gaussian log-density of each sampled z, summed over markers.
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            # Per-item weighting happens *after* the group mean, so a weighted item cannot leak its
            # weight into another item's advantage baseline.
            loss_rl = (-(adv * logp).mean(0) * w).mean()
            loss_ce = ((-(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)) * w).mean()
            loss = (loss_rl + args.ce_weight * loss_ce) / args.grad_accum

            if math.isfinite(float(loss.detach())):
                scaler.scale(loss).backward()
            accum += 1
            if accum % args.grad_accum == 0 or (b_idx + args.micro_batch) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach()) * args.grad_accum
            n_batches += 1
            if n_batches % max(1, args.log_every) == 0 and is_main:
                print(f"  epoch {epoch+1}/{args.epochs} | step {n_batches} | "
                      f"loss {float(loss.detach())*args.grad_accum:.4f} | reward {float(r.mean()):.3f} | "
                      f"sigma {sigma:.3f} | lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        history.append({"epoch": epoch + 1, "avg_loss": epoch_loss / max(1, n_batches),
                        "sigma": sigma, "seconds": round(time.time() - t0, 1)})

    # post-training type-level temperature calibration on the held-out slice
    model.eval()
    temps = [1.0, 1.0, 1.0]
    if calib_items and is_main:
        preds = _predict_items(model, tok, calib_items, device, args.batch_eval)
        for qt in range(3):
            pairs = [(z, t) for (q_, z, t) in preds if q_ == qt]
            if pairs:
                temps[qt] = fit_one_temperature(pairs, min_n=10)
    report = {"history": history, "seconds": round(time.time() - t0, 1),
              "n_items": len(my_items), "n_calib_holdout": len(calib_items),
              "temperatures_type_level": temps, "rare_class_weight_power": 0.35,
              "negative_boost": args.negative_boost, "world_size": world, "ddp": ddp is not None}
    model = model.module if ddp is not None else model
    if not is_main:
        return {**report, "rank_skipped_training_report": True}
    return report


def _nullcontext():
    import contextlib

    return contextlib.nullcontext()


def _predict_items(model, tok: FastTokenizerAdapter, items: Sequence[Dict[str, Any]], device,
                   batch_size: int = 8) -> List[Tuple[int, List[float], List[float]]]:
    """Run the model over items, returning (qtype, logits[:k], target[:k]) for calibration."""
    import torch

    model = model.module if hasattr(model, "module") else model
    out: List[Tuple[int, List[float], List[float]]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for chunk in core.iter_batches(list(items), batch_size):
            batch = core.collate(chunk, tok.pad_token_id)
            logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                              batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                              batch["qtype"].to(device))
            logits = logits.float().cpu().numpy()
            for i, it in enumerate(chunk):
                k = len(it["markers"])
                out.append((int(it["qtype"]), [float(v) for v in logits[i, :k]],
                            [float(v) for v in it["target"][:k]]))
    if was_training:
        model.train()
    return out


def save_checkpoint(model, tok: FastTokenizerAdapter, cfg: Dict[str, Any], out_dir: str,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """Mirror of the reference save: fp16 safetensors + encoder config + tokenizer + run config."""
    import torch
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    cfg = dict(cfg)
    cfg.update(extra or {})
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    _ = torch  # keep the import explicit: safetensors needs torch tensors
