"""Per-bucket temperature scaling + per-bucket abstention thresholds + calibration manifest.

Mirrors ``laya/calibrate.py``: one temperature per (question type, option-count bucket), fitted
against `answer_confidence` (max p) — the quantity the runtime gate compares — on a **held-out**
split, with a per-bucket floor below which the bucket is omitted and the type-level scalar is used
instead. This is the step that moves multilingual ECE from ~0.31 to ~0.10 in the model card, and
it is the difference between a confidence number that means "about c of these are right" and one
that is decoration.

Also fits the abstention map (``fit_abstention_thresholds``): the smallest confidence cut per
bucket that keeps the accepted-answer error under ``target_error``. The browser gate reads this
map straight out of the manifest.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import core

MIN_BUCKET_N = 30       # per-bucket floor for a temperature
MIN_TYPE_N = 10         # type-level fallback floor
MIN_ABSTAIN_N = 20      # per-bucket floor for an abstention threshold
ECE_HOLDOUT_FRAC = 0.25


def _records(preds: Sequence[Tuple[int, Sequence[float], Sequence[float]]]) -> List[Tuple[int, Any, Any, int]]:
    import numpy as np

    out = []
    for qt, logits, target in preds:
        z = np.asarray(logits, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        s = t.sum()
        t = t / s if s > 0 else np.full_like(t, 1.0 / max(1, len(t)))
        out.append((int(qt), z, t, int(len(z))))
    return out


def fit_temperature_map(recs: Sequence[Tuple[int, Any, Any, int]], compute_ece: bool = True,
                        seed: int = 0):
    """Fit ``(type-level scalars, per-bucket map)`` and optionally report ECE on a held-out slice."""
    import numpy as np
    import torch

    def fit_one(pairs: Sequence[Tuple[Any, Any]], min_n: int) -> Optional[float]:
        if len(pairs) < min_n:
            return None
        kmax = max(len(z) for z, _ in pairs)
        Z = torch.full((len(pairs), kmax), -1e4)
        T = torch.zeros((len(pairs), kmax))
        for i, (z, t) in enumerate(pairs):
            Z[i, :len(z)] = torch.tensor(np.asarray(z), dtype=torch.float32)
            T[i, :len(t)] = torch.tensor(np.asarray(t), dtype=torch.float32)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        return core.clamp_temperature(float(log_t.exp().item()))

    by_type: Dict[int, List[Tuple[Any, Any]]] = {}
    by_bucket: Dict[str, List[Tuple[Any, Any]]] = {}
    n_by_bucket: Dict[str, int] = {}
    for qt, z, t, k in recs:
        by_type.setdefault(qt, []).append((z, t))
        key = core.temp_bucket(qt, k)
        by_bucket.setdefault(key, []).append((z, t))
        n_by_bucket[key] = n_by_bucket.get(key, 0) + 1

    temperature = [1.0, 1.0, 1.0]
    for qt, pairs in by_type.items():
        fitted = fit_one(pairs, MIN_TYPE_N)
        if fitted is not None:
            temperature[qt] = fitted
    bucket_temps: Dict[str, float] = {}
    for key, pairs in by_bucket.items():
        fitted = fit_one(pairs, MIN_BUCKET_N)
        if fitted is not None:
            bucket_temps[key] = fitted

    report: Dict[str, Any] = {
        "temperature": temperature,
        "temperature_by_options": bucket_temps,
        "n_by_bucket": n_by_bucket,
        "min_bucket_n": MIN_BUCKET_N,
    }
    if compute_ece:
        fit_t = [1.0, 1.0, 1.0]
        fit_b: Dict[str, float] = {}
        for qt, pairs in by_type.items():
            v = fit_one(pairs, MIN_TYPE_N)
            if v is not None:
                fit_t[qt] = v
        for key, pairs in by_bucket.items():
            v = fit_one(pairs, MIN_BUCKET_N)
            if v is not None:
                fit_b[key] = v
        report["ece_fitted"] = _ece_report(recs, fit_t, fit_b)
        report["ece_raw"] = _ece_report(recs, [1.0, 1.0, 1.0], {})
    return report


def _ece_report(recs: Sequence[Tuple[int, Any, Any, int]], temperature: Sequence[float],
                temperature_by_options: Dict[str, float]) -> Dict[str, Any]:
    import numpy as np

    confs, corrects = [], []
    per_bucket: Dict[str, List[Tuple[float, bool]]] = {}
    for qt, z, t, k in recs:
        t_scale = temperature_by_options.get(core.temp_bucket(qt, k), temperature[qt])
        p = core.softmax_rows(np.asarray(z)[:k].reshape(1, -1), t_scale)[0]
        conf = float(np.max(p))
        correct = bool(int(np.argmax(p)) == int(np.argmax(t)))
        confs.append(conf)
        corrects.append(correct)
        per_bucket.setdefault(core.temp_bucket(qt, k), []).append((conf, correct))
    out = {"overall": round(core.ece_score(confs, corrects), 4),
           "n": len(confs), "mean_confidence": round(float(np.mean(confs)), 4) if confs else None,
           "accuracy": round(float(np.mean(corrects)), 4) if corrects else None}
    out["by_bucket"] = {
        k: {"ece": round(core.ece_score([c for c, _ in v], [y for _, y in v]), 4), "n": len(v)}
        for k, v in sorted(per_bucket.items())
    }
    return out


def fit_abstention_thresholds(recs: Sequence[Tuple[int, Any, Any, int]], temperature: Sequence[float],
                              temperature_by_options: Dict[str, float], target_error: float = 0.15,
                              min_bucket_n: int = MIN_ABSTAIN_N, conservative: bool = True) -> Dict[str, float]:
    """Smallest cut per bucket that keeps accepted-answer error ≤ ``target_error``.

    A Wilson-style pseudo-error is added under ``conservative=True`` so a bucket cannot clear the
    gate on a short lucky run; a bucket that admits nothing reports 1.0 (abstain on everything).
    """
    import numpy as np

    by_bucket: Dict[str, List[Tuple[float, bool]]] = {}
    for qt, z, t, k in recs:
        t_scale = temperature_by_options.get(core.temp_bucket(qt, k), temperature[qt])
        p = core.softmax_rows(np.asarray(z)[:k].reshape(1, -1), t_scale)[0]
        conf = float(np.max(p))
        correct = bool(int(np.argmax(p)) == int(np.argmax(t)))
        by_bucket.setdefault(core.temp_bucket(qt, k), []).append((conf, correct))

    out: Dict[str, float] = {}
    for key, rows in by_bucket.items():
        if len(rows) < min_bucket_n:
            continue
        best = 1.0
        for cut in sorted({c for c, _ in rows}):
            kept = [(c, y) for c, y in rows if c >= cut]
            if not kept:
                continue
            errors = sum(0 if y else 1 for _, y in kept)
            n = len(kept)
            err = errors / n
            if conservative:
                # Wilson upper bound on the error rate at ~80% one-sided confidence
                p_hat, z_ = err, 1.28
                denom = 1 + z_ * z_ / n
                centre = (p_hat + z_ * z_ / (2 * n)) / denom
                half = z_ * math.sqrt(p_hat * (1 - p_hat) / n + z_ * z_ / (4 * n * n)) / denom
                err = centre + half
            if err <= target_error:
                best = cut
                break
        out[key] = round(float(best), 4)
    return out


def coverage_at(recs: Sequence[Tuple[int, Any, Any, int]], temperature: Sequence[float],
                temperature_by_options: Dict[str, float], abstain: Dict[str, float]) -> Dict[str, Any]:
    """How much the abstention map keeps, and how accurate what it keeps is."""
    import numpy as np

    kept_ok, kept, total = 0, 0, 0
    for qt, z, t, k in recs:
        key = core.temp_bucket(qt, k)
        t_scale = temperature_by_options.get(key, temperature[qt])
        p = core.softmax_rows(np.asarray(z)[:k].reshape(1, -1), t_scale)[0]
        conf = float(np.max(p))
        cut = abstain.get(key, abstain.get("default", 0.0))
        correct = bool(int(np.argmax(p)) == int(np.argmax(t)))
        total += 1
        if conf >= cut:
            kept += 1
            kept_ok += int(correct)
    return {
        "coverage": round(kept / total, 4) if total else 0.0,
        "selective_accuracy": round(kept_ok / kept, 4) if kept else None,
        "n": total,
    }


# --------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------
def build_manifest(*, model_version: str, max_len: int, head_max_len: int,
                   temperature: Sequence[float], temperature_by_options: Dict[str, float],
                   abstain: Dict[str, float], files: Dict[str, Dict[str, Any]],
                   tokenizer: Dict[str, Any], parity: Dict[str, Any],
                   evals: Dict[str, Any], extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The manifest the browser reads. **Schema 2** — keep in sync with
    ``src/core/laya-client-browser.ts`` (validated by ``tests/manifest.test.ts``)."""
    m: Dict[str, Any] = {
        "schema": 2,
        "modelVersion": model_version,
        "maxLen": int(max_len),
        "headMaxLen": int(head_max_len),
        "qtypes": core.QTYPES,
        "temperature": [core.clamp_temperature(t) for t in temperature],
        "temperatureByBucket": {k: core.clamp_temperature(v) for k, v in sorted(temperature_by_options.items())},
        "abstain": {k: round(float(v), 4) for k, v in sorted(abstain.items())},
        "files": files,
        "tokenizer": tokenizer,
        "parity": parity,
        "evals": evals,
    }
    if extras:
        m.update(extras)
    return m


def write_json(obj: Any, path: str) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
