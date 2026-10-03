"""Evaluation metrics and the release gates.

Gates exist so a fine-tune that regresses on the thing that matters — catching a pay-to-withdraw
or impersonation script — cannot be published just because the average accuracy went up. Each gate
reports a measured value, a threshold, and pass/fail; ``pipeline.py`` refuses to publish on a fail.

Metrics mirror ``laya/evals.py``: choice accuracy, noul accuracy, score MAE, ECE, Brier, plus
selective accuracy under the abstention map. On top of those: per-language accuracy (the whole
point of the multilingual checkpoint), per-task accuracy, and the unpruned-vs-quantized agreement
rate used as the INT8/Q4 parity gate.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import core

#: Gates. Chosen against the model card's published numbers, then tightened as Kasauti needs.
DEFAULT_GATES: Dict[str, Any] = {
    "accuracy_overall": 0.80,
    "accuracy_per_language_min": 0.70,
    "ece_overall_max": 0.12,
    "ece_per_bucket_max": 0.20,
    "noul_accuracy_min": 0.85,
    "choice_accuracy_min": 0.75,
    "precision_pay_to_withdraw_min": 0.80,
    "recall_pay_to_withdraw_min": 0.80,
    "recall_impersonation_min": 0.85,
    "false_positive_rate_negatives_max": 0.10,
    "abstention_coverage_min": 0.55,
    "quantized_agreement_min": 0.95,
    "tokenizer_parity_exact": True,
}


def evaluate_predictions(rows: Sequence[Dict[str, Any]], preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare predictions to gold rows.

    ``rows`` come from ``datagen.build_rows`` (runtime question shape + ``gold.probabilities``);
    ``preds`` are ``{row_id, qid, probabilities: {key: p}, confidence}`` — the exact shape the
    browser's calibrated decode produces, so eval and runtime measure the same quantity.
    """
    import numpy as np

    by_id = {(r["id"], qid): (r, qid) for r in rows for qid in r["questions"]}
    n = correct = 0
    per_task: Dict[str, List[int]] = {}
    per_lang: Dict[str, List[int]] = {}
    per_bucket: Dict[str, List[Tuple[float, bool]]] = {}
    confs_all, correct_all = [], []
    noul_ok = noul_n = 0
    choice_ok = choice_n = 0
    tp_p2w = fp_p2w = fn_p2w = 0
    tp_imp = fn_imp = 0
    fp_neg = neg_n = 0
    score_err: List[float] = []

    for p in preds:
        row, qid = by_id[(p["row_id"], p["qid"])]
        q = row["questions"][qid]
        gold = row["gold"][qid]["probabilities"]
        keys = list(gold.keys())
        probs = np.array([float(p["probabilities"].get(k, 0.0)) for k in keys])
        if probs.sum() <= 0:
            probs = np.full(len(keys), 1.0 / len(keys))
        else:
            probs = probs / probs.sum()
        g = np.array([float(gold[k]) for k in keys])
        g = g / g.sum() if g.sum() > 0 else np.full(len(keys), 1.0 / len(keys))

        argmax_p, argmax_g = int(probs.argmax()), int(g.argmax())
        is_correct = argmax_p == argmax_g
        correct += int(is_correct)
        n += 1
        per_task.setdefault(row["task"], []).append(int(is_correct))
        per_lang.setdefault(row["lang"], []).append(int(is_correct))
        bucket = core.temp_bucket(core.QTYPES[q["type"]], len(keys))
        conf = float(probs.max())
        per_bucket.setdefault(bucket, []).append((conf, is_correct))
        confs_all.append(conf)
        correct_all.append(is_correct)

        if q["type"] == "noul":
            noul_n += 1
            noul_ok += int((probs[1] >= 0.5) == (g[1] >= 0.5))
            if qid == "pay_to_withdraw":
                pred_yes, gold_yes = probs[1] >= 0.5, g[1] >= 0.5
                tp_p2w += int(pred_yes and gold_yes)
                fp_p2w += int(pred_yes and not gold_yes)
                fn_p2w += int(not pred_yes and gold_yes)
            if qid in ("digital_arrest", "otp", "secrecy", "transfer"):
                tp_imp += int(probs[1] >= 0.5 and g[1] >= 0.5)
                fn_imp += int(probs[1] < 0.5 and g[1] >= 0.5)
        if q["type"] == "choice":
            choice_n += 1
            choice_ok += int(is_correct)
        if q["type"] == "score":
            k = len(keys)
            exp_score = float((np.arange(k) * probs).sum())
            gold_score = float((np.arange(k) * g).sum())
            score_err.append(abs(exp_score - gold_score))

        if row.get("negative"):
            neg_n += 1
            fp_neg += int(not is_correct)

    def _rel(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "choice_accuracy": _rel(choice_ok, choice_n),
        "noul_accuracy": _rel(noul_ok, noul_n),
        "score_mae": round(float(np.mean(score_err)), 4) if score_err else None,
        "ece": round(core.ece_score(confs_all, correct_all), 4) if n else None,
        "brier": round(float(np.mean([(c - float(y)) ** 2 for c, y in zip(confs_all, correct_all)])), 4) if n else None,
        "per_task": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_task.items())},
        "per_language": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_lang.items())},
        "per_bucket_ece": {k: round(core.ece_score([c for c, _ in v], [y for _, y in v]), 4)
                           for k, v in sorted(per_bucket.items())},
        "precision_pay_to_withdraw": _rel(tp_p2w, tp_p2w + fp_p2w),
        "recall_pay_to_withdraw": _rel(tp_p2w, tp_p2w + fn_p2w),
        "recall_impersonation": _rel(tp_imp, tp_imp + fn_imp),
        "false_positive_rate_negatives": _rel(fp_neg, neg_n),
        "n_negatives": neg_n,
    }


def run_gates(metrics: Dict[str, Any], gates: Optional[Dict[str, Any]] = None,
              extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate every gate; returns ``{passed, failures[], checks[]}``."""
    g = dict(DEFAULT_GATES)
    g.update(gates or {})
    checks: List[Dict[str, Any]] = []

    def check(name: str, value: Any, threshold: Any, ok: bool, note: str = "") -> None:
        checks.append({"name": name, "value": value, "threshold": threshold, "ok": bool(ok), "note": note})

    check("accuracy_overall", metrics.get("accuracy"), g["accuracy_overall"],
          (metrics.get("accuracy") or 0) >= g["accuracy_overall"])
    langs = metrics.get("per_language") or {}
    worst = min(langs.values()) if langs else 0
    worst_lang = min(langs, key=langs.get) if langs else None
    check("accuracy_per_language_min", worst, g["accuracy_per_language_min"],
          worst >= g["accuracy_per_language_min"], f"worst = {worst_lang}")
    check("ece_overall_max", metrics.get("ece"), g["ece_overall_max"],
          (metrics.get("ece") if metrics.get("ece") is not None else 1) <= g["ece_overall_max"])
    buckets = metrics.get("per_bucket_ece") or {}
    worst_bucket = max(buckets.values()) if buckets else 1
    check("ece_per_bucket_max", worst_bucket, g["ece_per_bucket_max"], worst_bucket <= g["ece_per_bucket_max"])
    check("noul_accuracy_min", metrics.get("noul_accuracy"), g["noul_accuracy_min"],
          (metrics.get("noul_accuracy") or 0) >= g["noul_accuracy_min"])
    check("choice_accuracy_min", metrics.get("choice_accuracy"), g["choice_accuracy_min"],
          (metrics.get("choice_accuracy") or 0) >= g["choice_accuracy_min"])
    check("precision_pay_to_withdraw_min", metrics.get("precision_pay_to_withdraw"), g["precision_pay_to_withdraw_min"],
          (metrics.get("precision_pay_to_withdraw") or 0) >= g["precision_pay_to_withdraw_min"],
          "a false 'pay-to-withdraw' alarm is the fastest way to lose a user's trust")
    check("recall_pay_to_withdraw_min", metrics.get("recall_pay_to_withdraw"), g["recall_pay_to_withdraw_min"],
          (metrics.get("recall_pay_to_withdraw") or 0) >= g["recall_pay_to_withdraw_min"])
    check("recall_impersonation_min", metrics.get("recall_impersonation"), g["recall_impersonation_min"],
          (metrics.get("recall_impersonation") or 0) >= g["recall_impersonation_min"])
    check("false_positive_rate_negatives_max", metrics.get("false_positive_rate_negatives"),
          g["false_positive_rate_negatives_max"],
          (metrics.get("false_positive_rate_negatives") if metrics.get("false_positive_rate_negatives") is not None else 1)
          <= g["false_positive_rate_negatives_max"], "hard negatives are legitimate advisories")
    if extra:
        for k, v in extra.items():
            if k in g and isinstance(v, (int, float)):
                ok = v >= g[k]
                check(k, v, g[k], ok)
            elif k == "tokenizer_parity_exact":
                check(k, v, g[k], bool(v) is True)
            elif k == "coverage":
                check("abstention_coverage_min", v, g["abstention_coverage_min"], v >= g["abstention_coverage_min"])
    failures = [c["name"] for c in checks if not c["ok"]]
    return {"passed": not failures, "failures": failures, "checks": checks, "gates": g}
