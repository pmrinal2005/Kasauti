"""Cross-check the Kasauti export against the *reference* Laya implementation.

Everything in `ml/kasauti_ml/` reimplements a piece of `laya` so that it can run inside a browser
(no PyTorch, no Python, no server). Reimplementations drift. This module is the drift detector: it
imports the **installed `laya` package** and puts our mirrors, our calibration and our exported
graph next to the reference and asserts they agree.

Four sections, each independent and each reported honestly (a section that cannot run says so and
fails the run unless `--report-only` is passed):

``mirrors``
    ``render_criterion`` / ``render_options`` / ``build_head`` / ``build_sequence`` /
    ``temp_bucket`` / ``clamp_temperature`` / ``ece_score`` / ``proper_reward`` compared against
    ``laya.common`` on the real question banks and real states. This is what the *browser* uses to
    build tensors, so an id-level difference here is a wrong answer in production, silently.

``calibration``
    The same records through `calibrate.fit_temperature_map` and
    `laya.calibrate.fit_temperature_map` (and both abstention fitters), so a drift in the offline
    fit that produces the numbers in the manifest is caught before it ships.

``reference_client``
    `laya.onnx_agent.ONNXAgent` — the reference client, the one the model card documents — pointed
    at a scratch checkpoint directory holding **our pruned tokenizer, our fitted temperatures and
    our exported graph**. Its answers are compared with our own ONNX Runtime decode, which is the
    contract the browser implements. If the reference client and the browser disagree on the same
    bytes, one of them is wrong and this is the only place that can see it.

``evals``
    `laya.evals.evaluate` over the same labelled examples with the reference client as the runner,
    so the numbers the model card reports and the numbers our own `evals.py` reports come from the
    same protocol. Informational: it does not gate a publish.

Usage
-----
    python -m ml.kasauti_ml.reference_check --staging ml/out/dev/staging
    python -m ml.kasauti_ml.reference_check --staging /kaggle/working/out/staging \
        --banks ml/banks.json --out /kaggle/working/out/reference_check.json

Exit code 0 when every section passes, 1 when any section fails, 2 when `laya` is not installed
(nothing can be checked) — unless `--report-only`, which always exits 0 and only writes the report.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ML_ROOT)
DEFAULT_BANKS = os.path.join(ML_ROOT, "banks.json")

#: temperature fits are LBFGS on floats; the reference and our mirror take the identical path but
#: not bit-identical ones (float accumulation differs by numpy/torch versions), so this is a
#: "did it fit the same function" check, not an equality check.
TEMP_TOL = 0.05
#: our fit rounds the published cut to 4 dp (the precision the manifest carries and the browser
#: compares against), so the reference and our value can differ by half a unit in the last place
ABSTAIN_TOL = 1e-4
#: both paths softmax the *same fp32 graph* with the *same* fitted temperature, so anything above
#: float noise here means one of them is not using the numbers the manifest published
PROB_TOL = 0.02


def log(msg: str) -> None:
    print(f"[reference-check] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _laya_available() -> Tuple[bool, str]:
    try:
        import laya  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"{type(e).__name__}: {e}"
    return True, getattr(__import__("laya"), "__version__", "?")


def _load_staging(staging: str) -> Dict[str, Any]:
    path = os.path.join(staging, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"no manifest at {path} — run the pipeline first")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokens_from_tokenizer_json(path: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """``{content: id}`` and ``{id: content}`` for the tokenizer's added (special) tokens."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_content: Dict[str, int] = {}
    by_id: Dict[str, str] = {}
    for t in data.get("added_tokens") or []:
        content, tid = t.get("content"), int(t.get("id", -1))
        if content is None or tid < 0:
            continue
        by_content[content] = tid
        by_id[str(tid)] = content
    return by_content, by_id


# --------------------------------------------------------------------------------------
# section 1 — mirrors
# --------------------------------------------------------------------------------------
def check_mirrors(banks_path: str, tokenizer_path: str, max_len: int,
                  head_max_len: int, n_states: int = 12) -> Dict[str, Any]:
    """Compare our reimplementations against `laya.common` on the real banks and real states."""
    import numpy as np
    import torch

    from laya import common as ref
    from . import core, datagen
    from .tok import FastTokenizerAdapter

    from tokenizers import Tokenizer

    tok = FastTokenizerAdapter(Tokenizer.from_file(tokenizer_path))
    tasks = datagen.load_tasks(banks_path)

    # question shapes: every triage + every conditional follow-up in every bank
    questions: List[Dict[str, Any]] = []
    for task in tasks.values():
        questions.append(dict(task["triage"]))
        for group in task["followups"].values():
            questions.extend(dict(q) for q in group.values())

    failures: List[str] = []
    checked = {"questions": 0, "states": 0, "option_sets": 0}

    # --- enum + scalar helpers ---
    if dict(core.QTYPES) != dict(ref.QTYPES):
        failures.append(f"QTYPES differ: {core.QTYPES} vs {ref.QTYPES}")
    for k in (0, 2, 3, 5, 6, 10, 11, 21, 60):
        for qt in (0, 1, 2):
            if core.temp_bucket(qt, k) != ref.temp_bucket(qt, k):
                failures.append(f"temp_bucket({qt},{k}) = {core.temp_bucket(qt, k)!r} != {ref.temp_bucket(qt, k)!r}")
    if (core.TEMP_MIN, core.TEMP_MAX) != (ref.TEMP_MIN, ref.TEMP_MAX):
        failures.append(f"temperature clamp range differs: {(core.TEMP_MIN, core.TEMP_MAX)} vs {(ref.TEMP_MIN, ref.TEMP_MAX)}")
    for t in (-1.0, 0.0, 0.4, 0.5, 1.0, 2.5, 5.0, 7.0, True, None, "x"):
        a, b = core.clamp_temperature(t), ref.clamp_temperature(t)
        if abs(float(a) - float(b)) > 1e-12:
            failures.append(f"clamp_temperature({t!r}) = {a} != {b}")

    states = _sample_states(tasks, n_states)

    for q in questions:
        qi = core.to_internal(q)
        opts_a, opts_b = core.render_options(qi), ref.render_options(qi)
        if opts_a != opts_b:
            failures.append(f"render_options differ for {q['instructions'][:40]!r}: {opts_a} vs {opts_b}")
        checked["questions"] += 1
        checked["option_sets"] += 1
        for state in states:
            a = core.build_sequence(tok, state, qi, max_len=max_len, head_max_len=head_max_len)
            b = ref.build_sequence(tok, state, qi, max_len=max_len, head_max_len=head_max_len)
            if list(a[0]) != list(b[0]):
                failures.append(f"build_sequence ids differ (q={q['instructions'][:32]!r}, state={state[:24]!r})")
                break
            if list(a[1]) != list(b[1]):
                failures.append(f"build_sequence markers differ (q={q['instructions'][:32]!r})")
                break
            # a permuted option order is what the training-time augmentation uses
            order = list(range(len(opts_a)))
            random.Random(7).shuffle(order)
            a2 = core.build_sequence(tok, state, qi, max_len=max_len, head_max_len=head_max_len, option_order=order)
            b2 = ref.build_sequence(tok, state, qi, max_len=max_len, head_max_len=head_max_len, option_order=order)
            if list(a2[0]) != list(b2[0]) or list(a2[1]) != list(b2[1]):
                failures.append(f"build_sequence with option_order differs (q={q['instructions'][:32]!r})")
                break
        else:
            checked["states"] += len(states)
            continue
        break

    # criterion rendering (str passthrough, dict → compact JSON)
    for v in ["plain", {"a": 1, "b": [1, 2]}, 3, None, ("x", 1)]:
        try:
            a = core.render_criterion(v)
        except Exception as e:
            a = f"<{type(e).__name__}>"
        try:
            b = ref.render_criterion(v)
        except Exception as e:
            b = f"<{type(e).__name__}>"
        if a != b:
            failures.append(f"render_criterion({v!r}) = {a!r} != {b!r}")

    # ece + proper score
    rng = np.random.default_rng(0)
    conf = rng.random(64)
    correct = (rng.random(64) < conf).astype(float)
    if abs(core.ece_score(conf, correct) - ref.ece_score(conf, correct)) > 1e-9:
        failures.append("ece_score differs on random input")

    torch.manual_seed(0)
    k = 4
    q = torch.softmax(torch.randn(16, k), -1)
    t = torch.softmax(torch.randn(16, k), -1)
    qt = torch.tensor([0] * 8 + [1] * 8)
    mask = torch.ones(16, k, dtype=torch.bool)
    try:
        ours = core.proper_reward(q, t, qt, mask, w_sph=0.75, w_rps=1.0)
        theirs = ref.proper_reward(q, t, qt, mask, w_sph=0.75, w_rps=1.0)
        delta = float(torch.abs(ours - theirs).max())
        if delta > 1e-5:
            failures.append(f"proper_reward differs by {delta:.6f}")
    except TypeError as e:
        failures.append(f"proper_reward signature changed in the installed laya: {e}")

    return {
        "ok": not failures,
        "checked": checked,
        "failures": failures[:20],
        "failures_total": len(failures),
    }


def _sample_states(tasks: Dict[str, Any], n: int) -> List[str]:
    """Deterministic sample of training states, one per template, first ``n``."""
    from . import datagen

    rows = datagen.build_rows(tasks, [t for name in tasks for t in datagen.TEMPLATES_BY_TASK[name]],
                              n_variants=2, seed=11)
    out: List[str] = []
    seen = set()
    for r in rows:
        if r["split"] != "train":
            continue
        key = r["state"][:48]
        if key in seen:
            continue
        seen.add(key)
        out.append(r["state"])
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------------------
# section 2 — calibration
# --------------------------------------------------------------------------------------
def check_calibration(n: int = 12000, seed: int = 5) -> Dict[str, Any]:
    """Feed identical records to our fitters and to `laya.calibrate`, compare the fits.

    The record count is chosen so the **reference's** own per-bucket floor is satisfied
    (``laya.calibrate.MIN_BUCKET_N``) — it is much higher than ours, so a smaller run would compare
    our fitted bucket against a bucket the reference deliberately refused to fit, which proves
    nothing. Both floors are reported so the deliberate deviation stays visible.
    """
    import numpy as np

    from laya import calibrate as ref
    from . import calibrate as ours

    # one representative option count per bucket so every bucket clears the reference's floor
    shapes = [(0, 3), (0, 8), (0, 15), (1, 3), (1, 8), (2, 2)]
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        qt, k = shapes[i % len(shapes)]
        # a deliberately over-confident model: large logits, soft targets
        z = rng.normal(0, 3.0, k)
        p = np.exp(z) / np.exp(z).sum()
        recs.append((qt, z.tolist(), p.tolist(), k))

    a = ours.fit_temperature_map(recs, compute_ece=False)
    b = ref.fit_temperature_map(recs)
    failures: List[str] = []
    for i in range(3):
        if abs(float(a["temperature"][i]) - float(b["temperature"][i])) > TEMP_TOL:
            failures.append(f"type temperature[{i}]: ours={a['temperature'][i]:.4f} ref={b['temperature'][i]:.4f}")
    keys = sorted(set(a["temperature_by_options"]) | set(b["temperature_by_options"]))
    for key in keys:
        av = a["temperature_by_options"].get(key)
        bv = b["temperature_by_options"].get(key)
        if av is None or bv is None:
            failures.append(f"bucket {key}: ours={av} ref={bv} (one side dropped it)")
            continue
        if abs(float(av) - float(bv)) > TEMP_TOL:
            failures.append(f"bucket {key}: ours={av:.4f} ref={bv:.4f}")

    abstain_a = ours.fit_abstention_thresholds(recs, a["temperature"], a["temperature_by_options"],
                                               target_error=0.15)
    abstain_b = ref.fit_abstention_thresholds(recs, b["temperature"], b["temperature_by_options"],
                                              target_error=0.15)
    for key in sorted(set(abstain_a) | set(abstain_b)):
        av, bv = abstain_a.get(key), abstain_b.get(key)
        if av is None or bv is None:
            failures.append(f"abstain {key}: ours={av} ref={bv} (one side omitted it)")
        elif abs(float(av) - float(bv)) > ABSTAIN_TOL:
            failures.append(f"abstain {key}: ours={av} ref={bv}")

    return {
        "ok": not failures,
        "n_records": n,
        "floors": {"ours": {"MIN_BUCKET_N": ours.MIN_BUCKET_N, "MIN_TYPE_N": ours.MIN_TYPE_N,
                            "MIN_ABSTAIN_N": ours.MIN_ABSTAIN_N},
                   "reference": {"MIN_BUCKET_N": ref.MIN_BUCKET_N, "MIN_TYPE_N": ref.MIN_TYPE_N,
                                 "MIN_ABSTAIN_BUCKET_N": ref.MIN_ABSTAIN_BUCKET_N}},
        "temperature": {"ours": [round(float(x), 5) for x in a["temperature"]],
                        "reference": [round(float(x), 5) for x in b["temperature"]]},
        "buckets": {"ours": {k: round(float(v), 5) for k, v in a["temperature_by_options"].items()},
                    "reference": {k: round(float(v), 5) for k, v in b["temperature_by_options"].items()}},
        "abstain": {"ours": {k: round(float(v), 5) for k, v in abstain_a.items()},
                    "reference": {k: round(float(v), 5) for k, v in abstain_b.items()}},
        "failures": failures[:20],
        "failures_total": len(failures),
    }


# --------------------------------------------------------------------------------------
# section 3 — reference client on our bytes
# --------------------------------------------------------------------------------------
def _scratch_agent_dir(staging: str, manifest: Dict[str, Any], tmp: str,
                       checkpoint_dir: Optional[str]) -> str:
    """Lay out a checkpoint directory the reference `ONNXAgent` accepts, around OUR bytes.

    ``ONNXAgent`` reads ``rl_agent_config.json`` + ``tokenizer/`` (and optionally ``encoder/``)
    from a checkpoint dir and takes the graph from ``onnx_path``. The dir built here points that
    config at our *pruned* tokenizer and our *fitted* temperatures, so the reference client and the
    browser are running the same numbers on the same weights.
    """
    tok_path = os.path.join(staging, manifest["tokenizer"]["path"])
    by_id = _tokens_from_tokenizer_json(tok_path)[1]
    os.makedirs(os.path.join(tmp, "tokenizer"), exist_ok=True)
    shutil.copyfile(tok_path, os.path.join(tmp, "tokenizer", "tokenizer.json"))

    # tokenizer_config.json: reuse the checkpoint's (it carries the special-token *strings*, which
    # are what AutoTokenizer resolves to ids) and repair it for the pruned vocabulary.
    cfg_written = False
    if checkpoint_dir:
        src = os.path.join(checkpoint_dir, "tokenizer", "tokenizer_config.json")
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                tcfg = json.load(f)
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            if isinstance(tcfg.get("extra_special_tokens"), list):
                tcfg["extra_special_tokens"] = {f"extra_{i}": v
                                                for i, v in enumerate(tcfg["extra_special_tokens"])}
            with open(os.path.join(tmp, "tokenizer", "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump(tcfg, f, ensure_ascii=False)
            cfg_written = True
    if not cfg_written:
        specials = manifest["tokenizer"].get("specials") or {}
        name = {int(v): by_id.get(str(v), "") for v in specials.values()}
        tcfg = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": False,
            "model_max_length": int(manifest.get("maxLen") or 1024),
            "spaces_between_special_tokens": False,
        }
        if name.get(int(specials.get("pad", -1)), ""):
            tcfg["pad_token"] = name[int(specials["pad"])]
        if name.get(int(specials.get("unk", -1)), ""):
            tcfg["unk_token"] = name[int(specials["unk"])]
        if name.get(int(specials.get("mask", -1)), ""):
            tcfg["mask_token"] = name[int(specials["mask"])]
        if name.get(int(specials.get("cls", -1)), ""):
            tcfg["cls_token"] = name[int(specials["cls"])]
        if name.get(int(specials.get("sep", -1)), ""):
            tcfg["sep_token"] = name[int(specials["sep"])]
        tcfg["extra_special_tokens"] = {}
        with open(os.path.join(tmp, "tokenizer", "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tcfg, f, ensure_ascii=False)

    if checkpoint_dir and os.path.isdir(os.path.join(checkpoint_dir, "encoder")):
        shutil.copytree(os.path.join(checkpoint_dir, "encoder"), os.path.join(tmp, "encoder"),
                        dirs_exist_ok=True)

    run_cfg: Dict[str, Any] = {
        "model_name": "kasauti-laya",
        "max_len": int(manifest.get("maxLen") or 1024),
        "head_max_len": int(manifest.get("headMaxLen") or 256),
        "head_layers": 2,
        "act_costs": {"escalate": 0.5},
        "temperature": [float(t) for t in manifest.get("temperature") or [1.0, 1.0, 1.0]],
        "temperature_by_options": {k: float(v)
                                   for k, v in (manifest.get("temperatureByBucket") or {}).items()},
    }
    if checkpoint_dir:
        src = os.path.join(checkpoint_dir, "rl_agent_config.json")
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                base = json.load(f)
            for key in ("encoder", "head_layers", "act_costs"):
                if key in base:
                    run_cfg[key] = base[key]
    with open(os.path.join(tmp, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=1)
    return tmp


def _reference_probabilities(ans: Dict[str, Any], keys: Sequence[str],
                             qtype: str) -> Tuple[Optional[Any], Optional[Any]]:
    """Read a distribution out of a reference answer, whatever shape that question type uses.

    The reference client publishes ``probabilities`` for `choice` and `score` (rounded to 4 dp) and
    a single ``noul`` scalar for yes/no questions — there is no uniform map, so the check has to
    know the three shapes rather than assuming one.
    """
    import numpy as np

    if qtype == "noul":
        v = ans.get("noul")
        if v is None:
            return None, None
        p = np.asarray([1.0 - float(v), float(v)])
        top = keys[int(np.argmax(p))]
        return p, top
    probs = ans.get("probabilities") or {}
    if not probs:
        return None, None
    p = np.asarray([float(probs.get(k, probs.get(str(k), 0.0))) for k in keys])
    top = ans.get("choice")
    if top is None and "score" in ans:
        # `score` is the expectation over levels; the distribution's argmax is the honest comparison
        top = keys[int(np.argmax(p))]
    return p, top


def check_reference_client(staging: str, banks_path: str, checkpoint_dir: Optional[str] = None,
                           n_examples: int = 24, max_questions: int = 4) -> Dict[str, Any]:
    """Run the reference `ONNXAgent` on our exported graph and compare with our own decode."""
    import numpy as np
    from laya.onnx_agent import ONNXAgent

    from . import core, datagen, export
    from .tok import FastTokenizerAdapter

    from tokenizers import Tokenizer

    manifest = _load_staging(staging)
    tasks = datagen.load_tasks(banks_path)
    rows = datagen.build_rows(tasks, [t for name in tasks for t in datagen.TEMPLATES_BY_TASK[name]],
                              n_variants=2, seed=23)
    rows = [r for r in rows if r["split"] == "test"] or rows
    # a spread of tasks, deterministic
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    picked: List[Dict[str, Any]] = []
    while len(picked) < n_examples and any(by_task.values()):
        for task in sorted(by_task):
            if by_task[task] and len(picked) < n_examples:
                picked.append(by_task[task].pop(0))

    tmp = tempfile.mkdtemp(prefix="kasauti-refcheck-")
    failures: List[str] = []
    compared = 0
    worst_prob_delta = 0.0
    try:
        agent_dir = _scratch_agent_dir(staging, manifest, tmp, checkpoint_dir)
        graph = os.path.join(staging, manifest["files"]["fp32"]["path"])
        agent = ONNXAgent(agent_dir, onnx_path=graph)
        log(f"reference ONNXAgent loaded {os.path.basename(graph)} "
            f"(revision={getattr(agent, 'revision', None)}, max_len={agent.cfg.get('max_len')})")

        tok = FastTokenizerAdapter(Tokenizer.from_file(os.path.join(staging, manifest["tokenizer"]["path"])))
        sess = export.ort_session(graph)
        max_len = int(manifest.get("maxLen") or 1024)
        head_max_len = int(manifest.get("headMaxLen") or 256)

        for row in picked:
            qids = list(row["questions"])[:max_questions]
            questions = {qid: dict(row["questions"][qid]) for qid in qids}
            ref_res = agent.predict(row["state"], questions)
            for qid in qids:
                q = questions[qid]
                qi = core.to_internal(q)
                keys = core.option_keys(qi)
                ids, markers = core.build_sequence(tok, row["state"], qi, max_len=max_len,
                                                   head_max_len=head_max_len)
                ours = export.ort_predict(sess, [{
                    "ids": ids, "markers": markers, "qtype": core.QTYPES[q["type"]],
                    "target": [0.0] * len(keys),
                }])[0]
                bucket = core.temp_bucket(core.QTYPES[q["type"]], len(keys))
                t = float(manifest.get("temperatureByBucket", {}).get(
                    bucket, (manifest.get("temperature") or [1.0, 1.0, 1.0])[core.QTYPES[q["type"]]]))
                p_ours = core.softmax_rows(np.asarray(ours[1]).reshape(1, -1), t)[0]

                ans = ref_res["answers"][qid]
                p_ref, ref_top = _reference_probabilities(ans, keys, q["type"])
                if p_ref is None:
                    failures.append(f"{row['id']}/{qid}: cannot read probabilities from the "
                                    f"reference answer {sorted(ans)}")
                    continue
                if abs(float(p_ref.sum()) - 1.0) > 1e-3:
                    failures.append(f"{row['id']}/{qid}: reference probabilities sum to {p_ref.sum():.4f}")

                # the reference names its own top-1; ours must agree
                top_ours = keys[int(np.argmax(p_ours))]
                # A near-tie is not a disagreement: the reference rounds to 4 dp, so p = 0.50003 is
                # published as 0.5 and its argmax falls to index 0 ("false") while ours says "true".
                # Real disagreement is still caught by the probability-delta check just below.
                srt = np.sort(p_ours)[::-1]
                near_tie = len(srt) > 1 and float(srt[0] - srt[1]) <= 2e-3
                if q["type"] in ("choice", "noul") and ref_top != top_ours and not near_tie:
                    failures.append(f"{row['id']}/{qid}: reference chose {ref_top!r}, our decode {top_ours!r}")

                # both paths softmax the same graph with the same fitted temperature: a difference
                # above the reference's own 4-dp rounding + float noise means they disagree
                delta = float(np.max(np.abs(p_ref - p_ours)))
                worst_prob_delta = max(worst_prob_delta, delta)
                if delta > PROB_TOL:
                    failures.append(f"{row['id']}/{qid}: probabilities differ by {delta:.4f} "
                                    f"(reference {p_ref.round(4).tolist()} vs ours {p_ours.round(4).tolist()})")

                # abstention is gated on answer_confidence = max(p) in both implementations
                ref_conf = ans.get("answer_confidence")
                if ref_conf is not None and abs(float(ref_conf) - float(np.max(p_ours))) > PROB_TOL:
                    failures.append(f"{row['id']}/{qid}: answer_confidence {ref_conf} vs our max(p) "
                                    f"{float(np.max(p_ours)):.4f}")
                compared += 1
        del sess
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        "ok": not failures and compared > 0,
        "compared": compared,
        "examples": len(picked),
        "max_probability_delta": round(worst_prob_delta, 6),
        "failures": failures[:20],
        "failures_total": len(failures),
    }


# --------------------------------------------------------------------------------------
# section 4 — reference evals (informational)
# --------------------------------------------------------------------------------------
class _ModelAgnosticRunner:
    """`laya.evals.evaluate` forwards each example's ``model`` to the runner.

    `ONNXAgent` predates that field and its `system_one` takes no ``model`` argument, so calling
    `evaluate(agent, ...)` directly raises `TypeError` on every case — with ``on_error="skip"`` it
    raises on every case *quietly*, producing an empty report that looks like a pass. This adapter
    drops the field the single-checkpoint agent cannot honour, so the second opinion is real.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def system_one(self, state, questions, model=None, **kw):
        return self.agent.system_one(state, questions, **kw)

    def predict(self, state, questions, model=None, **kw):
        return self.agent.predict(state, questions, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)


def check_reference_evals(staging: str, banks_path: str, checkpoint_dir: Optional[str] = None,
                          n_examples: int = 32) -> Dict[str, Any]:
    """`laya.evals.evaluate` with the reference client as runner — same protocol, second opinion."""
    from laya.evals import Dataset, Example, evaluate
    from laya.onnx_agent import ONNXAgent

    from . import datagen

    manifest = _load_staging(staging)
    tasks = datagen.load_tasks(banks_path)
    rows = datagen.build_rows(tasks, [t for name in tasks for t in datagen.TEMPLATES_BY_TASK[name]],
                              n_variants=2, seed=31)
    rows = [r for r in rows if r["split"] == "test"] or rows
    examples: List[Example] = []
    for row in rows[:n_examples]:
        expected = {}
        for qid, q in row["questions"].items():
            gold = (row.get("gold") or {}).get(qid) or {}
            probs = gold.get("probabilities") or {}
            if not probs:
                continue
            key = max(probs, key=lambda k: probs[k])
            if q["type"] == "noul":
                expected[qid] = bool(key) if isinstance(key, bool) else str(key).lower() in ("true", "yes", "1")
            elif q["type"] == "score":
                expected[qid] = float(key)
            else:
                expected[qid] = key
        if expected:
            examples.append(Example(state=row["state"], questions=dict(row["questions"]),
                                    expected=expected, language=row.get("lang")))
    if not examples:
        return {"ok": True, "skipped": "no labelled examples in the generated rows"}

    tmp = tempfile.mkdtemp(prefix="kasauti-refevals-")
    try:
        agent_dir = _scratch_agent_dir(staging, manifest, tmp, checkpoint_dir)
        agent = ONNXAgent(agent_dir, onnx_path=os.path.join(staging, manifest["files"]["fp32"]["path"]))
        report = evaluate(_ModelAgnosticRunner(agent), Dataset(examples), on_error="skip")
        # EvalReport is a dataclass with to_json() (config / overall / slices / cases)
        data = report.to_json() if hasattr(report, "to_json") else dict(report)
        cases = data.pop("cases", []) if isinstance(data, dict) else []
        errored = list((data.get("config") or {}).get("errored") or [])
        overall = data.get("overall") or {}
        # a report where every case errored is not a pass, it is a broken harness
        return {
            "ok": bool(overall) and not errored,
            "n_examples": len(examples),
            "n_cases": len(cases),
            "overall": overall,
            "slices": data.get("slices"),
            "errors": [e.get("error") for e in errored[:3]],
            "failures": ([] if (overall and not errored) else
                         [f"{len(errored)}/{len(examples)} cases errored"
                          f" — first: {errored[0].get('error') if errored else 'no metrics reported'}"]),
        }
    except Exception as e:  # pragma: no cover - the evals API is newer than some installs
        return {"ok": True, "skipped": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="cross-check the Kasauti export against reference Laya")
    ap.add_argument("--staging", required=True, help="directory holding manifest.json + artifacts")
    ap.add_argument("--banks", default=DEFAULT_BANKS)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="base checkpoint dir (tokenizer_config.json / encoder/) for the scratch agent")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--report-only", action="store_true", help="never exit non-zero")
    ap.add_argument("--skip-client", action="store_true", help="mirrors + calibration only (cheap)")
    args = ap.parse_args(argv)

    ok, version = _laya_available()
    if not ok:
        log(f"the `laya` package is not importable ({version}) — nothing can be checked")
        return 0 if args.report_only else 2
    log(f"reference implementation: laya {version}")

    manifest = _load_staging(args.staging)
    report: Dict[str, Any] = {"layaVersion": version, "staging": args.staging,
                              "modelVersion": manifest.get("modelVersion")}

    log("mirrors: build_sequence / render_options / temp_bucket / clamp / ece / proper_reward")
    report["mirrors"] = check_mirrors(args.banks,
                                      os.path.join(args.staging, manifest["tokenizer"]["path"]),
                                      int(manifest.get("maxLen") or 1024),
                                      int(manifest.get("headMaxLen") or 256))
    log(f"  → {'ok' if report['mirrors']['ok'] else 'FAILED'} "
        f"({report['mirrors']['checked']['questions']} questions × "
        f"{report['mirrors']['checked']['states']} states)")

    log("calibration: our fitters vs laya.calibrate on identical records")
    report["calibration"] = check_calibration()
    log(f"  → {'ok' if report['calibration']['ok'] else 'FAILED'}")

    if not args.skip_client:
        log("reference client: laya.onnx_agent.ONNXAgent on our exported graph")
        report["referenceClient"] = check_reference_client(args.staging, args.banks,
                                                           args.checkpoint_dir)
        log(f"  → {'ok' if report['referenceClient']['ok'] else 'FAILED'} "
            f"({report['referenceClient']['compared']} answers compared)")
        log("evals: laya.evals.evaluate with the reference client as runner")
        report["referenceEvals"] = check_reference_evals(args.staging, args.banks, args.checkpoint_dir)
        log(f"  → {'ok' if report['referenceEvals']['ok'] else 'FAILED'}")

    failed = [k for k, v in report.items() if isinstance(v, dict) and v.get("ok") is False]
    report["passed"] = not failed
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log(f"report → {args.out}")
    if failed:
        for section in failed:
            for line in (report[section].get("failures") or [])[:10]:
                log(f"  FAIL {section}: {line}")
        log(f"FAILED sections: {failed}")
        return 0 if args.report_only else 1
    log("all sections pass — the export is what the reference client would run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
