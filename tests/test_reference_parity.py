"""Parity tests: the mirrors in `ml/kasauti_ml/` against the installed `laya` package.

`ml/kasauti_ml/__init__.py` states the design rule this file enforces: every function that mirrors
the reference implementation says so in its docstring, and *this* module checks those mirrors
against the real thing. It is the reason the mirrored code is allowed to exist at all — the browser
cannot run PyTorch, so the tensor construction, the calibration fit and the reward have to be
reimplemented in Python and then in TypeScript, and a silent drift in any of them shows up as a
slightly worse answer rather than an error.

Run it with the package installed:

    python tests/test_reference_parity.py            # plain, no pytest needed
    python -m pytest tests/test_reference_parity.py -q

Exit codes: 0 all checks pass, 1 a check failed, 77 `laya` (or `tokenizers`) is not installed —
skipped, not failed, so a Node-only checkout stays green.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BANKS = os.path.join(REPO, "ml", "banks.json")
FIXTURE_TOKENIZER = os.path.join(REPO, "tests", "fixtures", "laya-tokenizer.mini.json")

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str) -> Callable:
    def deco(fn: Callable[[], Any]) -> Callable[[], Any]:
        def run() -> None:
            try:
                detail = fn() or ""
                RESULTS.append((name, True, str(detail)))
            except AssertionError as e:
                RESULTS.append((name, False, str(e)))
            except Exception as e:  # an error is a failure, not a skip
                RESULTS.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
        run.__name__ = fn.__name__
        return run

    return deco


def _need(mods: List[str]) -> None:
    for m in mods:
        __import__(m)


def _tokenizer():
    """A real Rust tokenizer to run both implementations against.

    The 34 MB multilingual `tokenizer.json` is not in the repo, so the fixture built by
    `ml/scripts/make_tokenizer_fixture.py` (original ids and merge ranks preserved) is used; on a
    machine where the real one is available, `KASAUTI_TOKENIZER` points the tests at it.
    """
    from tokenizers import Tokenizer

    from ml.kasauti_ml.tok import FastTokenizerAdapter

    path = os.environ.get("KASAUTI_TOKENIZER") or FIXTURE_TOKENIZER
    if not os.path.exists(path):
        raise SystemExit(f"no tokenizer fixture at {path}")
    return FastTokenizerAdapter(Tokenizer.from_file(path))


def _questions() -> List[Dict[str, Any]]:
    from ml.kasauti_ml import datagen

    tasks = datagen.load_tasks(BANKS)
    out: List[Dict[str, Any]] = []
    for task in tasks.values():
        out.append(task["triage"])
        for group in task["followups"].values():
            out.extend(group.values())
    return out


# --------------------------------------------------------------------------------------
# rendering + sequence construction (what the browser does)
# --------------------------------------------------------------------------------------
@check("QTYPES / temp_bucket / clamp_temperature match laya.common")
def t_scalars() -> str:
    from laya import common as ref

    from ml.kasauti_ml import core

    assert dict(core.QTYPES) == dict(ref.QTYPES), f"{core.QTYPES} != {ref.QTYPES}"
    assert (core.TEMP_MIN, core.TEMP_MAX) == (ref.TEMP_MIN, ref.TEMP_MAX)
    for qt in (0, 1, 2):
        for k in (0, 1, 2, 3, 5, 6, 10, 11, 20, 21, 60):
            assert core.temp_bucket(qt, k) == ref.temp_bucket(qt, k), (qt, k)
    for t in (-1.0, 0.0, 0.4, 0.5, 1.0, 2.5, 5.0, 7.0, True, None, "x"):
        assert abs(float(core.clamp_temperature(t)) - float(ref.clamp_temperature(t))) < 1e-12, t
    return "3 types × 11 option counts · 11 clamp values"


@check("render_criterion / render_options match laya.common on every bank question")
def t_render() -> str:
    from laya import common as ref

    from ml.kasauti_ml import core

    n = 0
    for q in _questions():
        qi = core.to_internal(q)
        assert core.render_options(qi) == ref.render_options(qi), q["instructions"]
        n += 1
    for v in ["plain", {"a": 1, "b": [1, 2]}, 3, None, True, ["x", 1]]:
        assert core.render_criterion(v) == ref.render_criterion(v), v
    return f"{n} questions + 6 criterion shapes"


@check("build_sequence / build_head match laya.common token-for-token")
def t_sequence() -> str:
    from laya import common as ref

    from ml.kasauti_ml import core, datagen

    tok = _tokenizer()
    tasks = datagen.load_tasks(BANKS)
    rows = datagen.build_rows(tasks, [t for name in tasks for t in datagen.TEMPLATES_BY_TASK[name]],
                              n_variants=2, seed=4)
    states = [r["state"] for r in rows[:6]]
    n = 0
    for q in _questions()[:40]:
        qi = core.to_internal(q)
        for state in states:
            a = core.build_sequence(tok, state, qi, max_len=512, head_max_len=192)
            b = ref.build_sequence(tok, state, qi, max_len=512, head_max_len=192)
            assert list(a[0]) == list(b[0]), f"ids differ for {q['instructions'][:40]!r}"
            assert list(a[1]) == list(b[1]), f"markers differ for {q['instructions'][:40]!r}"
            # the training-time option-order augmentation permutes the slots
            order = list(range(len(a[1])))[::-1]
            a2 = core.build_sequence(tok, state, qi, max_len=512, head_max_len=192, option_order=order)
            b2 = ref.build_sequence(tok, state, qi, max_len=512, head_max_len=192, option_order=order)
            assert list(a2[0]) == list(b2[0]) and list(a2[1]) == list(b2[1]), "option_order differs"
            n += 1
    return f"{n} (question, state) pairs, incl. permuted option orders"


# --------------------------------------------------------------------------------------
# confidence, scoring, calibration
# --------------------------------------------------------------------------------------
@check("ece_score and proper_reward match laya.common")
def t_metrics() -> str:
    import numpy as np
    import torch
    from laya import common as ref

    from ml.kasauti_ml import core

    rng = np.random.default_rng(1)
    conf = rng.random(128)
    correct = (rng.random(128) < conf).astype(float)
    assert abs(core.ece_score(conf, correct) - ref.ece_score(conf, correct)) < 1e-9

    torch.manual_seed(2)
    for k in (2, 5):
        q = torch.softmax(torch.randn(16, k), -1)
        t = torch.softmax(torch.randn(16, k), -1)
        qt = torch.tensor([0] * 8 + [1] * 8)
        mask = torch.ones(16, k, dtype=torch.bool)
        ours = core.proper_reward(q, t, qt, mask, w_sph=0.75, w_rps=1.0)
        theirs = ref.proper_reward(q, t, qt, mask, w_sph=0.75, w_rps=1.0)
        assert float(torch.abs(ours - theirs).max()) < 1e-5
    return "ece on 128 samples · proper_reward at k=2,5"


@check("calibration fitters agree with laya.calibrate")
def t_calibration() -> str:
    import numpy as np
    from laya import calibrate as ref

    from ml.kasauti_ml import calibrate as ours

    # enough records per bucket to clear the *reference's* floor (MIN_BUCKET_N is much higher)
    shapes = [(0, 3), (0, 8), (0, 15), (1, 3), (1, 8), (2, 2)]
    per = max(ref.MIN_BUCKET_N, 1) + 25
    rng = np.random.default_rng(3)
    recs = []
    for i in range(per * len(shapes)):
        qt, k = shapes[i % len(shapes)]
        z = rng.normal(0, 2.5, k)
        p = np.exp(z) / np.exp(z).sum()
        recs.append((qt, z.tolist(), p.tolist(), k))

    a = ours.fit_temperature_map(recs, compute_ece=False)
    b = ref.fit_temperature_map(recs)
    for i in range(3):
        assert abs(a["temperature"][i] - b["temperature"][i]) < 0.05, (i, a["temperature"][i], b["temperature"][i])
    for key in sorted(set(a["temperature_by_options"]) | set(b["temperature_by_options"])):
        av, bv = a["temperature_by_options"].get(key), b["temperature_by_options"].get(key)
        assert av is not None and bv is not None, f"{key}: {av} vs {bv}"
        assert abs(av - bv) < 0.05, (key, av, bv)

    aa = ours.fit_abstention_thresholds(recs, a["temperature"], a["temperature_by_options"],
                                        target_error=0.15)
    ba = ref.fit_abstention_thresholds(recs, b["temperature"], b["temperature_by_options"],
                                       target_error=0.15)
    for key in sorted(set(aa) | set(ba)):
        assert key in aa and key in ba, f"{key}: {aa.get(key)} vs {ba.get(key)}"
        # our published cut is rounded to 4 dp, the precision the manifest carries
        assert abs(aa[key] - ba[key]) < 1e-4, (key, aa[key], ba[key])
    return (f"{len(recs)} records · type temps {[round(x, 3) for x in a['temperature']]} · "
            f"{len(a['temperature_by_options'])} buckets")


# --------------------------------------------------------------------------------------
# export contract
# --------------------------------------------------------------------------------------
@check("the exported graph keeps the I/O contract the browser sends")
def t_graph_contract() -> str:
    from ml.kasauti_ml import datagen, export, parity

    graph = os.environ.get("KASAUTI_GRAPH")
    candidates = [graph] if graph else []
    dev = os.path.join(REPO, "public", "dev-model")
    if os.path.isdir(dev):
        man = json.load(open(os.path.join(dev, "manifest.json"), encoding="utf-8"))
        candidates.append(os.path.join(dev, man["files"]["wasm"]["path"]))
    if not candidates:
        raise SystemExit("no exported graph found — run the smoke pipeline first")
    path = candidates[0]
    io = export.graph_io_summary(path)
    names = {i["name"] for i in io["inputs"]}
    assert {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"} <= names, names
    assert {"logits"} <= {o["name"] for o in io["outputs"]}, io["outputs"]
    return f"{os.path.basename(path)}: {sorted(names)} → {[o['name'] for o in io['outputs']]}"


@check("published parity vectors replay through the reference tokenizer")
def t_parity_vectors() -> str:
    from tokenizers import Tokenizer
    from ml.kasauti_ml import parity

    dev = os.path.join(REPO, "public", "dev-model")
    pj = os.path.join(dev, "parity.json")
    if not os.path.exists(pj):
        raise SystemExit("no parity.json — run the smoke pipeline first")
    data = json.load(open(pj, encoding="utf-8"))
    tok = Tokenizer.from_file(os.path.join(dev, data.get("tokenizer", {}).get("path", "tokenizer.json")))
    checked = 0
    for v in data["tokenizer"]["vectors"]:
        ids = list(tok.encode(v["text"], add_special_tokens=False).ids)
        assert ids == list(v["ids"]), f"ids drifted for {v['text']!r}"
        checked += 1
    return f"{checked} tokenizer vectors replay exactly"


CHECKS = [t_scalars, t_render, t_sequence, t_metrics, t_calibration, t_graph_contract, t_parity_vectors]


def main() -> int:
    try:
        _need(["laya", "tokenizers", "numpy", "torch"])
    except Exception as e:
        print(f"SKIP: the `laya` reference package is not importable ({type(e).__name__}: {e})")
        print("      install it with  pip install laya[onnx]")
        return 77
    print(f"reference: laya {__import__('laya').__version__} · python {sys.version.split()[0]}\n")
    for fn in CHECKS:
        fn()
    width = max(len(n) for n, _, _ in RESULTS)
    failures = 0
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail.splitlines()[0][:110] if detail else ''}")
        if not ok:
            failures += 1
            print("\n".join("        " + line for line in detail.splitlines()[1:]))
    print(f"\n{len(RESULTS) - failures}/{len(RESULTS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
