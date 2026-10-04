"""Vocabulary pruning, ONNX export and the two browser builds (INT8 WASM, 4-bit WebGPU).

Why this file exists at all: the multilingual checkpoint is mmBERT-base — 22 layers at hidden 768
with a ~256k-entry vocabulary. Shipped as-is that is a ~650 MB download for a phone, and the
embedding table alone is most of it. Three measured, reversible transforms get it to a size a
₹9,000 Android can actually fetch:

1. **Vocabulary pruning** — keep the tokens the target languages actually emit (plus their BPE
   parts, the 256 byte-fallback tokens and the specials) and re-index the embedding rows. This is
   *lossless by construction*: rows are copied, never recomputed, and by keeping the merge closure
   the pruned tokenizer segments the corpus identically to the full one.
2. **INT8 dynamic quantization** (per-tensor, ``per_channel=False``) for the WASM build. The
   reference is explicit that per-channel scales collapse this model on the dynamic MatMulInteger
   path (issue #790: 31/96 vs 64/96 agreement on English), so per-tensor is not a shortcut, it is
   the only correct setting. INT8 also quantizes the embedding ``Gather``, which is where the
   size win is; the accuracy cost is *measured* and gated, never assumed.
3. **4-bit block quantization (MatMulNBits)** for the WebGPU build, block 32, symmetric — with the
   embedding left in fp32/fp16 so a lookup cannot be corrupted by a shared scale.

Every artifact gets a sha256 in the manifest, and both quantized builds are re-validated with ONNX
Runtime against the fp32 graph (agreement rate) before anything is publishable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Tuple

from . import core


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {"path": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    meta.update(extra or {})
    return meta


def content_hashed_name(stem: str, path: str, ext: str = ".onnx") -> str:
    """``model.int8.onnx`` -> ``model.int8.<sha8>.onnx``: content-addressed, cache-forever."""
    return f"{stem}.{sha256_file(path)[:8]}{ext}"


# --------------------------------------------------------------------------------------
# 1) vocabulary pruning
# --------------------------------------------------------------------------------------
def _load_tokenizer_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_parts(merges: Sequence[Any]) -> Dict[str, List[Tuple[str, str]]]:
    """token string → **every** (left, right) pair that can produce it.

    A BPE merge list is a multimap, not a function: in the multilingual tokenizer 177k of 235k
    merge results have more than one producer (up to 30). Keeping only the first pair seen — the
    obvious `setdefault` typing — silently walks the wrong merge chain for 1,120 of the ~1,900
    tokens an ordinary corpus uses, and the pruned tokenizer then cannot rebuild those tokens the
    way the real one did. Measured on `convaiinnovations/laya` multilingual with the full bank
    corpus, that showed up as **16.5% segmentation drift**; with every producer kept it is 0.
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    for m in merges:
        pair = m.split(" ") if isinstance(m, str) else list(m)
        if len(pair) >= 2:
            out.setdefault(pair[0] + pair[1], []).append((pair[0], pair[1]))
    return out


def prune_tokenizer(tok_json_path: str, corpus: Iterable[str], out_dir: str,
                    keep_extra: Sequence[str] = ()) -> Dict[str, Any]:
    """Write a pruned ``tokenizer.json`` + an old→new id map. Returns a stats dict."""
    from tokenizers import Tokenizer

    data = _load_tokenizer_json(tok_json_path)
    model = data.get("model") or {}
    vocab: Dict[str, int] = model.get("vocab") or {}
    merges: List[Any] = model.get("merges") or []
    if not vocab:
        raise ValueError("tokenizer.json has no model.vocab")

    tok = Tokenizer.from_file(tok_json_path)
    used: set = set()
    n_texts = 0
    for text in corpus:
        n_texts += 1
        used.update(tok.encode(text, add_special_tokens=False).ids)

    keep: set = set(used)
    keep.update(v for _, v in ((s, i) for s, i in [(t, vocab.get(t, -1)) for t in keep_extra] if i >= 0))
    # specials and added tokens
    added = data.get("added_tokens") or []
    for a in added:
        if isinstance(a.get("id"), int):
            keep.add(int(a["id"]))
    for name in ("unk_token", "pad_token", "mask_token", "eos_token", "bos_token"):
        t = model.get(name)
        if isinstance(t, str) and t in vocab:
            keep.add(vocab[t])
    # byte-fallback tokens are the safety net for anything the pruned vocab cannot spell
    for t, i in vocab.items():
        if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
            keep.add(i)
    # every single character present in the corpus (so unknown words still decompose)
    for text in corpus:
        for ch in text:
            i = vocab.get(ch)
            if i is not None:
                keep.add(i)
    # BPE closure: a kept merged token needs the parts of *every* merge that can produce it,
    # recursively, or the pruned tokenizer reassembles the token a different way (or not at all)
    producers = _merge_parts(merges)
    stack = [t for t, i in vocab.items() if i in keep]
    while stack:
        t = stack.pop()
        for pair in producers.get(t, ()):
            for p in pair:
                i = vocab.get(p)
                if i is not None and i not in keep:
                    keep.add(i)
                    stack.append(p)

    new_tokens = sorted(keep, key=lambda i: i)
    old_to_new = {old: new for new, old in enumerate(new_tokens)}
    new_vocab = {t: old_to_new[i] for t, i in vocab.items() if i in old_to_new}
    new_merges = [m for m in merges
                  if (m if isinstance(m, str) else " ".join(m)).replace(" ", "") in new_vocab
                  and all(p in new_vocab for p in (m.split(" ") if isinstance(m, str) else list(m))[:2])]

    out = json.loads(json.dumps(data))  # deep copy, keep every other field intact
    out["model"]["vocab"] = new_vocab
    out["model"]["merges"] = new_merges
    for a in out.get("added_tokens") or []:
        if isinstance(a.get("id"), int) and a["id"] in old_to_new:
            a["id"] = old_to_new[a["id"]]
    for name in ("unk_token", "pad_token", "mask_token", "eos_token", "bos_token"):
        t = out["model"].get(name)
        if isinstance(t, str) and t not in new_vocab:
            out["model"][name] = None

    os.makedirs(out_dir, exist_ok=True)
    tok_out = os.path.join(out_dir, "tokenizer.json")
    with open(tok_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    # remap table (old id -> new id, -1 = dropped) for gathering embedding rows
    remap = [-1] * (max(vocab.values()) + 1)
    for old, new in old_to_new.items():
        remap[old] = new

    # verify: the pruned tokenizer must segment the corpus exactly like the full one (after remap)
    pruned = Tokenizer.from_file(tok_out)
    drift = 0
    checked = 0
    for text in corpus:
        checked += 1
        a = [remap[i] for i in tok.encode(text, add_special_tokens=False).ids]
        b = pruned.encode(text, add_special_tokens=False).ids
        if a != b:
            drift += 1
    stats = {
        "vocab_before": len(vocab),
        "vocab_after": len(new_vocab),
        "merges_before": len(merges),
        "merges_after": len(new_merges),
        "texts": n_texts,
        "segmentation_drift": drift,
        "drift_rate": round(drift / max(1, checked), 5),
        "tokenizer_json": tok_out,
        "tokenizer_json_bytes": os.path.getsize(tok_out),
    }
    with open(os.path.join(out_dir, "vocab_remap.json"), "w") as f:
        json.dump({"remap": remap, "vocab_after": len(new_vocab)}, f)
    return stats


def _kept_old_ids(remap: Sequence[int]) -> List[int]:
    """Old ids in *new-id order* (``remap[old] = new``; -1 = dropped)."""
    kept = sorted(((new, old) for old, new in enumerate(remap) if new >= 0))
    if [n for n, _ in kept] != list(range(len(kept))):
        raise ValueError("vocab remap is not a dense 0..n-1 renumbering")
    return [old for _, old in kept]


def gather_embeddings(state_dict: Dict[str, Any], remap: Sequence[int]) -> Dict[str, Any]:
    """Rewrite every ``vocab_size``-row parameter to the pruned order (lossless row copy).

    Kept for callers that operate on a bare state dict. Matching is by **shape** (rows ==
    ``len(remap)``) rather than by parameter name: the old name heuristic missed the smoke model's
    ``encoder.emb.weight`` entirely, which silently shipped an un-pruned table that the pruned
    tokenizer's ids no longer indexed correctly. Use :func:`resize_vocab` on a live model.
    """
    import torch

    idx = torch.tensor(_kept_old_ids(remap), dtype=torch.long)
    out: Dict[str, Any] = {}
    for name, tensor in state_dict.items():
        if tensor.dim() >= 2 and tensor.shape[0] == len(remap):
            out[name] = tensor.index_select(0, idx.to(tensor.device)).contiguous()
        else:
            out[name] = tensor
    return out


def resize_vocab(model, remap: Sequence[int]) -> List[str]:
    """Shrink every token-embedding table of a live model to the pruned vocabulary, in place.

    ``load_state_dict(strict=True)`` refuses a state dict whose embedding has fewer rows than the
    module, so a gathered state dict cannot simply be loaded back (that is the crash the real
    multilingual run hit at 256 000 → ~5 000 rows). Instead each ``nn.Embedding`` with
    ``num_embeddings == len(remap)`` gets a new, smaller ``weight`` Parameter holding the kept rows
    in new-id order, and any output projection tied to it is re-pointed. Returns the module names
    that were resized; raises if none was, because a pruned tokenizer feeding an un-pruned table is
    exactly the silent misalignment this guards against.
    """
    import torch

    n_old = len(remap)
    idx = _kept_old_ids(remap)
    resized: List[str] = []
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Embedding) and mod.num_embeddings == n_old:
                old_w = mod.weight
                new_w = torch.nn.Parameter(
                    old_w.index_select(0, torch.tensor(idx, dtype=torch.long, device=old_w.device))
                    .clone().contiguous(), requires_grad=old_w.requires_grad)
                mod.weight = new_w
                mod.num_embeddings = len(idx)
                if mod.padding_idx is not None:
                    mod.padding_idx = remap[mod.padding_idx] if remap[mod.padding_idx] >= 0 else None
                resized.append(name)
                # tied output heads (MaskedLM-style) share the Parameter object
                for _n2, mod2 in model.named_modules():
                    if isinstance(mod2, torch.nn.Linear) and mod2.weight is old_w:
                        mod2.weight = new_w
                        mod2.out_features = len(idx)
    if not resized:
        raise RuntimeError(f"resize_vocab: no nn.Embedding with {n_old} rows found — the pruned "
                           f"tokenizer would index an un-pruned table")
    cfg = getattr(getattr(model, "encoder", None), "config", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        setattr(cfg, "vocab_size", len(idx))
    return resized


def remap_ids(ids: Sequence[int], remap: Sequence[int]) -> List[int]:
    """Full-tokenizer ids → pruned ids. Raises on a dropped id (it would index a missing row)."""
    out = []
    for i in ids:
        j = remap[i] if 0 <= i < len(remap) else -1
        if j < 0:
            raise ValueError(f"token id {i} was pruned but is used by an export sample")
        out.append(j)
    return out


# --------------------------------------------------------------------------------------
# 2) ONNX export
# --------------------------------------------------------------------------------------
def export_fp32(model, out_path: str, seq_len: int = 17, batch: int = 2, num_markers: int = 3,
                opset: int = 18) -> str:
    """Reference-faithful export (``scripts/export_onnx.py``): batch/seq/markers all >1 and
    pairwise different, opset 18, dynamic axes on all five inputs and both outputs.

    The dummies matter: a batch-1 dummy baked batch=1 into the head's attention and the export then
    failed at batch ≥ 2 (Laya issue #695).
    """
    import torch

    model.eval()
    dummy_input_ids = torch.randint(0, 100, (batch, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
    dummy_marker_pos = torch.tensor([[1, 5, 9]] * batch, dtype=torch.long)
    dummy_marker_mask = torch.ones((batch, num_markers), dtype=torch.bool)
    dummy_qtype = torch.zeros(batch, dtype=torch.long)
    inputs = (dummy_input_ids, dummy_attention_mask, dummy_marker_pos, dummy_marker_mask, dummy_qtype)
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    kwargs: Dict[str, Any] = dict(export_params=True, opset_version=opset,
                                  do_constant_folding=True,
                                  input_names=["input_ids", "attention_mask", "marker_pos",
                                               "marker_mask", "qtype"],
                                  output_names=["logits", "act_logits"], dynamic_axes=dynamic_axes)
    try:
        torch.onnx.export(model, inputs, out_path, dynamo=False, **kwargs)
    except TypeError:  # older torch: no `dynamo` kwarg
        torch.onnx.export(model, inputs, out_path, **kwargs)
    return out_path


def quantize_int8(src: str, dst: str, per_channel: bool = False,
                  quantize_embedding: bool = True) -> str:
    """Dynamic INT8 (CPU/WASM). ``per_channel=False`` is required (issue #790)."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = onnx.load(src)
    del model.graph.value_info[:]  # stale shapes disagree with the quantizer's own inference pass
    ops = ["MatMul", "Gather"] if quantize_embedding else ["MatMul"]
    quantize_dynamic(model_input=model, model_output=dst, op_types_to_quantize=ops,
                     weight_type=QuantType.QInt8, per_channel=per_channel)
    return dst


def quantize_q4(src: str, dst: str, block_size: int = 32, bits: int = 4) -> str:
    """4-bit block quantization for the WebGPU execution provider (MatMulNBits)."""
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer,
    )

    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import QuantFormat
    except Exception:  # pragma: no cover
        QuantFormat = None
    cfg_kwargs: Dict[str, Any] = dict(block_size=block_size, is_symmetric=True, bits=bits,
                                      op_types_to_quantize=("MatMul",), accuracy_level=4)
    if QuantFormat is not None:
        cfg_kwargs["quant_format"] = QuantFormat.QOperator
    cfg = DefaultWeightOnlyQuantConfig(**cfg_kwargs)
    quant = MatMulNBitsQuantizer(src, algo_config=cfg)
    quant.process()
    quant.model.save_model_to_file(dst, use_external_data_format=False)
    return dst


def convert_fp16(src: str, dst: str) -> Optional[str]:
    """fp16 pass for the WebGPU build (needs ``onnxconverter-common``; skipped if unavailable)."""
    try:
        from onnxconverter_common import float16
    except Exception:
        return None
    import onnx

    model = onnx.load(src)
    model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model, dst)
    return dst


# --------------------------------------------------------------------------------------
# 3) ORT re-evaluation
# --------------------------------------------------------------------------------------
def ort_session(path: str, providers: Optional[Sequence[str]] = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 1
    prov = list(providers or ["CPUExecutionProvider"])
    return ort.InferenceSession(path, sess_options=so, providers=prov)


def ort_predict(sess, items: Sequence[Dict[str, Any]]) -> List[Tuple[int, List[float], List[float]]]:
    """Run one sequence at a time (batch-size-1 shapes are what the browser sees)."""
    import numpy as np

    out: List[Tuple[int, List[float], List[float]]] = []
    for it in items:
        ids = np.asarray([it["ids"]], dtype=np.int64)
        att = np.ones_like(ids)
        k = len(it["markers"])
        mpos = np.asarray([it["markers"]], dtype=np.int64)
        mmask = np.ones((1, k), dtype=bool)
        qtype = np.asarray([it["qtype"]], dtype=np.int64)
        logits = sess.run(["logits"], {
            "input_ids": ids, "attention_mask": att, "marker_pos": mpos,
            "marker_mask": mmask, "qtype": qtype,
        })[0]
        out.append((int(it["qtype"]), [float(v) for v in logits[0, :k]], [float(v) for v in it["target"][:k]]))
    return out


def agreement(a: Sequence[Tuple[int, Sequence[float], Sequence[float]]],
              b: Sequence[Tuple[int, Sequence[float], Sequence[float]]],
              decisive_margin: float = 0.02) -> Dict[str, Any]:
    """Top-1 agreement + mean absolute logit delta between two prediction sets.

    ``top1_agreement`` counts every item, so it is dominated by near-ties: on an item whose two
    best options are 0.501 / 0.499 under the reference graph, any quantization noise at all flips
    the argmax without changing a single decision the runtime would act on (the abstention cut
    sits far above a coin-flip). ``decisive_top1_agreement`` is the same number restricted to items
    where the reference top-1 leads by at least ``decisive_margin`` in probability, which is what
    the export gate reads; both are reported.
    """
    import numpy as np

    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0, "top1_agreement": None, "decisive_top1_agreement": None, "n_decisive": 0,
                "mean_abs_logit_delta": None, "mean_kl": None}
    same = 0
    dec_n, dec_same = 0, 0
    deltas, kls = [], []
    for i in range(n):
        za, zb = np.asarray(a[i][1]), np.asarray(b[i][1])
        m = min(len(za), len(zb))
        za, zb = za[:m], zb[:m]
        agree = int(int(np.argmax(za)) == int(np.argmax(zb)))
        same += agree
        deltas.append(float(np.mean(np.abs(za - zb))))
        pa = core.softmax_rows(za, 1.0)[0]
        pb = core.softmax_rows(zb, 1.0)[0]
        kls.append(float(np.sum(pa * np.log(np.clip(pa, 1e-9, None) / np.clip(pb, 1e-9, None)))))
        srt = np.sort(pa)[::-1]
        if len(srt) < 2 or float(srt[0] - srt[1]) >= decisive_margin:
            dec_n += 1
            dec_same += agree
    return {"n": n, "top1_agreement": round(same / n, 4),
            "decisive_top1_agreement": (round(dec_same / dec_n, 4) if dec_n else None),
            "n_decisive": dec_n, "decisive_margin": decisive_margin,
            "mean_abs_logit_delta": round(float(np.mean(deltas)), 4),
            "mean_kl": round(float(np.mean(kls)), 4)}


def graph_io_summary(path: str) -> Dict[str, Any]:
    import onnx

    try:
        m = onnx.load(path, load_external_data=False)
    except Exception:
        m = onnx.load(path)
    return {
        "inputs": [{"name": i.name, "dtype": str(i.type.tensor_type.elem_type)} for i in m.graph.input],
        "outputs": [{"name": o.name, "dtype": str(o.type.tensor_type.elem_type)} for o in m.graph.output],
        "opset": [{"domain": op.domain, "version": op.version} for op in m.opset_import],
        "nodes": len(m.graph.node),
        "ops": sorted({n.op_type for n in m.graph.node}),
    }


def copy_with_content_hash(src: str, out_dir: str, stem: str) -> Tuple[str, Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    name = content_hashed_name(stem, src)
    dst = os.path.join(out_dir, name)
    shutil.copyfile(src, dst)
    return dst, file_meta(dst)
