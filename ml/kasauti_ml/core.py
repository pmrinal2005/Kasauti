"""Reference-faithful Laya primitives, reimplemented so the pipeline runs without the `laya`
package installed (CPU smoke tests, CI) and cross-checked against it when it is installed.

Mirrors: ``laya/common.py`` — ``QTYPES``, ``render_criterion``, ``render_options``, ``build_head``,
``build_sequence``, ``state_room``, ``confidence_from_probs``, ``answer_confidence``, ``ece_score``,
``temp_bucket``, ``clamp_temperature``, ``TEMP_MIN``/``TEMP_MAX``, ``proper_reward``, and the
``DecisionModel`` head (used only by the tiny smoke-test profile).

Nothing here generates text: Laya answers flat `choice` / `score` / `noul` questions only.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}

#: A fitted temperature below 1 sharpens logits instead of softening them; the reference clamps
#: to [0.5, 5.0] because a sharper-than-1 fit is never an honest calibration (issue #394).
TEMP_MIN = 0.5
TEMP_MAX = 5.0


# --------------------------------------------------------------------------------------
# text / question rendering
# --------------------------------------------------------------------------------------
def render_criterion(value: Any) -> str:
    """Mirror of ``laya.common.render_criterion``: str passthrough, anything else compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict[str, Any]) -> List[str]:
    """Mirror of ``laya.common.render_options``. `q` is an internal question dict.

    Internal shape (exactly what the reference trainer and `build_sequence` consume)::

        {"t": "choice"|"score"|"noul", "ins": "<instructions>", "crit": {...}|[...]|None}

    Runtime shape (what the app sends and the ONNX client rebuilds from a manifest) uses the
    public keys `type` / `instructions` / `criteria`; ``to_internal`` converts between them.
    """
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        return [str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label + ": " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def _resolve_noul_labels(labels: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    if labels is None:
        labels = DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    f, t = labels["false"].strip(), labels["true"].strip()
    if not f or not t or f == t:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return f, t


def option_keys(q: Dict[str, Any]) -> List[str]:
    """The answer keys of an internal question, in marker order."""
    if q["t"] == "choice":
        return [str(k) for k in q["crit"].keys()]
    if q["t"] == "score":
        return [str(i) for i in range(len(q["crit"]))]
    return ["false", "true"]


def build_head(tok, q: Dict[str, Any], head_max_len: int = 192,
               option_order: Optional[Sequence[int]] = None) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Mirror of ``laya.common.build_head`` → ``(ids, markers, stats)``.

    ``ids`` starts ``[CLS] "<type> question: <ins>" [SEP]`` and then, per option in
    ``option_order``, ``[MASK] <option tokens (≤48)>``; a closing ``[SEP]`` is appended by
    ``build_sequence``. ``markers[i]`` is the index of option ``i``'s ``[MASK]``.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = list(option_order) if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = list(tok.encode("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"])
    opt_ids: List[List[int]] = []
    for i in order:
        toks = tok.encode(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False,
                          truncation=True, max_length=48)["input_ids"]
        opt_ids.append([tok.mask_token_id] + list(toks))
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per_option = None
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        per_option = per
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + list(head_ids) + [tok.sep_token_id]
    markers: List[int] = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    return ids, markers, {
        "options": len(opt_ids),
        "options_distinct": len({tuple(o) for o in opt_ids}),
        "tokens_per_option": per_option,
    }


def build_sequence(tok, state, q: Dict[str, Any], max_len: int = 512, head_max_len: int = 192,
                   option_order: Optional[Sequence[int]] = None, truncate_left: bool = False,
                   return_stats: bool = False, return_truncation_stats: bool = False):
    """Mirror of ``laya.common.build_sequence``.

    ``[CLS] <type> question: <ins> [SEP] ([MASK] opt)* [SEP] <state[:room]> [SEP]`` clamped to
    ``max_len``; markers past the clamp are dropped.
    """
    ids, markers, stats = build_head(tok, q, head_max_len, option_order=option_order)
    room = max(0, max_len - len(ids) - 1)
    state_text = serialize_state(state).replace(tok.mask_token, " ")
    state_ids = list(tok.encode(state_text, add_special_tokens=False)["input_ids"])
    st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    ids, markers = ids[:max_len], [m for m in markers if m < max_len]
    extra: Tuple[Any, ...] = ()
    if return_truncation_stats:
        extra = ({
            "state_tokens": len(state_ids),
            "state_tokens_used": len(st),
            "state_tokens_dropped": len(state_ids) - len(st),
            "truncated": len(st) < len(state_ids),
        },)
    if not return_stats:
        return (ids, markers) + extra
    return (ids, markers, stats) + extra


def serialize_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def state_room(tok, q: Dict[str, Any], max_len: int = 512, head_max_len: int = 192) -> int:
    head, _, _ = build_head(tok, q, head_max_len)
    return max(0, max_len - len(head) - 1)


# --------------------------------------------------------------------------------------
# confidence / calibration
# --------------------------------------------------------------------------------------
def confidence_from_probs(p, k: int) -> float:
    """Normalized Shannon entropy confidence — **not** calibrated. Mirrors the reference."""
    import numpy as np

    if k < 2:
        return 1.0
    p = np.asarray(p)[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def answer_confidence(p, k: int) -> float:
    """``max(p)`` — the quantity temperature scaling fits and the abstention gate compares."""
    import numpy as np

    if k < 1:
        return 1.0
    return float(np.clip(np.max(np.asarray(p)[:k]), 0.0, 1.0))


def ece_score(conf, correct, bins: int = 15) -> float:
    """Mirror of ``laya.common.ece_score``."""
    import numpy as np

    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def temp_bucket(qtype: int, k: int) -> str:
    """Mirror of ``laya.common.temp_bucket``: ``"choice:3-5"``, ``"noul:2"``, …"""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """Mirror of ``laya.common.clamp_temperature`` (bools and non-numbers fall back to 1.0)."""
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return float(min(hi, max(lo, t)))


def softmax_rows(logits, temperature: float = 1.0):
    """Stable softmax over the last axis divided by a scalar temperature."""
    import numpy as np

    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def unpermute_probs(p, option_order: Optional[Sequence[int]]):
    """Invert an option permutation: slot order → caller's option order. Reference-parity."""
    import numpy as np

    p = np.asarray(p)
    if option_order is None:
        return p
    out = np.empty_like(p)
    for slot, opt in enumerate(option_order):
        out[opt] = p[slot]
    return out


def proper_reward(q, target, qtype, mask, w_sph: float = 0.5, w_rps: float = 1.0,
                  log_floor: float = -9.21):
    """Mirror of ``laya.common.proper_reward`` (log score + spherical, RPS on `score` questions).

    Strictly proper: reporting the honest probability is the only way to maximise expected reward,
    which is what makes the published confidence numbers meaningful rather than decorative.
    """
    import torch

    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if bool(is_score.any()):
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


# --------------------------------------------------------------------------------------
# runtime ⇄ internal question conversion (the app sends the runtime shape)
# --------------------------------------------------------------------------------------
def to_internal(q: Dict[str, Any]) -> Dict[str, Any]:
    """Runtime question → internal question dict used by build_sequence / the trainer."""
    return {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}


def to_runtime(q: Dict[str, Any]) -> Dict[str, Any]:
    """Internal question dict → runtime question (what `/api/explain`, eval sets and the app use)."""
    out: Dict[str, Any] = {"type": q["t"], "instructions": q["ins"]}
    if q.get("crit") is not None:
        out["criteria"] = q["crit"]
    if q.get("labels"):
        out["labels"] = q["labels"]
    return out


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class TinyEncoderConfig:
    def __init__(self, hidden_size: int = 64, num_hidden_layers: int = 2, vocab_size: int = 512):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size

    def to_dict(self) -> Dict[str, Any]:
        return {"hidden_size": self.hidden_size, "num_hidden_layers": self.num_hidden_layers,
                "vocab_size": self.vocab_size, "model_type": "kasauti-tiny-encoder"}

    def save_pretrained(self, path: str) -> None:  # pragma: no cover - filesystem helper
        import os

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "TinyEncoderConfig":
        import os

        with open(os.path.join(path, "config.json")) as f:
            d = json.load(f)
        return cls(d.get("hidden_size", 64), d.get("num_hidden_layers", 2), d.get("vocab_size", 512))


class MultiheadSelfAttention:
    pass


def _attention_module(torch):
    """Traceable self-attention factory (see the docstring for why not the stock module)."""

    class MultiheadSelfAttention(torch.nn.Module):
        """Fixed QKV projection, explicit shapes, no traced length.

        The stock ``nn.MultiheadAttention`` bakes the traced sequence length into the exported
        graph and reaches for internals (``in_proj_bias``) that a custom module does not have, so
        the exported ONNX would refuse any length other than the 17 it was traced with — which is
        precisely what the browser sends. Not used on the Kaggle path, where the reference
        ``DecisionModel`` from ``laya.common`` is built instead; the smoke profile needs the same
        I/O contract and an exportable graph.
        """

        def __init__(self, d: int, nhead: int, dropout: float = 0.0, batch_first: bool = True):
            super().__init__()
            assert d % nhead == 0
            self.d, self.nhead, self.dh = d, nhead, d // nhead
            self.batch_first = batch_first
            self.head_dim = self.dh
            self.num_heads = nhead
            self.dropout_p = dropout
            self.qkv = torch.nn.Linear(d, 3 * d, bias=True)
            self.out = torch.nn.Linear(d, d, bias=True)
            self.drop = torch.nn.Dropout(dropout)

        def forward(self, query, key=None, value=None, attn_mask=None, key_padding_mask=None,
                    is_causal=None, need_weights=False, **kw):
            x = query
            B, L, D = x.shape
            qkv = self.qkv(x).view(B, L, 3, self.nhead, self.dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            if key_padding_mask is not None:
                kpm = key_padding_mask if key_padding_mask.dtype == torch.bool else key_padding_mask.bool()
                s = s.masked_fill(kpm[:, None, None, :], float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    s = s.masked_fill(attn_mask, float("-inf"))
                else:
                    s = s + attn_mask
            p = self.drop(torch.softmax(s, dim=-1))
            o = (p @ v).transpose(1, 2).reshape(B, L, D)
            return self.out(o), None

    return MultiheadSelfAttention


def _encoder_block_class(torch):
    Attn = _attention_module(torch)

    class EncoderBlock(torch.nn.Module):
        """Pre-norm transformer block (norm_first, batch_first), mirroring the reference head."""

        def __init__(self, d: int, nhead: int, ff: int, dropout: float = 0.1):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(d)
            self.norm2 = torch.nn.LayerNorm(d)
            self.self_attn = Attn(d, nhead, dropout=dropout, batch_first=True)
            self.linear1 = torch.nn.Linear(d, ff)
            self.linear2 = torch.nn.Linear(ff, d)
            self.dropout = torch.nn.Dropout(dropout)
            self.dropout1 = torch.nn.Dropout(dropout)
            self.dropout2 = torch.nn.Dropout(dropout)
            self.activation = torch.nn.GELU()

        def forward(self, x, src_key_padding_mask=None, **kw):
            x = x + self.dropout1(self.self_attn(self.norm1(x), key_padding_mask=src_key_padding_mask)[0])
            y = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
            return x + self.dropout2(y)

    return EncoderBlock


def _head_stack_class(torch):
    Block = _encoder_block_class(torch)

    class HeadStack(torch.nn.Module):
        """``nn.TransformerEncoder``-shaped container: ``.layers`` is iterated by DecisionModel."""

        def __init__(self, d: int, nhead: int, layers: int, dropout: float = 0.1):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(d, nhead, 4 * d, dropout) for _ in range(layers)])

    return HeadStack


def make_decision_model(encoder, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
    """The reference ``DecisionModel`` (``laya/common.py``), built against any encoder.

    Kept here so the CPU smoke profile can run the *entire* pipeline (train → calibrate → export →
    parity) without downloading a 647 MB checkpoint, on the identical forward/export contract:

        forward(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)
    """
    import torch
    import torch.nn as nn

    HeadStack = _head_stack_class(torch)

    class DecisionModel(nn.Module):
        def __init__(self, encoder, head_layers_: int = 2, n_act_: int = 2, dropout_: float = 0.1):
            super().__init__()
            self.encoder = encoder
            d = encoder.config.hidden_size
            nhead = max(1, d // 64)
            self.head = HeadStack(d, nhead, head_layers_, dropout_) if head_layers_ > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act_))
            self.register_buffer("temperature", torch.ones(3))
            self.head_checkpointing = False

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if detach_encoder:
                h = h.detach()
            h = h + self.type_emb(qtype)[:, None, :]
            if self.head is not None:
                pad = ~attention_mask.bool()
                for layer in self.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx)
            logits = self.scorer(m).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)
            p = torch.softmax(logits.detach(), -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            if p.size(-1) >= 2:
                top2 = p.topk(2, -1).values
            else:
                top1 = p.topk(1, -1).values
                top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
            feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
            pooled = h[:, 0].float()
            act_input = torch.cat([pooled, feats], -1)
            if not torch.is_autocast_enabled(h.device.type):
                act_input = act_input.to(self.act_head[0].weight.dtype)
            act_logits = self.act_head(act_input)
            return logits, act_logits

    return DecisionModel(encoder, head_layers, n_act, dropout)


def build_tiny_model(vocab_size: int, hidden: int = 64, layers: int = 2, head_layers: int = 1,
                     max_len: int = 128, seed: int = 1234):
    """A randomly-initialised stand-in with the real model's I/O contract (smoke profile only)."""
    import torch
    import torch.nn as nn

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TinyEncoderConfig(hidden, layers, vocab_size)
            self.emb = nn.Embedding(vocab_size, hidden)
            # Learned position table indexed by arange(L): verified to survive tracing as a dynamic
            # Range/Slice (checked at L = 17/64/128 against the exported graph). Without any
            # positional signal the tiny model collapses to a token-identity function — every
            # [MASK] marker then scores identically, which makes the smoke profile's calibration and
            # accuracy numbers meaningless. The shipped checkpoints use RoPE for the same reason.
            self.pos = nn.Embedding(max_len, hidden)
            self.max_len = max_len
            Block = _encoder_block_class(torch)
            self.layers = nn.ModuleList([
                Block(hidden, max(1, hidden // 64), 4 * hidden, 0.0) for _ in range(layers)
            ])
            self.norm = nn.LayerNorm(hidden)

        def forward(self, input_ids, attention_mask=None):
            B, L = input_ids.shape
            pos = torch.arange(L, device=input_ids.device)[None, :].expand(B, -1)
            h = self.emb(input_ids) + self.pos(pos)
            pad = None if attention_mask is None else ~attention_mask.bool()
            for layer in self.layers:
                h = layer(h, src_key_padding_mask=pad)
            return type("Out", (), {"last_hidden_state": self.norm(h)})()

    torch.manual_seed(seed)
    return make_decision_model(TinyEncoder(), head_layers=head_layers)


def build_model_from_cfg(cfg: Dict[str, Any], encoder_dir: Optional[str] = None, tiny: bool = False,
                         vocab_size: int = 512):
    """Real `laya.common.build_model` when available, tiny stand-in otherwise."""
    if tiny:
        return build_tiny_model(vocab_size, hidden=cfg.get("hidden", 64), layers=cfg.get("layers", 2),
                                head_layers=cfg.get("head_layers", 1), max_len=cfg.get("max_len", 128))
    from laya.common import build_model  # noqa: WPS433 (deliberately lazy: training-only dependency)

    return build_model(cfg, encoder_dir=encoder_dir)


# --------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------
def collate(items: Sequence[Dict[str, Any]], pad_id: int):
    """Mirror of the reference notebook's ``collate_train_batch``."""
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        # eval items have no label (they are scored against the soft target instead)
        "label": torch.tensor([it.get("label", -1) for it in items]),
    }


def iter_batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
