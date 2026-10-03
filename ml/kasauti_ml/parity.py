"""Parity vectors: the browser's self-test.

Two independent checks ship as one small JSON file next to the model:

* **tokenizer parity** — texts from every supported language with the exact id sequence the Rust
  tokenizer produced. The browser runs its pure-TS tokenizer over the same texts and must match
  *exactly*; a mismatch means the app and the model disagree about what a token is, which would
  silently degrade every prediction. This is the test that caught the Metaspace
  ``prepend_scheme`` divergence between transformers.js and the Rust tokenizer.
* **logits parity** — a handful of full sequences with their fp32 logits. The browser runs the
  downloaded graph in the Worker and compares within tolerance: catches a corrupted cache entry,
  a wrong build, a WebGPU EP that silently mis-compiles, or a tokenizer that drifted.

Both vectors are generated from the **pruned** tokenizer and the **published** graph, so they
validate exactly what users download.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Texts chosen to exercise the awkward parts: Devanagari, Tamil, mixed script, digits,
#: romanised code-mixing, URLs, rupee amounts, zero-width injection, and the [MASK] token.
PARITY_TEXTS: Sequence[str] = (
    "Guaranteed 3% daily profit, join our VIP group today only.",
    "बधाई! हमारे VIP ग्रुप से जुड़ें, {pct} रोज़ का मुनाफ़ा पक्का।",
    "मराठीतून हमी परतावा 20% महिना, लगेच गुंतवा.",
    "உறுதியான லாபம் இன்றே சேருங்கள், ₹5,000 செலுத்துங்கள்.",
    "apna paisa double karo bhai, jaldi join karo",
    "Pay ₹4.2 lakh to vipfund@ybl or call 1600123456.",
    "http://kyc-update.xyz/login?utm_source=wa   extra   spaces",
    "zero\u200bwidth and Cyrillic homoglyph: guaranteed profit",
    "literal <mask> token must be scrubbed, not parsed as a marker",
    "1234567890 ०१२३४५६७८९ ₹1,20,000 lakh crore",
)


def tokenizer_vectors(tok_json_path: str, texts: Sequence[str] = PARITY_TEXTS,
                      max_len: Optional[int] = None) -> Dict[str, Any]:
    """Encode ``texts`` with the shipped tokenizer (Rust) — the browser must reproduce this."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_json_path)
    vectors = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False).ids
        if max_len:
            ids = ids[:max_len]
        vectors.append({"text": t, "ids": ids})
    specials = {"mask": tok.token_to_id("<mask>"), "cls": tok.token_to_id("<bos>"),
                "sep": tok.token_to_id("<eos>"), "pad": tok.token_to_id("<pad>"),
                "unk": tok.token_to_id("<unk>")}
    specials = {k: (v if v is not None else -1) for k, v in specials.items()}
    return {"kind": "tokenizer", "version": 1, "vocabSize": tok.get_vocab_size(),
            "specials": specials, "vectors": vectors}


def logits_vectors(sess, items: Sequence[Dict[str, Any]], tolerance: float = 0.05,
                   max_items: int = 6) -> Dict[str, Any]:
    """Full-sequence vectors: ids + marker positions + qtype + the logits the graph produced."""
    from .export import ort_predict

    picked = list(items[:max_items])
    preds = ort_predict(sess, picked)
    vectors = []
    for it, (qt, logits, _target) in zip(picked, preds):
        vectors.append({
            "ids": list(it["ids"]),
            "markers": list(it["markers"]),
            "qtype": int(qt),
            "logits": [round(float(v), 6) for v in logits],
        })
    return {"kind": "logits", "version": 1, "tolerance": tolerance, "vectors": vectors}


def write_parity(path: str, tokenizer_section: Dict[str, Any], logits_section: Dict[str, Any]) -> str:
    blob = {"schema": 1, "tokenizer": tokenizer_section, "logits": logits_section}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False)
    return path
