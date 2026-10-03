"""Tokenizer adapter + the tiny synthetic tokenizer used by the CPU smoke profile.

The adapter gives the HF fast tokenizer exactly the surface ``core.build_sequence`` needs
(``__call__(text, add_special_tokens=False)["input_ids"]``, ``mask_token``, ``*_token_id``), so the
same code path runs against ``convaiinnovations/laya``'s real tokenizer on Kaggle and against a
synthetic one on a CPU box.

``build_tiny_tokenizer`` emits a **real** HuggingFace `tokenizer.json` (Metaspace pre-tokenizer,
BPE with merge ranks, byte fallback, added special tokens). That matters: the browser's pure-TS
tokenizer is a reimplementation of the Rust one, so a synthetic-but-real tokenizer.json lets the
smoke profile prove byte-for-byte parity of the whole client stack before any 647 MB checkpoint
is downloaded.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

SPECIALS = [("<pad>", 0), ("<unk>", 1), ("<bos>", 2), ("<eos>", 3), ("<mask>", 4)]


class FastTokenizerAdapter:
    """Wrap a HuggingFace fast tokenizer (or ``tokenizers.Tokenizer``) for the pipeline."""

    def __init__(self, tok: Any):
        self.tok = tok
        self._ids = {
            "cls": self._id("cls_token", ("<s>", "<bos>", "[CLS]"), 2),
            "sep": self._id("sep_token", ("</s>", "<eos>", "[SEP]"), 1),
            "mask": self._id("mask_token", ("<mask>", "[MASK]"), 4),
            "pad": self._id("pad_token", ("<pad>", "[PAD]"), 0),
            "unk": self._id("unk_token", ("<unk>", "[UNK]"), 1),
        }
        self.mask_token = getattr(tok, "mask_token", None) or "<mask>"

    def _id(self, attr: str, aliases: Sequence[str], fallback: int) -> int:
        v = getattr(self.tok, f"{attr}_id", None)
        if isinstance(v, int) and v >= 0:
            return v
        for a in aliases:
            try:
                got = self.tok.convert_tokens_to_ids(a)
            except Exception:
                got = None
            if isinstance(got, int) and got >= 0:
                return got
        return fallback

    @property
    def cls_token_id(self) -> int:
        return self._ids["cls"]

    @property
    def sep_token_id(self) -> int:
        return self._ids["sep"]

    @property
    def mask_token_id(self) -> int:
        return self._ids["mask"]

    @property
    def pad_token_id(self) -> int:
        return self._ids["pad"]

    @property
    def vocab_size(self) -> int:
        try:
            return int(self.tok.get_vocab_size())
        except Exception:
            try:
                return len(self.tok.get_vocab())
            except Exception:
                return 0

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False,
                 max_length: Optional[int] = None, **kw) -> Dict[str, List[int]]:
        try:
            enc = self.tok(text, add_special_tokens=add_special_tokens, truncation=truncation,
                           max_length=max_length, **kw)
            return {"input_ids": list(enc["input_ids"])}
        except TypeError:
            pieces = self.tok.encode(text, add_special_tokens=add_special_tokens)
            ids = list(pieces.ids)
            if truncation and max_length:
                ids = ids[:max_length]
            return {"input_ids": ids}

    def encode(self, text: str, add_special_tokens: bool = False, truncation: bool = False,
               max_length: Optional[int] = None) -> Dict[str, List[int]]:
        return self(text, add_special_tokens=add_special_tokens, truncation=truncation,
                    max_length=max_length)

    def save_pretrained(self, path: str) -> None:
        if hasattr(self.tok, "save_pretrained"):
            self.tok.save_pretrained(path)
        else:  # raw tokenizers.Tokenizer
            os.makedirs(path, exist_ok=True)
            self.tok.save(os.path.join(path, "tokenizer.json"))


def load_fast_tokenizer(model_dir: str) -> FastTokenizerAdapter:
    """Load the checkpoint tokenizer exactly like the reference notebook does."""
    from transformers import AutoTokenizer

    for candidate in (os.path.join(model_dir, "tokenizer"), model_dir):
        try:
            return FastTokenizerAdapter(AutoTokenizer.from_pretrained(candidate))
        except Exception:
            continue
    raise RuntimeError(f"no tokenizer found under {model_dir}")


# --------------------------------------------------------------------------------------
# tiny synthetic tokenizer (smoke profile)
# --------------------------------------------------------------------------------------
_BYTE_TOKENS = ["<0x%02x>" % i for i in range(256)]


def _words(texts: Iterable[str]) -> List[str]:
    out: Dict[str, int] = {}
    for t in texts:
        for w in re.findall(r"\S+", t):
            out[w] = out.get(w, 0) + 1
    return sorted(out, key=lambda w: (-out[w], w))


def build_tiny_tokenizer(texts: Iterable[str], out_path: str, max_words: int = 1200,
                         merges_per_word: bool = True) -> Dict[str, Any]:
    """Write a small but *real* tokenizer.json covering the smoke corpus.

    Build order is fixed by construction: specials (0-4), byte tokens, characters, then words.
    Merges rank every adjacent character pair of each word by frequency, so BPE reconstructs words;
    characters absent from the vocab fall back to byte tokens (``byte_fallback: true``).
    """
    words = _words(texts)[:max_words]
    vocab: Dict[str, int] = {}
    for tok, _ in SPECIALS:
        vocab[tok] = len(vocab)
    for b in _BYTE_TOKENS:
        vocab[b] = len(vocab)
    chars: Dict[str, int] = {"▁": 0}
    for w in words:
        for ch in w:
            chars.setdefault(ch, 0)
            chars.setdefault("▁" + ch, 0)
    for ch in sorted(chars):
        vocab.setdefault(ch, len(vocab))
    for w in words:
        vocab.setdefault("▁" + w, len(vocab))

    # Metaspace prefixes every space-delimited piece with "▁", so BPE merge chains are built over
    # "▁word". Every intermediate piece must exist in the vocab: the Rust parser rejects a merge
    # whose result is missing ("Token `th` out of vocabulary").
    merges: List[str] = []
    seen = set()
    if merges_per_word:
        for w in words:
            pieces = list("▁" + w)
            while len(pieces) > 1:
                pair = (pieces[0], pieces[1])
                merged = pair[0] + pair[1]
                vocab.setdefault(merged, len(vocab))
                key = " ".join(pair)
                if key not in seen:
                    seen.add(key)
                    merges.append(key)
                pieces = [merged] + pieces[2:]

    data = {
        "version": "1.0",
        "added_tokens": [
            {"id": i, "content": tok, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": True}
            for tok, i in SPECIALS
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always",
                          "split": True},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return {"vocab_size": len(vocab), "merges": len(merges), "words": len(words)}


def load_tiny_tokenizer(path: str) -> FastTokenizerAdapter:
    """Parse the synthetic tokenizer.json with the real Rust tokenizer (parity anchor)."""
    from tokenizers import Tokenizer

    return FastTokenizerAdapter(Tokenizer.from_file(path))
