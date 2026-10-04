#!/usr/bin/env python3
"""Build a self-contained tokenizer parity fixture from a real Laya checkpoint tokenizer.

The browser tokenizer (`src/core/tokenizer.ts`) is a from-scratch port of tokenizer.json
semantics: Metaspace pre-tokenization, rank-ordered BPE, `byte_fallback`, normalizer `Replace`
rules and the two-phase `added_tokens` scan. Ports like this fail silently — a wrong id only
shows up as a slightly worse answer — so the port is pinned to the *reference* implementation
(the Rust `tokenizers` library the model was trained with) by replaying exact id sequences.

A full multilingual tokenizer.json is ~34 MB, which has no business in a git repo. So this
script keeps only what the fixture texts actually need, preserving the ORIGINAL token ids and
merge ranks (so ids stay comparable), and writes:

    tests/fixtures/laya-tokenizer.mini.json    vocab + merges + normalizer/pre-tokenizer/added_tokens
    tests/fixtures/laya-tokenizer.vectors.json  texts + reference ids + provenance (repo, sha256)

`tests/tokenizer-parity.test.ts` then runs offline with no network and no 34 MB download.

Usage
-----
    python -m ml.scripts.make_tokenizer_fixture --tokenizer /path/to/tokenizer.json \
        --out-dir tests/fixtures
    python -m ml.scripts.make_tokenizer_fixture --repo convaiinnovations/laya \
        --subfolder multilingual/tokenizer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Set, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Texts chosen to exercise every branch of the port, not to look representative:
# metaspace words, newline runs, non-Latin scripts, unspaced CJK/Thai (the heap path),
# emoji + surrogate pairs, digits/currency, URLs (the byte-level path when used),
# empty string, a lone space, the literal added-token strings, and a long unbroken run.
PARITY_TEXTS: List[str] = [
    "",
    " ",
    "hi",
    "hello world",
    "Guaranteed 3% daily profit, zero risk. Limited seats, join today only. Pay ₹5,000 to vipfund@ybl",
    "This is CBI officer. Your Aadhaar is linked to a money laundering case. Stay on video call, do not tell anyone in your family, transfer ₹2 lakh for verification.",
    "SEBI warns investors: beware of unregistered tips groups promising guaranteed return. Report fraud to 1930.",
    "Your profit of ₹4.8 lakh is ready. Pay 18% withdrawal tax to unlock your profit. Install AnyDesk so our team can help.",
    "Research by INH000009001: quarterly results summary of the auto sector, no recommendations.",
    "आपका KYC expired है, खाता बंद होगा। तुरंत इस लिंक पर क्लिक करें: http://kyc-update.xyz",
    "म्यूचुअल फंड में निवेश से पहले पूरा डॉक्यूमेंट ज़रूर पढ़ें, कमाई की कोई गारंटी नहीं होती।",
    "இந்த குழுவில் உறுப்பினர் சேருங்கள், உறுதியான லாபம் கிடைக்கும்.",
    "টাকা ডাবল হবে, এখনই জয়েন করুন ভিআইপি গ্রুপে।",
    "KYC अपडेट नहीं किया तो अकाउंट ब्लॉक हो जाएगा, OTP और CVV शेयर करें।",
    "https://kyc-update-bank.example.com/verify?id=123456&otp=1",
    "upi: vipfund@ybl payee: 9876543210",
    "Registrar: INH000009001 · phone 1600-123-4567 · tollfree 1800 123 4567",
    "line one\nline two\n\nline three\n\n\nline four",
    "   leading and trailing spaces   ",
    "tab\tseparated\tvalues",
    "Guaranteed 100% safe investment scheme",
    "GUARANTEED RETURNS!!! ACT NOW!!!",
    "🚨🚨 URGENT 🚨🚨 join now 💰💰",
    "Zero-width test: Gua\u200branteed",
    "Homoglyph test: рrofit gοld",
    "नमस्ते दुनिया 12345 ६७८९",
    "ತುಂಬಾ ಒಳ್ಳೆಯ ಲಾಭ ಸಿಗುತ್ತದೆ ಎಂದು ಹೇಳಿದರು",
    "మీ పెట్టుబడికి రెట్టింపు లాభం ఇస్తాము",
    "गारंटीड रिटर्न के साथ निवेश करें और पैसा दोगुना करें, कोई जोखिम नहीं।",
    "投资保证收益，零风险，立即加入我们的VIP群组。",
    "このグループに参加すれば必ず利益が出ます。今すぐお金を振り込んでください。",
    "این گروه تضمین سود می‌دهد",
    "مطلوب استثمار آمن بدون مخاطر",
    "గ్యారంటీ రిటర్న్స్ తో డబ్బు రెట్టింపు చేసుకోండి",
    "تحويل الأموال إلى حسابنا لاستلام الأرباح",
    "นี่คือเจ้าหน้าที่ตำรวจ กรุณาโอนเงินเพื่อตรวจสอบ",
    "이것은 경찰입니다. 돈을 이체하세요",
    "मैं एक निवेश सलाहकार हूं, मुझ पर भरोसा करें",
    "x" * 400,
    "अ" * 200,
    "投資" * 120,
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9",
    "The quick brown fox jumps over the lazy dog. " * 6,
]


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def load_tokenizer(path: str, repo: str | None, subfolder: str | None, token: str | None):
    from tokenizers import Tokenizer

    if path:
        return Tokenizer.from_file(path)
    if not repo:
        raise SystemExit("[fixture] pass --tokenizer or --repo")
    from huggingface_hub import hf_hub_download

    fname = f"{subfolder}/tokenizer.json" if subfolder else "tokenizer.json"
    local = hf_hub_download(repo, fname, token=token)
    return Tokenizer.from_file(local)


def closure(model: Dict[str, Any], keep: Set[int]) -> Set[int]:
    """Add every piece a kept token's merge chain is built from (transitive, to fixpoint).

    Walks **all** producers of a token, not the first one found: a merge list is a multimap, and
    keeping only one producer's parts makes the fixture unable to rebuild tokens the real tokenizer
    produced that way. `ml/kasauti_ml/export.py::prune_tokenizer` carries the same rule — on the
    multilingual checkpoint the single-producer version drifted on 16.5% of a 10k-text corpus.
    """
    vocab_map: Dict[str, int] = model["vocab"]
    vocab: List[str] = [""] * (max(vocab_map.values()) + 1)
    for t, i in vocab_map.items():
        vocab[i] = t
    # result-string -> every (left, right) pair producing it
    producers: Dict[str, List[Tuple[str, str]]] = {}
    for m in model["merges"]:
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m[0], m[1]
        producers.setdefault(a + b, []).append((a, b))
    by_token = {t: i for i, t in enumerate(vocab)}
    stack = [i for i in keep if i < len(vocab)]
    seen = set(stack)
    while stack:
        i = stack.pop()
        for a, b in producers.get(vocab[i], ()):
            for part in (a, b):
                j = by_token.get(part)
                if j is not None and j not in seen:
                    seen.add(j)
                    keep.add(j)
                    stack.append(j)
    return keep


def build_mini(tj: Dict[str, Any], texts: List[str], ids_seen: Set[int]) -> Dict[str, Any]:
    model = tj["model"]
    vocab_map: Dict[str, int] = model["vocab"]
    vocab: List[str] = [""] * (max(vocab_map.values()) + 1)
    for t, i in vocab_map.items():
        vocab[i] = t
    keep: Set[int] = set(ids_seen)
    # every special/single id declared in added_tokens
    for t in tj.get("added_tokens", []):
        if isinstance(t.get("id"), int):
            keep.add(int(t["id"]))
    chars: Set[str] = set()
    for text in texts:
        chars.update(text)
    for ch in chars:
        for cand in (ch, "▁" + ch):
            i = vocab_map.get(cand)
            if i is not None:
                keep.add(i)
    # byte-fallback tokens: a port that mishandles byte_fallback must fail loudly, so keep them all
    for i, t in enumerate(vocab):
        if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
            keep.add(i)
    keep = closure(model, keep)

    mini_vocab: Dict[str, int] = {vocab[i]: i for i in sorted(keep) if i < len(vocab)}
    # A merge may only be kept when its *result* is in the vocabulary, not just its two parts.
    # Keeping a merge whose result is absent produces a tokenizer.json the Rust parser refuses to
    # load at all ("Token `ge` out of vocabulary") — and one that the pure-TS port would skip
    # silently, i.e. a fixture that cannot act as the reference anchor it exists to be. This is the
    # same rule `ml/kasauti_ml/export.py::prune_tokenizer` applies to the shipped tokenizer.
    merges_out: List[str] = []
    for m in model["merges"]:
        pair = m if isinstance(m, str) else f"{m[0]} {m[1]}"
        a, b = pair.split(" ", 1)
        if a in mini_vocab and b in mini_vocab and (a + b) in mini_vocab:
            merges_out.append(pair)
    out: Dict[str, Any] = {
        "version": tj.get("version"),
        "truncation": tj.get("truncation"),
        "padding": tj.get("padding"),
        "added_tokens": [t for t in tj.get("added_tokens", []) if t.get("content") in mini_vocab],
        "normalizer": tj.get("normalizer"),
        "pre_tokenizer": tj.get("pre_tokenizer"),
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": model.get("type"),
            "dropout": model.get("dropout"),
            "unk_token": model.get("unk_token"),
            "continuing_subword_prefix": model.get("continuing_subword_prefix"),
            "end_of_word_suffix": model.get("end_of_word_suffix"),
            "fuse_unk": model.get("fuse_unk"),
            "byte_fallback": model.get("byte_fallback"),
            "vocab": mini_vocab,
            "merges": merges_out,
        },
    }
    # NOTE: no extra top-level keys. `tokenizers`' Rust parser rejects unknown fields in
    # `tokenizer.json` with a misleading "expected `,` or `}` at line 1 column N" — the metadata
    # lives in the vectors file beside it instead, which nothing but this project reads.
    return out


def self_check(mini: Dict[str, Any], vectors: List[Dict[str, Any]], tmp_dir: str) -> int:
    """Segment every parity text with the mini tokenizer and compare with the reference ids.

    This is the check that was missing: a fixture whose merges cannot resolve loads (or not) and
    still looks plausible, while the *ids* it produces for the texts the TS port is pinned against
    have quietly changed. Ids are preserved by construction, so the comparison is exact.
    """
    from tokenizers import Tokenizer

    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, "mini.selfcheck.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mini, f, ensure_ascii=False)
    tok = Tokenizer.from_file(path)          # raises if the fixture is not loadable
    drift = []
    for vec in vectors:
        got = list(tok.encode(vec["text"], add_special_tokens=False).ids)
        if got != list(vec["ids"]):
            drift.append((vec["text"][:40], vec["ids"][:8], got[:8]))
    os.remove(path)
    if drift:
        print(f"[fixture] FAIL: {len(drift)}/{len(vectors)} texts segment differently")
        for text, want, got in drift[:5]:
            print(f"           {text!r}\n             want {want}\n             got  {got}")
    return len(drift)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the tokenizer parity fixture")
    ap.add_argument("--tokenizer", default="", help="path to a local tokenizer.json")
    ap.add_argument("--repo", default="", help="HF repo id to download tokenizer.json from")
    ap.add_argument("--subfolder", default="tokenizer")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "tests", "fixtures"))
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the checked-in fixture is stale (CI mode; needs the real "
                         "tokenizer, so it is a maintainer check, not a per-commit one)")
    args = ap.parse_args(argv)

    tok = load_tokenizer(args.tokenizer, args.repo or None, args.subfolder or None, args.hf_token or None)
    texts = PARITY_TEXTS
    vectors: List[Dict[str, Any]] = []
    ids_seen: Set[int] = set()
    for text in texts:
        enc = tok.encode(text, add_special_tokens=False)
        vectors.append({"text": text, "ids": list(enc.ids), "tokens": list(enc.tokens)})
        ids_seen.update(enc.ids)

    raw = tok.to_str()
    tj = json.loads(raw)
    mini = build_mini(tj, texts, ids_seen)

    drift = self_check(mini, vectors, tempfile.mkdtemp(prefix="kasauti-fixture-"))
    if drift:
        raise SystemExit("[fixture] refusing to write a fixture that does not reproduce the "
                         "reference ids — widen the kept vocabulary (see build_mini) and retry")

    os.makedirs(args.out_dir, exist_ok=True)
    mini_path = os.path.join(args.out_dir, "laya-tokenizer.mini.json")
    vec_path = os.path.join(args.out_dir, "laya-tokenizer.vectors.json")
    mini_blob = json.dumps(mini, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(mini_path, "wb") as f:
        f.write(mini_blob)

    payload = {
        "note": "Reference ids produced by the Rust `tokenizers` library from the real Laya "
                "checkpoint — the source of truth for src/core/tokenizer.ts. Generated by "
                "ml/scripts/make_tokenizer_fixture.py; do not edit.",
        "source": {
            "repo": args.repo or "(local file)",
            "subfolder": args.subfolder or "",
            "tokenizer_sha256": sha256_bytes(raw.encode("utf-8")),
            "full_vocab": len(tj["model"]["vocab"]),
            "full_merges": len(tj["model"]["merges"]),
            "pretokenizer": tj.get("pre_tokenizer", {}).get("type") if isinstance(tj.get("pre_tokenizer"), dict) else None,
        },
        "mini_tokenizer": os.path.basename(mini_path),
        "mini_tokenizer_sha256": sha256_bytes(mini_blob),
        "mini_counts": {"vocab": len(mini["model"]["vocab"]), "merges": len(mini["model"]["merges"])},
        # every text below was re-segmented with the mini tokenizer itself and matched exactly;
        # without that, the fixture could "pass" while no longer describing the same tokenizer
        "mini_reproduces_reference_ids": True,
        "vectors": vectors,
    }
    with open(vec_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    if args.check:
        current = open(mini_path, "rb").read() if os.path.exists(mini_path) else b""
        if current != mini_blob:
            print("[fixture] STALE: tests/fixtures/laya-tokenizer.mini.json differs", file=sys.stderr)
            return 1
        print("[fixture] up to date")
        return 0

    n_ids = sum(len(v["ids"]) for v in vectors)
    print(f"[fixture] {len(vectors)} texts / {n_ids} ids from {len(tj['model']['vocab']):,} vocab "
          f"→ mini {len(mini['model']['vocab']):,} vocab, {len(mini['model']['merges']):,} merges")
    print(f"[fixture] wrote {os.path.relpath(mini_path, REPO)} ({len(mini_blob) / 1024:.0f} KB) "
          f"and {os.path.relpath(vec_path, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
