#!/usr/bin/env python3
"""Generate ``src/lib/banks.ts`` from ``ml/banks.json``.

``ml/banks.json`` is the single source of truth for every Laya question bank Kasauti uses.
The Python trainer reads it directly; the browser runtime reads the TypeScript mirror this
script writes. Generating (instead of hand-maintaining) the runtime copy is the only way the
two can never drift apart — which matters a lot here, because a question whose wording differs
between training and inference silently degrades accuracy without any error.

Usage
-----
    python -m ml.scripts.gen_banks_ts             # write src/lib/banks.ts
    python -m ml.scripts.gen_banks_ts --check     # exit 1 if the file is stale (CI / pre-commit)
    python -m ml.scripts.gen_banks_ts --out -     # print to stdout

Validation performed before writing (hard failures, so a bad bank never reaches a model):
  * question types are limited to ``choice`` / ``score`` / ``noul``
  * ``choice`` has 2..20 options (Laya's own limit: accuracy collapses above ~20)
  * ``score`` has 2..10 levels
  * ``choice`` criteria are flat ``key -> description`` maps, never nested
  * every ``followups`` key is one of the triage option keys
  * instruction strings are single-line, <= 300 characters, and contain no ``<mask>``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BANKS_JSON = os.path.join(REPO, "ml", "banks.json")
OUT_TS = os.path.join(REPO, "src", "lib", "banks.ts")
MAX_CHOICE_OPTIONS = 20
MAX_SCORE_LEVELS = 10


def canonical_sha(banks: Dict[str, Any]) -> str:
    """sha256 of the canonical JSON — the same digest the pipeline records in the manifest."""
    blob = json.dumps(banks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fail(msg: str) -> None:
    raise SystemExit(f"[gen_banks_ts] {msg}")


def validate(banks: Dict[str, Any]) -> None:
    version = banks.get("version")
    if not isinstance(version, str) or not version:
        fail("banks.json is missing a string `version`")
    tasks = banks.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        fail("banks.json is missing a non-empty `tasks` map")
    for bank_id, bank in tasks.items():
        title = bank.get("title")
        purpose = bank.get("purpose", "")
        if not isinstance(title, str) or not title:
            fail(f"bank {bank_id}: missing title")
        if not isinstance(purpose, str):
            fail(f"bank {bank_id}: purpose must be a string")
        triage = bank.get("triage")
        if not isinstance(triage, dict):
            fail(f"bank {bank_id}: missing triage question")
        keys = _check_question(f"{bank_id}.triage", triage)
        followups = bank.get("followups", {})
        if not isinstance(followups, dict):
            fail(f"bank {bank_id}: followups must be a map of triage key -> question set")
        for triage_key, qset in followups.items():
            if triage_key not in keys:
                fail(f"bank {bank_id}.followups[{triage_key}]: not a triage option key ({sorted(keys)})")
            if not isinstance(qset, dict):
                fail(f"bank {bank_id}.followups[{triage_key}]: must be a map of question id -> question")
            for qid, q in qset.items():
                if not qid.replace("_", "").isalnum():
                    fail(f"bank {bank_id}.followups[{triage_key}].{qid}: id must be [a-z0-9_]")
                _check_question(f"{bank_id}.followups[{triage_key}].{qid}", q)


def _check_question(path: str, q: Dict[str, Any]) -> List[str]:
    qtype = q.get("type")
    if qtype not in ("choice", "score", "noul"):
        fail(f"{path}: type must be one of choice|score|noul (got {qtype!r})")
    ins = q.get("instructions")
    if not isinstance(ins, str) or not ins.strip():
        fail(f"{path}: missing instructions")
    if "\n" in ins or len(ins) > 300:
        fail(f"{path}: instructions must be single-line and <= 300 chars")
    if "<mask>" in ins:
        fail(f"{path}: instructions must not contain the literal mask token")
    crit = q.get("criteria")
    if qtype == "choice":
        if not isinstance(crit, dict) or not crit:
            fail(f"{path}: choice needs a criteria map of option key -> description")
        n = len(crit)
        if not (2 <= n <= MAX_CHOICE_OPTIONS):
            fail(f"{path}: choice has {n} options; must be 2..{MAX_CHOICE_OPTIONS}")
        for k, v in crit.items():
            if not isinstance(k, str) or not k:
                fail(f"{path}: empty option key")
            if not isinstance(v, str):
                fail(f"{path}: option {k!r} description must be a string")
        return list(crit)
    if qtype == "score":
        if not isinstance(crit, list) or not crit:
            fail(f"{path}: score needs a criteria list of level descriptions")
        if not (2 <= len(crit) <= MAX_SCORE_LEVELS):
            fail(f"{path}: score has {len(crit)} levels; must be 2..{MAX_SCORE_LEVELS}")
        return [str(i) for i in range(len(crit))]
    # noul: fixed two-option shape; criteria (when present) only relabels false/true
    if crit is not None:
        if not isinstance(crit, dict) or set(crit) - {"false", "true"}:
            fail(f"{path}: noul criteria may only contain the keys 'false' and 'true'")
    return ["false", "true"]


def ts_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def ts_question(q: Dict[str, Any], indent: str) -> str:
    parts = [f"type: {ts_str(q['type'])}", f"instructions: {ts_str(q['instructions'])}"]
    crit = q.get("criteria")
    if isinstance(crit, dict):
        body = ", ".join(f"{ts_str(k)}: {ts_str(v)}" for k, v in crit.items())
        parts.append(f"criteria: {{ {body} }}")
    elif isinstance(crit, list):
        parts.append("criteria: [" + ", ".join(ts_str(v) for v in crit) + "]")
    return "{\n" + indent + ("  " + (",\n" + indent + "  ").join(parts)) + ",\n" + indent + "}"


def render(banks: Dict[str, Any], sha: str) -> str:
    tasks = banks["tasks"]
    out: List[str] = []
    out.append("/**")
    out.append(" * banks.ts — GENERATED by `python -m ml.scripts.gen_banks_ts`. Do not edit by hand.")
    out.append(" *")
    out.append(" * Source of truth: `ml/banks.json`. The trainer reads that file; the browser runtime")
    out.append(" * reads this mirror, so training and inference can never disagree about question wording")
    out.append(" * (a mismatch here costs accuracy silently, with no error anywhere).")
    out.append(" *")
    out.append(f" * banks version : {banks['version']}")
    out.append(f" * banks sha256  : {sha}")
    out.append(" *")
    out.append(" * The model answers only flat `choice` / `score` / `noul` questions, one forward pass per")
    out.append(" * batch of questions, and never generates text. `followups` is the adaptive stage-B tree:")
    out.append(" * the triage answer picks which follow-ups are worth asking, within the device's budget.")
    out.append(" */")
    out.append("")
    out.append('export type QType = "choice" | "score" | "noul";')
    out.append("")
    out.append("export interface Question {")
    out.append("  type: QType;")
    out.append("  instructions: string;")
    out.append("  /** choice: option key -> description · score: ordered level descriptions · noul: {false,true} relabels */")
    out.append("  criteria?: Record<string, string> | string[];")
    out.append("}")
    out.append("")
    out.append("export type QuestionSet = Record<string, Question>;")
    out.append("")
    out.append("export interface Bank {")
    out.append("  /** the key this bank is stored under in BANKS (typed as BankId after the map is defined) */")
    out.append("  id: string;")
    out.append("  title: string;")
    out.append("  purpose: string;")
    out.append("  triage: Question;")
    out.append("  followups: Record<string, QuestionSet>;")
    out.append("}")
    out.append("")
    out.append("export const BANKS_VERSION = " + ts_str(banks["version"]) + ";")
    out.append("/** sha256 of the canonical banks.json this file was generated from. */")
    out.append("export const BANKS_SHA256 = " + ts_str(sha) + ";")
    out.append("")
    out.append("export const BANKS = {")
    for bank_id, bank in tasks.items():
        out.append(f"  {bank_id}: {{")
        out.append(f"    id: {ts_str(bank_id)},")
        out.append(f"    title: {ts_str(bank['title'])},")
        out.append(f"    purpose: {ts_str(bank.get('purpose', ''))},")
        out.append("    triage: " + ts_question(bank["triage"], "    ") + ",")
        out.append("    followups: {")
        for triage_key, qset in (bank.get("followups") or {}).items():
            out.append(f"      {ts_str(triage_key)}: {{")
            for qid, q in qset.items():
                out.append(f"        {qid}: " + ts_question(q, "        ") + ",")
            out.append("      },")
        out.append("    },")
        out.append("  },")
    out.append("} as const satisfies Record<string, Bank>;")
    out.append("")
    out.append("export type BankId = keyof typeof BANKS;")
    out.append("")
    out.append("export const BANK_IDS = Object.keys(BANKS) as BankId[];")
    out.append("")
    out.append("/** Triage option keys for a bank — `followups` is keyed by these. */")
    out.append("export function triageOptions(bankId: BankId): string[] {")
    out.append("  const c = BANKS[bankId].triage.criteria;")
    out.append("  if (!c) return [];")
    out.append("  return Array.isArray(c) ? c.map((_, i) => String(i)) : Object.keys(c);")
    out.append("}")
    out.append("")
    out.append("/**")
    out.append(" * Stage-B planner: pick at most `budget` follow-ups for the triage answer, skipping the")
    out.append(" * ones Tier-0 lexical evidence already proved (asking those wastes the visitor's time and")
    out.append(" * the device's battery for an answer we already hold).")
    out.append(" */")
    out.append("export function bankFollowups(bankId: BankId, triageKey: string, alreadyKnown: Set<string> = new Set(), budget = 3): QuestionSet {")
    out.append("  const pools = BANKS[bankId].followups as Record<string, QuestionSet | undefined>;")
    out.append("  const pool: QuestionSet = pools[triageKey] ?? {};")
    out.append("  const out: QuestionSet = {};")
    out.append("  for (const [qid, q] of Object.entries(pool)) {")
    out.append("    if (Object.keys(out).length >= Math.max(0, budget)) break;")
    out.append("    if (alreadyKnown.has(qid)) continue;")
    out.append("    out[qid] = q;")
    out.append("  }")
    out.append("  return out;")
    out.append("}")
    out.append("")
    out.append("/** Total questions a bank can ever ask: 1 triage + the largest follow-up set. */")
    out.append("export function bankMaxQuestions(bankId: BankId): number {")
    out.append("  const sizes = Object.values(BANKS[bankId].followups).map((q) => Object.keys(q).length);")
    out.append("  return 1 + (sizes.length ? Math.max(...sizes) : 0);")
    out.append("}")
    out.append("")
    return "\n".join(out)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="generate src/lib/banks.ts from ml/banks.json")
    ap.add_argument("--banks", default=BANKS_JSON)
    ap.add_argument("--out", default=OUT_TS, help="output path, or '-' for stdout")
    ap.add_argument("--check", action="store_true", help="exit 1 when the generated file is stale")
    args = ap.parse_args(argv)

    with open(args.banks, "r", encoding="utf-8") as f:
        banks = json.load(f)
    validate(banks)
    sha = canonical_sha(banks)
    text = render(banks, sha)

    if args.out == "-":
        sys.stdout.write(text)
        return 0

    previous = None
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            previous = f.read()
    if args.check:
        if previous != text:
            print(f"[gen_banks_ts] STALE: {os.path.relpath(args.out, REPO)} differs from ml/banks.json",
                  file=sys.stderr)
            return 1
        print(f"[gen_banks_ts] up to date ({sha[:12]}, {len(banks['tasks'])} banks)")
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    changed = "updated" if previous is not None and previous != text else "written"
    print(f"[gen_banks_ts] {changed} {os.path.relpath(args.out, REPO)} "
          f"({len(banks['tasks'])} banks, sha {sha[:12]}, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
