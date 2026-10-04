"""Staging and publishing of the exported artifacts.

Two destinations, one layout:

* a **local directory** — ``public/dev-model`` in the Kasauti repo for the CPU smoke profile, so
  the browser E2E test can exercise the real Worker/download/decode path without a 650 MB fetch;
* a **public Hugging Face *model* repository** — static LFS bytes over the Hub CDN. No Space, no
  compute, no quota: Hugging Face only ever serves files, so there is nothing to bill and nothing
  to keep warm.

Every published file is content-hashed (``kasauti-laya-int8-<sha8>.onnx``), so the service worker
can treat it as immutable: a new fine-tune is a new name, and a returning visitor never
re-downloads an unchanged model.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

from .export import file_meta, sha256_file


def stage(staging_dir: str, artifacts: Dict[str, str], manifest: Dict[str, Any],
          extra_files: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Copy ``artifacts`` (logical name → path) into ``staging_dir`` and drop ``manifest.json``.

    The manifest is written **last** and its ``files`` entries are rewritten to the staged,
    content-hashed names, so a client that fetches the manifest can never be told about a file
    that isn't there yet.
    """
    os.makedirs(staging_dir, exist_ok=True)
    staged: Dict[str, Dict[str, Any]] = {}
    for key, src in artifacts.items():
        base = os.path.basename(src)
        dst = os.path.join(staging_dir, base)
        shutil.copyfile(src, dst)
        staged[key] = file_meta(dst)
    tokenizer = manifest.get("tokenizer") or {}
    if tokenizer.get("src"):
        src = tokenizer["src"]
        dst = os.path.join(staging_dir, os.path.basename(src))
        shutil.copyfile(src, dst)
        tokenizer = {k: v for k, v in tokenizer.items() if k != "src"}
        tokenizer.update(file_meta(dst))
        manifest["tokenizer"] = tokenizer
    for name, src in (extra_files or {}).items():
        dst = os.path.join(staging_dir, os.path.basename(src))
        shutil.copyfile(src, dst)
    manifest = dict(manifest)
    manifest["files"] = staged
    with open(os.path.join(staging_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def verify_staged(staging_dir: str, manifest: Dict[str, Any]) -> List[str]:
    """Re-hash every staged file against the manifest. Returns the list of problems (empty = OK)."""
    problems: List[str] = []
    for key, meta in (manifest.get("files") or {}).items():
        p = os.path.join(staging_dir, meta["path"])
        if not os.path.exists(p):
            problems.append(f"{key}: missing {meta['path']}")
        elif sha256_file(p) != meta["sha256"]:
            problems.append(f"{key}: sha256 mismatch for {meta['path']}")
        elif os.path.getsize(p) != meta["bytes"]:
            problems.append(f"{key}: size mismatch for {meta['path']}")
    tok = manifest.get("tokenizer") or {}
    if tok.get("path") and not tok.get("src"):
        p = os.path.join(staging_dir, tok["path"])
        if not os.path.exists(p):
            problems.append(f"tokenizer: missing {tok['path']}")
    return problems


def publish_local(staging_dir: str, target_dir: str) -> str:
    """Copy the staging directory to a local path (the app's ``public/dev-model``)."""
    if os.path.abspath(staging_dir) == os.path.abspath(target_dir):
        return target_dir
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(staging_dir, target_dir)
    return target_dir


def publish_hf(staging_dir: str, repo_id: str, token: Optional[str] = None,
               commit_message: str = "kasauti: publish Laya browser build",
               private: bool = False) -> Dict[str, Any]:
    """Upload the staging directory to a public HF **model** repo (LFS-backed static bytes)."""
    try:
        from huggingface_hub import HfApi, create_repo
    except Exception as e:  # pragma: no cover - training-side optional dependency
        return {"ok": False, "reason": f"huggingface_hub unavailable: {e}"}
    try:
        api = HfApi(token=token)
        create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)
        api.upload_folder(folder_path=staging_dir, repo_id=repo_id, repo_type="model",
                          commit_message=commit_message, token=token)
        return {"ok": True, "repo": repo_id,
                "url": f"https://huggingface.co/{repo_id}/resolve/main/"}
    except Exception as e:  # pragma: no cover
        return {"ok": False, "reason": str(e)}


def print_next_steps(base_url: str, manifest: Dict[str, Any]) -> str:
    """The copy-pasteable integration step for the web app (also used in ml/README.md)."""
    return (
        "# 1. point the app at the published model repo (Vercel env var, then redeploy)\n"
        f"NEXT_PUBLIC_LAYA_MODEL_BASE={base_url}\n\n"
        "# 2. verify locally before deploying\n"
        "npm run typecheck && npm test && npm run build\n"
        "NEXT_PUBLIC_LAYA_MODEL_BASE=$PWD/public/dev-model npm start   # or `npm run dev`\n"
        "node tests/e2e-devmode.mjs http://localhost:3000   # asserts the Worker loads the model\n\n"
        f"# model version: {manifest.get('modelVersion')}\n"
        f"# evals: {json.dumps(manifest.get('evals', {}))[:200]}\n"
    )
