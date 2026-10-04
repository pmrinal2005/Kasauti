#!/usr/bin/env python3
"""Build a *mini* Laya checkpoint with the real mmBERT architecture and the real 256k tokenizer.

Why: the ``kaggle`` profile path (pinned Hub layout → ``laya.common.build_model`` →
``load_state_dict(strict=True)`` → RLCD → laya-format save → vocabulary pruning of a 256 000-row
table → fp32/INT8/Q4 export → reference ``ONNXAgent``) is the code that only ever runs on a GPU
box. The ``smoke`` profile uses a hand-rolled tiny encoder and a tiny tokenizer, so it cannot reach
any of it. This script writes a directory that is laid out *exactly* like
``convaiinnovations/laya/multilingual`` (``encoder/config.json``, ``model.safetensors``,
``rl_agent_config.json``, ``tokenizer/``) but with a 2-layer, 64-wide ModernBERT, so the whole
real path runs on a laptop CPU in a few minutes:

    python -m ml.scripts.make_mini_checkpoint --out ml/out/mini-ckpt
    python -m ml.kasauti_ml.pipeline --profile kaggle --model-dir ml/out/mini-ckpt \
        --out ml/out/mini-run --variants 2 --epochs 1 --max-len 256 --head-max-len 128 \
        --micro-batch 4 --grad-accum 1 --gates ml/scripts/gates_plumbing.json --reference-check

The weights are random — accuracy numbers from such a run are meaningless; only the plumbing is
being tested. The tokenizer comes from the pinned Hub revision (~34 MB download, cached).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

PINNED = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="ml/out/mini-ckpt")
    ap.add_argument("--model-id", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default="multilingual")
    ap.add_argument("--revision", default=PINNED)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.environ.setdefault("USE_TF", "0")
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from transformers import AutoConfig

    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model

    torch.manual_seed(args.seed)
    sub = args.subfolder + "/" if args.subfolder else ""
    os.makedirs(os.path.join(args.out, "encoder"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "tokenizer"), exist_ok=True)
    for rel in ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "rl_agent_config.json",
                "encoder/config.json"):
        src = hf_hub_download(args.model_id, sub + rel, revision=args.revision)
        shutil.copyfile(src, os.path.join(args.out, rel))

    # shrink the encoder: same model_type / tokenizer / RoPE / special ids, tiny width and depth
    with open(os.path.join(args.out, "encoder", "config.json"), encoding="utf-8") as f:
        ecfg = json.load(f)
    heads = max(1, args.hidden // 32)
    ecfg.update({"hidden_size": args.hidden, "intermediate_size": args.hidden * 2,
                 "num_attention_heads": heads, "num_hidden_layers": args.layers,
                 "global_attn_every_n_layers": 3,
                 "layer_types": [("full_attention" if i % 3 == 0 else "sliding_attention")
                                 for i in range(args.layers)]})
    with open(os.path.join(args.out, "encoder", "config.json"), "w", encoding="utf-8") as f:
        json.dump(ecfg, f, indent=2)

    with open(os.path.join(args.out, "rl_agent_config.json"), encoding="utf-8") as f:
        run_cfg = json.load(f)
    run_cfg["training"] = {"mini_plumbing_checkpoint": True, "base_revision": args.revision}
    with open(os.path.join(args.out, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # real DecisionModel around a randomly initialised ModernBERT built from the shrunk config
    from transformers import AutoModel

    from laya.common import DecisionModel, _apply_rope_config  # noqa: F401  (same path build_model takes)

    cfg = AutoConfig.from_pretrained(os.path.join(args.out, "encoder"))
    _apply_rope_config(cfg)
    enc = AutoModel.from_config(cfg, attn_implementation="sdpa")
    model = DecisionModel(enc, run_cfg.get("head_layers", 2), len(run_cfg.get("act_costs", {})) + 1)
    sd = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(args.out, "model.safetensors"))

    # round-trip through the exact loader the pipeline uses, strict=True
    from safetensors.torch import load_file

    check = build_model(run_cfg, encoder_dir=os.path.join(args.out, "encoder"))
    check.load_state_dict(load_file(os.path.join(args.out, "model.safetensors")), strict=True)
    _fix_tokenizer_config(args.out)
    n = sum(p.numel() for p in check.parameters())
    print(f"[mini-ckpt] wrote {args.out}: ModernBERT {args.layers}x{args.hidden}, vocab "
          f"{ecfg['vocab_size']}, {n/1e6:.1f}M params (strict reload ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
