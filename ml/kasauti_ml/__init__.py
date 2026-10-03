"""Kasauti ML — fine-tune, calibrate, gate and export Laya for on-device browser inference.

This package is **training-side only**: nothing here ships to users. It targets Kaggle's free
2x T4 notebooks (or any CPU box for the smoke profile) and produces the artifacts the web app
loads: a pruned tokenizer, an INT8 WASM graph, a 4-bit WebGPU graph, a calibration manifest and
parity vectors.

Design rule: every function that mirrors the reference Laya implementation says so in its
docstring, and `tests/test_reference_parity.py` checks those mirrors against the installed
`laya` package when it is importable.
"""

__version__ = "1.0.0"
BANKS_VERSION = "1.0.0"
