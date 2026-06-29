# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path

_LOCAL_SRC = Path(__file__).resolve().parents[3]


def _load_upstream_attr(module_path: str, attr_name: str):
    stashed = {}
    for name in list(sys.modules):
        if not (name == "diffusers" or name.startswith("diffusers.")):
            continue
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        mod_paths = getattr(mod, "__path__", None)
        is_local = f"{_LOCAL_SRC / 'diffusers'}" in mod_file.replace("\\", "/")
        if mod_paths is not None:
            is_local = is_local or any(f"{_LOCAL_SRC / 'diffusers'}" in str(path) for path in mod_paths)
        if is_local:
            stashed[name] = sys.modules.pop(name)
    original_path = sys.path[:]
    try:
        sys.path = [entry for entry in sys.path if Path(entry).resolve() != _LOCAL_SRC.resolve()]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    finally:
        sys.path = original_path
        sys.modules.update(stashed)


__all__ = ["DiTTransformer2DModel"]


def __getattr__(name: str):
    if name == "DiTTransformer2DModel":
        return _load_upstream_attr("diffusers.models.transformers.dit_transformer_2d", "DiTTransformer2DModel")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
