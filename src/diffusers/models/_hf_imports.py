"""Lazy imports from the installed Hugging Face diffusers package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_LOCAL_SRC = Path(__file__).resolve().parents[2]


def _load_upstream_attr(module_path: str, attr_name: str):
    stashed = {}
    for name in list(sys.modules):
        if name == "diffusers" or name.startswith("diffusers."):
            stashed[name] = sys.modules.pop(name)
    original_path = sys.path[:]
    try:
        sys.path = [entry for entry in sys.path if Path(entry).resolve() != _LOCAL_SRC.resolve()]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    finally:
        sys.path = original_path
        sys.modules.update(stashed)


def get_config_mixin():
    return _load_upstream_attr("diffusers.configuration_utils", "ConfigMixin")


def get_register_to_config():
    return _load_upstream_attr("diffusers.configuration_utils", "register_to_config")


def get_model_mixin():
    return _load_upstream_attr("diffusers.models.modeling_utils", "ModelMixin")


def get_base_output():
    return _load_upstream_attr("diffusers.utils", "BaseOutput")


def get_transformer_2d_model_output():
    return _load_upstream_attr("diffusers.models.modeling_outputs", "Transformer2DModelOutput")


def get_rms_norm():
    return _load_upstream_attr("diffusers.models.normalization", "RMSNorm")
