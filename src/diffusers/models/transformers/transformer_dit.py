# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..._hf import get_hf_attr

__all__ = ["DiTTransformer2DModel"]


def __getattr__(name: str):
    if name == "DiTTransformer2DModel":
        return get_hf_attr("diffusers.models.transformers.dit_transformer_2d.DiTTransformer2DModel")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
