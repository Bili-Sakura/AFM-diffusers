"""Helpers for AFM models built on Hugging Face DiTTransformer2DModel."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..._hf import get_hf_attr
from ..transformers.transformer_sit import TimestepEmbedder


def get_dit_transformer_class():
    return get_hf_attr("diffusers.models.transformers.dit_transformer_2d.DiTTransformer2DModel")


def build_dit_config(
    *,
    depth: int = 28,
    hidden_size: int = 1152,
    patch_size: int = 2,
    num_heads: int = 16,
    learn_sigma: bool = False,
    class_dropout_prob: float = 0.0,
    input_size: int = 32,
    num_classes: int = 1000,
    in_channels: int = 4,
) -> Dict[str, Any]:
    attention_head_dim = hidden_size // num_heads
    num_embeds_ada_norm = num_classes + (1 if class_dropout_prob > 0 else 0)
    out_channels = in_channels * 2 if learn_sigma else in_channels
    return {
        "sample_size": input_size,
        "num_layers": depth,
        "num_attention_heads": num_heads,
        "attention_head_dim": attention_head_dim,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "patch_size": patch_size,
        "attention_bias": True,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": num_embeds_ada_norm,
        "norm_type": "ada_norm_zero",
        "norm_elementwise_affine": False,
        "dropout": 0.0,
        "norm_num_groups": 32,
        "norm_eps": 1e-5,
        "upcast_attention": False,
    }


def _dit_model_name(depth: int, patch_size: int, hidden_size: int, num_heads: int) -> Optional[str]:
    attention_head_dim = hidden_size // num_heads
    for name, preset in {
        "DiT-XL/2": (28, 2, 72),
        "DiT-L/2": (24, 2, 64),
        "DiT-B/2": (12, 2, 64),
        "DiT-S/2": (12, 2, 64),
    }.items():
        if depth == preset[0] and patch_size == preset[1] and attention_head_dim == preset[2]:
            return name
    return None


def load_legacy_dit_state_dict(
    state_dict: Dict[str, torch.Tensor],
    depth: int,
    patch_size: int,
    hidden_size: int,
    num_heads: int,
) -> Dict[str, torch.Tensor]:
    if "transformer_blocks.0.attn1.to_q.weight" in state_dict:
        return state_dict
    from ...dit_utils.conversion import convert_original_state_dict

    model_name = _dit_model_name(depth, patch_size, hidden_size, num_heads)
    if model_name is None:
        return state_dict
    return convert_original_state_dict(state_dict, model_name)


def forward_dit_output(
    transformer,
    hidden_states: torch.Tensor,
    timestep: Optional[torch.Tensor],
    class_labels: torch.Tensor,
    extra_cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    hidden_states = transformer.pos_embed(hidden_states)

    for block in transformer.transformer_blocks:
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            hidden_states = transformer._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                None,
                None,
                timestep,
                None,
                class_labels,
            )
        else:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                timestep=timestep,
                cross_attention_kwargs=None,
                class_labels=class_labels,
            )

    conditioning = transformer.transformer_blocks[0].norm1.emb(
        timestep, class_labels, hidden_dtype=hidden_states.dtype
    )
    if extra_cond is not None:
        conditioning = conditioning + extra_cond
    shift, scale = transformer.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
    hidden_states = transformer.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
    hidden_states = transformer.proj_out_2(hidden_states)

    height = width = int(hidden_states.shape[1] ** 0.5)
    hidden_states = hidden_states.reshape(
        shape=(-1, height, width, transformer.patch_size, transformer.patch_size, transformer.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    return hidden_states.reshape(
        shape=(-1, transformer.out_channels, height * transformer.patch_size, width * transformer.patch_size)
    )


def remap_transformer_state_dict(
    state_dict: Dict[str, torch.Tensor],
    legacy_kwargs: Dict[str, int],
    extra_prefixes: tuple[str, ...] = (),
) -> Dict[str, torch.Tensor]:
    state_dict = load_legacy_dit_state_dict(state_dict, **legacy_kwargs)
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in extra_prefixes):
            remapped[key] = value
        elif key.startswith("transformer."):
            remapped[key] = value
        else:
            remapped[f"transformer.{key}"] = value
    return remapped


def split_output_channels(output: torch.Tensor, in_channels: int) -> torch.Tensor:
    if output.shape[1] == in_channels:
        return output
    output, _ = output.chunk(2, dim=1)
    return output
