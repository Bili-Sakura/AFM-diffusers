"""Self-contained AFM generator for Hugging Face diffusers checkpoints."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import nn


def get_dit_transformer_class():
    from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel

    return DiTTransformer2DModel


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


def _convert_legacy_dit_state_dict(state_dict: Dict[str, torch.Tensor], depth: int) -> Dict[str, torch.Tensor]:
    converted = copy.deepcopy(state_dict)

    converted["pos_embed.proj.weight"] = converted.pop("x_embedder.proj.weight").clone().contiguous()
    converted["pos_embed.proj.bias"] = converted.pop("x_embedder.proj.bias").clone().contiguous()

    if "t_embedder.mlp.0.weight" in converted:
        timestep_weights = {
            "linear_1.weight": converted.pop("t_embedder.mlp.0.weight"),
            "linear_1.bias": converted.pop("t_embedder.mlp.0.bias"),
            "linear_2.weight": converted.pop("t_embedder.mlp.2.weight"),
            "linear_2.bias": converted.pop("t_embedder.mlp.2.bias"),
        }
    else:
        timestep_weights = None
    class_embedding = converted.pop("y_embedder.embedding_table.weight")
    if class_embedding.shape[0] == 1000:
        null_class = torch.zeros(1, class_embedding.shape[1], dtype=class_embedding.dtype)
        class_embedding = torch.cat([class_embedding, null_class], dim=0)

    for block_idx in range(depth):
        if timestep_weights is not None:
            for key, tensor in timestep_weights.items():
                converted[f"transformer_blocks.{block_idx}.norm1.emb.timestep_embedder.{key}"] = tensor.clone()
        converted[f"transformer_blocks.{block_idx}.norm1.emb.class_embedder.embedding_table.weight"] = (
            class_embedding.clone()
        )
        converted[f"transformer_blocks.{block_idx}.norm1.linear.weight"] = converted[
            f"blocks.{block_idx}.adaLN_modulation.1.weight"
        ]
        converted[f"transformer_blocks.{block_idx}.norm1.linear.bias"] = converted[
            f"blocks.{block_idx}.adaLN_modulation.1.bias"
        ]

        q, k, v = torch.chunk(converted[f"blocks.{block_idx}.attn.qkv.weight"], 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(converted[f"blocks.{block_idx}.attn.qkv.bias"], 3, dim=0)
        converted[f"transformer_blocks.{block_idx}.attn1.to_q.weight"] = q
        converted[f"transformer_blocks.{block_idx}.attn1.to_q.bias"] = q_bias
        converted[f"transformer_blocks.{block_idx}.attn1.to_k.weight"] = k
        converted[f"transformer_blocks.{block_idx}.attn1.to_k.bias"] = k_bias
        converted[f"transformer_blocks.{block_idx}.attn1.to_v.weight"] = v
        converted[f"transformer_blocks.{block_idx}.attn1.to_v.bias"] = v_bias
        converted[f"transformer_blocks.{block_idx}.attn1.to_out.0.weight"] = converted[
            f"blocks.{block_idx}.attn.proj.weight"
        ]
        converted[f"transformer_blocks.{block_idx}.attn1.to_out.0.bias"] = converted[
            f"blocks.{block_idx}.attn.proj.bias"
        ]
        converted[f"transformer_blocks.{block_idx}.ff.net.0.proj.weight"] = converted[
            f"blocks.{block_idx}.mlp.fc1.weight"
        ]
        converted[f"transformer_blocks.{block_idx}.ff.net.0.proj.bias"] = converted[
            f"blocks.{block_idx}.mlp.fc1.bias"
        ]
        converted[f"transformer_blocks.{block_idx}.ff.net.2.weight"] = converted[
            f"blocks.{block_idx}.mlp.fc2.weight"
        ]
        converted[f"transformer_blocks.{block_idx}.ff.net.2.bias"] = converted[
            f"blocks.{block_idx}.mlp.fc2.bias"
        ]

        for suffix in (
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
            "adaLN_modulation.1.weight",
            "adaLN_modulation.1.bias",
        ):
            converted.pop(f"blocks.{block_idx}.{suffix}", None)

    converted["proj_out_1.weight"] = converted.pop("final_layer.adaLN_modulation.1.weight")
    converted["proj_out_1.bias"] = converted.pop("final_layer.adaLN_modulation.1.bias")
    converted["proj_out_2.weight"] = converted.pop("final_layer.linear.weight")
    converted["proj_out_2.bias"] = converted.pop("final_layer.linear.bias")

    converted.pop("pos_embed", None)
    for block_idx in range(depth):
        for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            converted.pop(f"blocks.{block_idx}.{suffix}", None)

    return {key: tensor.detach().clone().contiguous() for key, tensor in converted.items()}


def load_legacy_dit_state_dict(
    state_dict: Dict[str, torch.Tensor],
    depth: int,
    patch_size: int,
    hidden_size: int,
    num_heads: int,
) -> Dict[str, torch.Tensor]:
    if "transformer_blocks.0.attn1.to_q.weight" in state_dict:
        return state_dict
    if any(key.startswith("blocks.") for key in state_dict):
        return _convert_legacy_dit_state_dict(state_dict, depth)
    return state_dict


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


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.timestep_embedding(t.float(), self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        return self.mlp(emb.to(dtype=weight_dtype))


class Generator(nn.Module):
    def __init__(
        self,
        *,
        use_t_src: bool = False,
        use_t_tgt: bool = False,
        depth: int = 28,
        hidden_size: int = 1152,
        patch_size: int = 2,
        num_heads: int = 16,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.0,
        input_size: int = 32,
        num_classes: int = 1000,
        in_channels: int = 4,
        **_,
    ):
        super().__init__()
        self.use_t_src = use_t_src
        self.use_t_tgt = use_t_tgt
        self._legacy_kwargs = {
            "depth": depth,
            "hidden_size": hidden_size,
            "patch_size": patch_size,
            "num_heads": num_heads,
        }
        dit_cls = get_dit_transformer_class()
        self.transformer = dit_cls(
            **build_dit_config(
                depth=depth,
                hidden_size=hidden_size,
                patch_size=patch_size,
                num_heads=num_heads,
                learn_sigma=learn_sigma,
                class_dropout_prob=class_dropout_prob,
                input_size=input_size,
                num_classes=num_classes,
                in_channels=in_channels,
            )
        )
        self.in_channels = in_channels
        if use_t_tgt:
            self.t_tgt_embedder = TimestepEmbedder(hidden_size)
            nn.init.normal_(self.t_tgt_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_tgt_embedder.mlp[2].weight, std=0.02)

    def load_state_dict(self, state_dict, strict: bool = True):
        extra = ("t_tgt_embedder.",)
        has_extra = any(key.startswith(extra) for key in state_dict)
        has_t_embedder = any("t_embedder" in key for key in state_dict)
        remapped = remap_transformer_state_dict(state_dict, self._legacy_kwargs, extra_prefixes=extra)
        if self.use_t_tgt and not has_extra:
            strict = False
        if not has_t_embedder:
            strict = False
        out = super().load_state_dict(remapped, strict=strict)
        if not has_t_embedder:
            for name, param in self.named_parameters():
                if "timestep_embedder" in name:
                    param.data.zero_()
        return out

    def forward(self, x, y, t_src=None, t_tgt=None):
        if self.use_t_src and t_src is not None:
            timestep = (t_src * 1000).long()
        else:
            timestep = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        extra_cond = None
        if self.use_t_tgt and t_tgt is not None:
            extra_cond = self.t_tgt_embedder(t_tgt * 1000)

        output = forward_dit_output(self.transformer, x, timestep, y, extra_cond=extra_cond)
        return split_output_channels(output, self.in_channels)


class GeneratorDeep(nn.Module):
    def __init__(
        self,
        *,
        repeat: int,
        depth: int = 28,
        hidden_size: int = 1152,
        patch_size: int = 2,
        num_heads: int = 16,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.0,
        input_size: int = 32,
        num_classes: int = 1000,
        in_channels: int = 4,
        **_,
    ):
        super().__init__()
        self.repeat = repeat
        self._legacy_kwargs = {
            "depth": depth,
            "hidden_size": hidden_size,
            "patch_size": patch_size,
            "num_heads": num_heads,
        }
        dit_cls = get_dit_transformer_class()
        self.transformer = dit_cls(
            **build_dit_config(
                depth=depth,
                hidden_size=hidden_size,
                patch_size=patch_size,
                num_heads=num_heads,
                learn_sigma=learn_sigma,
                class_dropout_prob=class_dropout_prob,
                input_size=input_size,
                num_classes=num_classes,
                in_channels=in_channels,
            )
        )
        self.in_channels = in_channels

    def load_state_dict(self, state_dict, strict: bool = True):
        remapped = remap_transformer_state_dict(state_dict, self._legacy_kwargs)
        has_t_embedder = any("t_embedder" in key for key in state_dict)
        if not has_t_embedder:
            strict = False
        out = super().load_state_dict(remapped, strict=strict)
        if not has_t_embedder:
            for name, param in self.named_parameters():
                if "timestep_embedder" in name:
                    param.data.zero_()
        return out

    def forward(self, x, y, *args, **kwargs):
        hidden_states = self.transformer.pos_embed(x)
        for t in torch.arange(1, 0, -1 / self.repeat).tolist():
            timestep = torch.full([x.shape[0]], int(t * 1000), device=x.device, dtype=torch.long)
            for block in self.transformer.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=None,
                    class_labels=y,
                )

        conditioning = self.transformer.transformer_blocks[0].norm1.emb(
            torch.zeros(x.shape[0], device=x.device, dtype=torch.long),
            y,
            hidden_dtype=hidden_states.dtype,
        )
        shift, scale = self.transformer.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.transformer.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.transformer.proj_out_2(hidden_states)

        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.transformer.patch_size,
                self.transformer.patch_size,
                self.transformer.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.transformer.out_channels, height * self.transformer.patch_size, width * self.transformer.patch_size)
        )
        return split_output_channels(output, self.in_channels)


class AFMGenerator2DModel(Generator, ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        model_type: str = "AFM-XL/2",
        architecture: str = "standard",
        repeat: int = 1,
        use_t_src: bool = False,
        use_t_tgt: bool = False,
        depth: int = 28,
        hidden_size: int = 1152,
        patch_size: int = 2,
        num_heads: int = 16,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.0,
        input_size: int = 32,
        num_classes: int = 1000,
        in_channels: int = 4,
        pred_type: str = "x",
        num_inference_steps: int = 1,
    ) -> None:
        if architecture == "deep":
            raise ValueError("Use AFMGeneratorDeep2DModel for deep AFM checkpoints.")
        Generator.__init__(
            self,
            use_t_src=use_t_src,
            use_t_tgt=use_t_tgt,
            depth=depth,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_heads=num_heads,
            learn_sigma=learn_sigma,
            class_dropout_prob=class_dropout_prob,
            input_size=input_size,
            num_classes=num_classes,
            in_channels=in_channels,
        )


class AFMGeneratorDeep2DModel(GeneratorDeep, ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        model_type: str = "AFM-XL/2",
        architecture: str = "deep",
        repeat: int = 2,
        use_t_src: bool = False,
        use_t_tgt: bool = False,
        depth: int = 28,
        hidden_size: int = 1152,
        patch_size: int = 2,
        num_heads: int = 16,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.0,
        input_size: int = 32,
        num_classes: int = 1000,
        in_channels: int = 4,
        pred_type: str = "x",
        num_inference_steps: int = 1,
    ) -> None:
        GeneratorDeep.__init__(
            self,
            repeat=repeat,
            depth=depth,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_heads=num_heads,
            learn_sigma=learn_sigma,
            class_dropout_prob=class_dropout_prob,
            input_size=input_size,
            num_classes=num_classes,
            in_channels=in_channels,
        )


__all__ = ["AFMGenerator2DModel", "AFMGeneratorDeep2DModel"]
