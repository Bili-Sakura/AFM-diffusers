from __future__ import annotations

import torch
from torch import nn

from ._dit import (
    build_dit_config,
    forward_dit_output,
    get_dit_transformer_class,
    load_legacy_dit_state_dict,
    split_output_channels,
)
from ..transformers.transformer_sit import TimestepEmbedder


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
        from ._dit import remap_transformer_state_dict

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
