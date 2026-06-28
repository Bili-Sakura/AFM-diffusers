from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._dit import (
    build_dit_config,
    get_dit_transformer_class,
    load_legacy_dit_state_dict,
    split_output_channels,
)


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
        from ._dit import remap_transformer_state_dict

        remapped = remap_transformer_state_dict(state_dict, self._legacy_kwargs)
        return super().load_state_dict(remapped, strict=strict)

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
            None, y, hidden_dtype=hidden_states.dtype
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
