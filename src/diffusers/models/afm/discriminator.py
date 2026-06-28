from __future__ import annotations

import torch
from torch import nn

from ._dit import build_dit_config, get_dit_transformer_class, load_legacy_dit_state_dict


class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        use_t: bool = True,
        depth: int = 28,
        hidden_size: int = 1152,
        patch_size: int = 2,
        num_heads: int = 16,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.0,
        input_size: int = 32,
        num_classes: int = 1000,
        in_channels: int = 4,
        eps: float = 1e-6,
        **_,
    ):
        super().__init__()
        self.use_t = use_t
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
        self.dis_embed = nn.Parameter(torch.randn([hidden_size]) * 0.02)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        from ._dit import remap_transformer_state_dict

        remapped = remap_transformer_state_dict(
            state_dict,
            self._legacy_kwargs,
            extra_prefixes=("dis_embed", "final_layer."),
        )
        return super().load_state_dict(remapped, strict=strict)

    def forward(self, x, y, t=None):
        hidden_states = self.transformer.pos_embed(x)
        dis_token = self.dis_embed.view(1, 1, -1).expand(hidden_states.shape[0], 1, -1)
        hidden_states = torch.cat([dis_token, hidden_states], dim=1)

        if self.use_t and t is not None:
            timestep = (t * 1000).long()
        else:
            timestep = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

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

        hidden_states = hidden_states[:, :1, :]
        return self.final_layer(hidden_states).view(-1)
