# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

from diffusers.configuration_utils import register_to_config

from .transformer_sit import SiTTransformer2DModel


class CAFMSiTGeneratorModel(SiTTransformer2DModel):
    def forward(self, x, y, t):
        # flip t and v because training uses x0=image, x1=noise,
        # but SiT was trained with x0=noise, x1=image.
        return -self._forward_impl(x, 1.0 - t, y)


class CAFMSiTDiscriminatorModel(SiTTransformer2DModel):
    @register_to_config
    def __init__(self, *, norm_type: str = "rms", eps: float = 1e-6, **kwargs):
        super().__init__(norm_type=norm_type, **kwargs)
        self.dis_embed = nn.Parameter(torch.randn([kwargs["hidden_size"]]) * 0.02)
        norm_cls = self._get_norm(norm_type)
        self.final_layer = nn.Sequential(
            norm_cls(kwargs["hidden_size"], elementwise_affine=False, eps=eps),
            nn.Linear(kwargs["hidden_size"], 1, bias=False),
        )

    def forward(self, x, y, t):
        t = 1.0 - t

        x = self.x_embedder(x) + self.pos_embed.type_as(x)
        c = self.y_embedder(y, self.training).type_as(x)
        if hasattr(self, "t_embedder"):
            c = c + self.t_embedder(t)
        c = c.type_as(x)

        d = self.dis_embed.view(1, 1, -1)
        d = d.repeat(x.size(0), 1, 1)
        d = d.type_as(x)
        x = torch.cat([d, x], dim=1)

        for block in self.blocks:
            x = block(x, c)
        x = x[:, :1, :]
        x = self.final_layer(x)
        return x.view(-1)
