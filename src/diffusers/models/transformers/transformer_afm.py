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

from .embedding_utils import TimestepEmbedder
from .transformer_dit import DiTTransformer2DModel


class AFMGeneratorModel(DiTTransformer2DModel):
    @register_to_config
    def __init__(self, *, use_t_src: bool = False, use_t_tgt: bool = False, **kwargs):
        super().__init__(**kwargs)
        if not use_t_src:
            del self.t_embedder
        if use_t_tgt:
            self.t_tgt_embedder = TimestepEmbedder(kwargs["hidden_size"])
            nn.init.normal_(self.t_tgt_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_tgt_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, y, t_src=None, t_tgt=None):
        x = self.x_embedder(x) + self.pos_embed
        c = self.y_embedder(y, self.training)
        if hasattr(self, "t_embedder"):
            c = c + self.t_embedder(t_src * 1000)
        if hasattr(self, "t_tgt_embedder"):
            c = c + self.t_tgt_embedder(t_tgt * 1000)

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


class AFMDiscriminatorModel(DiTTransformer2DModel):
    @register_to_config
    def __init__(self, *, use_t: bool = False, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dis_embed = nn.Parameter(torch.randn([kwargs["hidden_size"]]) * 0.02)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(kwargs["hidden_size"], elementwise_affine=False, eps=eps),
            nn.Linear(kwargs["hidden_size"], 1, bias=False),
        )
        if not use_t:
            del self.t_embedder

    def forward(self, x, y, t=None):
        x = self.x_embedder(x) + self.pos_embed
        c = self.y_embedder(y, self.training)
        if hasattr(self, "t_embedder"):
            c = c + self.t_embedder(t * 1000)

        d = self.dis_embed.view(1, 1, -1)
        d = d.repeat(x.size(0), 1, 1)
        x = torch.cat([d, x], dim=1)

        for block in self.blocks:
            x = block(x, c)
        x = x[:, :1, :]
        x = self.final_layer(x)
        return x.view(-1)


class AFMClassifierModel(DiTTransformer2DModel):
    @register_to_config
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_embed = nn.Parameter(torch.randn([kwargs["hidden_size"]]) * 0.02)
        self.final_layer = nn.Sequential(
            nn.LayerNorm(kwargs["hidden_size"], elementwise_affine=False, eps=1e-6),
            nn.Linear(kwargs["hidden_size"], kwargs.get("num_classes", 1000), bias=True),
        )
        del self.y_embedder

    def forward(self, x, t=None):
        x = self.x_embedder(x) + self.pos_embed
        e = self.t_embedder(t * 1000)

        o = self.cls_embed.view(1, 1, -1)
        o = o.repeat(x.size(0), 1, 1)
        x = torch.cat([o, x], dim=1)

        for block in self.blocks:
            x = block(x, e)
        o = x[:, :1, :]
        o = self.final_layer(o)
        o = o.squeeze(1)
        return o


class AFMGeneratorDeepModel(DiTTransformer2DModel):
    @register_to_config
    def __init__(self, repeat: int, **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat

    def forward(self, x, y, *args, **kwargs):
        x = self.x_embedder(x) + self.pos_embed
        c = self.y_embedder(y, self.training)

        for t in torch.arange(1, 0, -1 / self.repeat).tolist():
            t = torch.full([len(c)], t, device=c.device, dtype=c.dtype)
            c_i = c + self.t_embedder(t * 1000)
            for block in self.blocks:
                x = block(x, c_i)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
