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

from .transformer_jit import JiTTransformer2DModel


class CAFMJiTGeneratorModel(JiTTransformer2DModel):
    def forward(self, x, y, t):
        x_pred = super().forward(x=x, y=y, t=1 - t, return_dict=False)[0]
        v = (x - x_pred) / t.view(-1, 1, 1, 1).clamp_min(0.05)
        return v


class CAFMJiTDiscriminatorModel(JiTTransformer2DModel):
    @register_to_config
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.final_layer
        self.proj_out = nn.Sequential(
            nn.RMSNorm(kwargs["hidden_size"], elementwise_affine=False),
            nn.Linear(kwargs["hidden_size"], 1),
        )

    def forward(self, x, y, t):
        t = 1 - t
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, 0]
        x = self.proj_out(x)
        return x.reshape(-1)
