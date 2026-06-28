import torch
from torch import nn

from .sit_mod import SiTMod, get_norm


class Discriminator(SiTMod):
    def __init__(self, *, norm_type="rms", **kwargs):
        super().__init__(norm_type=norm_type, **kwargs)
        self.dis_embed = nn.Parameter(torch.randn([kwargs["hidden_size"]]) * 0.02)
        self.final_layer = nn.Sequential(
            get_norm(norm_type)(kwargs["hidden_size"], elementwise_affine=False, eps=kwargs.pop("eps", 1e-6)),
            nn.Linear(kwargs["hidden_size"], 1, bias=False),
        )

    def forward(self, x, y, t):
        t = 1.0 - t

        x = self.x_embedder(x) + self.pos_embed.type_as(x)
        c = self.y_embedder(y, self.training).type_as(x)
        if hasattr(self, "t_embedder"):
            c = c + self.t_embedder(t)
        c = c.type_as(x)

        d = self.dis_embed.view(1, 1, -1).repeat(x.size(0), 1, 1).type_as(x)
        x = torch.cat([d, x], dim=1)

        for block in self.blocks:
            x = block(x, c)
        x = x[:, :1, :]
        return self.final_layer(x).view(-1)
