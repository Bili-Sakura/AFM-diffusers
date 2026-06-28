import torch
from torch import nn

from ...transformers.transformer_jit import JiTTransformer2DModel


class Discriminator(JiTTransformer2DModel):
    def __init__(self, **kwargs):
        depth = kwargs.pop("depth", kwargs.get("num_layers", 32))
        num_heads = kwargs.pop("num_heads", kwargs.get("num_attention_heads", 16))
        super().__init__(
            hidden_size=kwargs.get("hidden_size", 1280),
            num_layers=depth,
            num_attention_heads=num_heads,
            bottleneck_dim=kwargs.get("bottleneck_dim", 256),
            in_context_len=kwargs.get("in_context_len", 32),
            in_context_start=kwargs.get("in_context_start", 10),
            patch_size=kwargs.get("patch_size", 16),
            sample_size=kwargs.get("sample_size", 256),
            dropout=kwargs.get("dropout", 0.2),
            attention_dropout=kwargs.get("attention_dropout", 0.0),
            num_classes=kwargs.get("num_classes", 1000),
        )
        del self.norm_final
        del self.linear_final
        del self.act_final
        del self.adaLN_modulation_final
        self.proj_out = nn.Sequential(
            nn.RMSNorm(self.hidden_size, elementwise_affine=False),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, x, y, t):
        t = 1 - t

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x += self.pos_embed.to(x.dtype)

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb.to(in_context_tokens.dtype)
                x = torch.cat([in_context_tokens, x], dim=1)
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, feat_rope=rope)

        x = x[:, 0]
        return self.proj_out(x).reshape(-1)
