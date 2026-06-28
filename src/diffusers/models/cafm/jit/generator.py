from ...transformers.transformer_jit import JiTTransformer2DModel


class Generator(JiTTransformer2DModel):
    def forward(self, x, y, t):
        x_pred = super().forward(sample=x, timestep=1 - t, class_labels=y).sample
        v = (x - x_pred) / t.view(-1, 1, 1, 1).clamp_min(0.05)
        return v
