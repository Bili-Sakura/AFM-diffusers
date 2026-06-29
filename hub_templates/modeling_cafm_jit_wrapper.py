class CAFMJiTGenerator2DModel(JiTTransformer2DModel):
    """JiT backbone with continuous adversarial flow velocity head."""

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_pred = super().forward(
            hidden_states=x,
            timestep=1.0 - t,
            class_labels=y,
            return_dict=True,
        ).sample
        t_batch = t.view(-1, 1, 1, 1).clamp_min(0.05)
        return (x - x_pred) / t_batch
