class CAFMSiTGenerator2DModel(SiTTransformer2DModel):
    """SiT backbone with continuous adversarial flow velocity head."""

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -super().forward(
            hidden_states=x,
            timestep=1.0 - t,
            class_labels=y,
            return_dict=True,
        ).sample
