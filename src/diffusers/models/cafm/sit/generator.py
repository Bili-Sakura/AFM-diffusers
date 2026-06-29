from .sit_mod import SiTMod


class Generator(SiTMod):
    def forward(self, x, y, t):
        return -super().forward(
            hidden_states=x,
            timestep=1.0 - t,
            class_labels=y,
            return_dict=True,
        ).sample
