from .sit_mod import SiTMod


class Generator(SiTMod):
    def forward(self, x, y, t):
        return -super().forward(x, (1.0 - t), y).sample
