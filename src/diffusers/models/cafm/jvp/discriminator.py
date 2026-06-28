from torch import nn
from torch.func import jvp, vmap


class DiscriminatorJVP(nn.Module):
    """Discriminator wrapper for JVP and vmap used in CAFM ImageNet training."""

    def __init__(self, dis):
        super().__init__()
        self.dis = dis

    def forward(self, x, y, t, dx, dt):
        def dis_fn(x, t):
            return self.dis(x, y, t)

        def dis_jvp(dx, dt):
            return jvp(dis_fn, (x, t), (dx, dt))

        def dis_jvp_vmap(dx, dt):
            return vmap(dis_jvp)(dx, dt)

        if x.ndim == dx.ndim:
            o, do = dis_jvp(dx, dt)
        else:
            o, do = dis_jvp_vmap(dx, dt)

        return o, do
