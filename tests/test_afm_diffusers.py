import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from common.diffusers_ext import register_diffusers_extensions

register_diffusers_extensions()

import pytest

torch = pytest.importorskip("torch")

from diffusers.models.transformers.transformer_afm import (
    AFMDiscriminatorModel,
    AFMGeneratorModel,
)
from diffusers.models.transformers.transformer_cafm_jit import (
    CAFMJiTDiscriminatorModel,
    CAFMJiTGeneratorModel,
)
from diffusers.models.transformers.transformer_cafm_sit import (
    CAFMSiTDiscriminatorModel,
    CAFMSiTGeneratorModel,
)
from diffusers.models.transformers.transformer_dit import DiTTransformer2DModel
from diffusers.models.transformers.transformer_jit import JiTTransformer2DModel
from diffusers.models.transformers.transformer_sit import SiTTransformer2DModel
from diffusers.models.discriminators.jvp_discriminator import DiscriminatorJVP


def test_dit_transformer_forward_shape():
    model = DiTTransformer2DModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    )
    latents = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([0.25, 0.5])
    class_labels = torch.tensor([1, 2])
    output = model(latents, timesteps, class_labels).sample
    assert output.shape == latents.shape


def test_afm_generator_forward_shape():
    model = AFMGeneratorModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
        use_t_src=True,
        use_t_tgt=True,
    )
    latents = torch.randn(2, 4, 8, 8)
    labels = torch.tensor([1, 2])
    t_src = torch.tensor([0.1, 0.2])
    t_tgt = torch.tensor([0.9, 0.8])
    output = model(latents, labels, t_src, t_tgt)
    assert output.shape == latents.shape


def test_afm_discriminator_forward_shape():
    model = AFMDiscriminatorModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
        use_t=True,
    )
    latents = torch.randn(2, 4, 8, 8)
    labels = torch.tensor([1, 2])
    t = torch.tensor([0.5, 0.6])
    output = model(latents, labels, t)
    assert output.shape == (2,)


def test_sit_transformer_forward_shape():
    model = SiTTransformer2DModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    )
    latents = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([0.25, 0.5])
    class_labels = torch.tensor([1, 2])
    output = model(latents, timesteps, class_labels).sample
    assert output.shape == latents.shape


def test_cafm_sit_generator_forward_shape():
    model = CAFMSiTGeneratorModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    )
    latents = torch.randn(2, 4, 8, 8)
    labels = torch.tensor([1, 2])
    t = torch.tensor([0.25, 0.5])
    output = model(latents, labels, t)
    assert output.shape == latents.shape


def test_cafm_sit_discriminator_forward_shape():
    model = CAFMSiTDiscriminatorModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    )
    latents = torch.randn(2, 4, 8, 8)
    labels = torch.tensor([1, 2])
    t = torch.tensor([0.25, 0.5])
    output = model(latents, labels, t)
    assert output.shape == (2,)


def test_jit_transformer_forward_shape():
    model = JiTTransformer2DModel(
        input_size=32,
        patch_size=16,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=4,
        bottleneck_dim=16,
        in_context_len=4,
        in_context_start=1,
    )
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0.25, 0.5])
    class_labels = torch.tensor([1, 2])
    output = model(images, timesteps, class_labels).sample
    assert output.shape == images.shape


def test_cafm_jit_generator_forward_shape():
    model = CAFMJiTGeneratorModel(
        input_size=32,
        patch_size=16,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=4,
        bottleneck_dim=16,
        in_context_len=4,
        in_context_start=1,
    )
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 2])
    t = torch.tensor([0.25, 0.5])
    output = model(images, labels, t)
    assert output.shape == images.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="JVP forward-mode AD requires CUDA in this environment")
def test_jvp_discriminator_forward_shape():
    dis = CAFMSiTDiscriminatorModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    ).cuda()
    model = DiscriminatorJVP(dis).cuda()
    latents = torch.randn(2, 4, 8, 8, device="cuda")
    labels = torch.tensor([1, 2], device="cuda")
    t = torch.tensor([0.25, 0.5], device="cuda")
    dx = torch.randn(2, 4, 8, 8, device="cuda")
    dt = torch.tensor([0.01, 0.02], device="cuda")
    out, dout = model(latents, labels, t, dx, dt)
    assert out.shape == (2,)
    assert dout.shape == (2,)
