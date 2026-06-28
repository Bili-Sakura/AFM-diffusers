import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

torch = pytest.importorskip("torch")

from diffusers.models.afm.generator import Generator as AFMGenerator
from diffusers.models.afm.discriminator import Discriminator as AFMDiscriminator
from diffusers.models.cafm.sit.generator import Generator as SiTGenerator
from diffusers.models.cafm.jit.generator import Generator as JiTGenerator
from diffusers.models.transformers.transformer_sit import SiTTransformer2DModel
from diffusers.models.transformers.transformer_jit import JiTTransformer2DModel


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
    output = model(latents, torch.tensor([0.25, 0.5]), torch.tensor([1, 2])).sample
    assert output.shape == latents.shape


def test_cafm_sit_generator_forward():
    model = SiTGenerator(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    )
    x = torch.randn(2, 4, 8, 8)
    y = torch.tensor([1, 2])
    t = torch.tensor([0.5, 0.25])
    out = model(x, y, t)
    assert out.shape == x.shape


def test_cafm_jit_generator_forward():
    model = JiTGenerator(
        sample_size=64,
        patch_size=16,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        bottleneck_dim=32,
        in_context_len=4,
        in_context_start=1,
        num_classes=10,
    )
    x = torch.randn(2, 3, 64, 64)
    y = torch.tensor([1, 2])
    t = torch.tensor([0.5, 0.25])
    out = model(x, y, t)
    assert out.shape == x.shape


def test_afm_generator_forward():
    model = AFMGenerator(
        depth=2,
        hidden_size=32,
        patch_size=2,
        num_heads=4,
        learn_sigma=False,
        input_size=8,
        num_classes=10,
        use_t_src=True,
    )
    x = torch.randn(2, 4, 8, 8)
    y = torch.tensor([1, 2])
    t = torch.tensor([0.5, 0.25])
    out = model(x, y, t_src=t, t_tgt=None)
    assert out.shape == x.shape


def test_afm_discriminator_forward():
    model = AFMDiscriminator(
        depth=2,
        hidden_size=32,
        patch_size=2,
        num_heads=4,
        learn_sigma=False,
        input_size=8,
        num_classes=10,
        use_t=True,
    )
    x = torch.randn(2, 4, 8, 8)
    y = torch.tensor([1, 2])
    t = torch.tensor([0.5, 0.25])
    out = model(x, y, t)
    assert out.shape == (2,)
