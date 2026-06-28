import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

torch = pytest.importorskip("torch")

from diffusers.models.afm.generator import Generator as AFMGenerator
from diffusers.models.afm.vae import AutoencoderKLWrapper
from diffusers.models.cafm.jit.generator import Generator as CAFMJiTGenerator
from diffusers.models.cafm.sit.generator import Generator as CAFMSiTGenerator
from diffusers.pipelines.afm.pipeline_afm import AFMPipeline
from diffusers.pipelines.cafm.pipeline_cafm_jit import CAFMJiTPipeline
from diffusers.pipelines.cafm.pipeline_cafm_sit import CAFMSiTPipeline
from diffusers.schedulers.scheduling_continuous_flow import ContinuousFlowMatchScheduler


class _DummyVAE(torch.nn.Module):
    class Config:
        block_out_channels = [128, 256, 512, 512]
        scaling_factor = 0.18215

    config = Config()

    def decode(self, latents):
        return type("Decoded", (), {"sample": latents})()


def test_continuous_flow_scheduler_euler_step():
    scheduler = ContinuousFlowMatchScheduler(solver="euler")
    scheduler.set_timesteps(4)
    sample = torch.ones(1, 4, 2, 2)
    velocity = torch.full_like(sample, 0.5)
    output = scheduler.step(velocity, 1.0, 0.75, sample, prediction_type="v").prev_sample
    assert output.shape == sample.shape
    assert torch.allclose(output, sample - 0.25 * velocity)


def test_afm_pipeline_latent_output():
    generator = AFMGenerator(
        depth=2,
        hidden_size=32,
        patch_size=2,
        num_heads=4,
        learn_sigma=False,
        input_size=8,
        num_classes=10,
    )
    pipe = AFMPipeline(
        generator=generator,
        vae=_DummyVAE(),
        scheduler=ContinuousFlowMatchScheduler(solver="euler"),
        pred_type="x",
    )
    result = pipe(
        class_labels=torch.tensor([1]),
        height=64,
        width=64,
        num_inference_steps=1,
        output_type="latent",
    )
    assert result.images.shape == (1, 4, 8, 8)


def test_cafm_sit_pipeline_latent_output():
    generator = CAFMSiTGenerator(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    )
    pipe = CAFMSiTPipeline(
        generator=generator,
        vae=_DummyVAE(),
        scheduler=ContinuousFlowMatchScheduler(solver="euler"),
    )
    result = pipe(
        class_labels=torch.tensor([1]),
        height=64,
        width=64,
        num_inference_steps=2,
        sampler="euler",
        output_type="latent",
    )
    assert result.images.shape == (1, 4, 8, 8)


def test_cafm_jit_pipeline_pixel_output():
    generator = CAFMJiTGenerator(
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
    pipe = CAFMJiTPipeline(
        generator=generator,
        scheduler=ContinuousFlowMatchScheduler(solver="euler"),
    )
    result = pipe(
        class_labels=torch.tensor([1]),
        height=64,
        width=64,
        num_inference_steps=2,
        sampler="euler",
        output_type="pt",
    )
    assert result.images.shape == (1, 3, 64, 64)
