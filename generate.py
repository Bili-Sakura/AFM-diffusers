"""
Generation entrypoint for AFM and CAFM models using native diffusers pipelines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import datetime
import os

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from common.config import create_object, load_config
from common.decorators import barrier_on_entry, log_on_entry
from common.distributed import (
    get_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    init_torch,
)
from common.entrypoint import Entrypoint
from common.fs import download, mkdir
from common.partition import partition_by_groups
from common.seed import set_seed
from diffusers.pipelines.afm.pipeline_afm import AFMPipeline
from diffusers.pipelines.cafm.pipeline_cafm_jit import CAFMJiTPipeline
from diffusers.pipelines.cafm.pipeline_cafm_sit import CAFMSiTPipeline
from diffusers.schedulers.scheduling_continuous_flow import ContinuousFlowMatchScheduler


class AdversarialFlowGenerator(Entrypoint):
    def entrypoint(self):
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        self.configure_seed()
        self.configure_pipeline()
        self.configure_seed()
        self.inference_loop()

    def configure_seed(self):
        set_seed(self.config.generation.seed)

    @log_on_entry
    def configure_pipeline(self, device=get_device()):
        self.gen = create_object(self.config.gen.model).to(device)
        checkpoint = self.config.gen.get("checkpoint", None)
        if checkpoint:
            state = torch.load(download(checkpoint), map_location=device)
            self.gen.load_state_dict(state, strict=self.config.gen.get("strict", True))

        object_path = self.config.gen.model.__object__.path
        scheduler = ContinuousFlowMatchScheduler(
            solver=self.config.generation.get("sampler", "euler"),
        )

        if self.config.get("vae"):
            dtype = getattr(torch, self.config.vae.dtype)
            vae = create_object(self.config.vae.model)
            vae.requires_grad_(False).eval().to(device=device, dtype=dtype)
            if self.config.vae.get("checkpoint"):
                state = torch.load(download(self.config.vae.checkpoint), map_location=device)
                vae.load_state_dict(state, strict=True)
            if self.config.vae.compile:
                vae.encode = torch.compile(vae.encode)
                vae.decode = torch.compile(vae.decode)

            if "cafm.sit" in object_path:
                self.pipeline = CAFMSiTPipeline(
                    generator=self.gen,
                    vae=vae,
                    scheduler=scheduler,
                ).to(device)
                latent_size = int(getattr(self.gen.config, "input_size", 32))
                self.latent_shape = (int(self.gen.config.in_channels), latent_size, latent_size)
            else:
                self.pipeline = AFMPipeline(
                    generator=self.gen,
                    vae=vae,
                    scheduler=scheduler,
                    pred_type=self.config.gen.get("pred_type", "x"),
                ).to(device)
                latent_size = int(getattr(self.gen.transformer.config, "sample_size", 32))
                in_channels = int(getattr(self.gen, "in_channels", 4))
                self.latent_shape = (in_channels, latent_size, latent_size)
        else:
            self.pipeline = CAFMJiTPipeline(
                generator=self.gen,
                scheduler=scheduler,
            ).to(device)
            resolution = int(self.config.generation.get("resolution", getattr(self.gen, "sample_size", 256)))
            self.latent_shape = (3, resolution, resolution)

        self.pipeline.set_progress_bar_config(disable=get_local_rank() != 0)

    @barrier_on_entry
    @torch.no_grad()
    def inference_loop(self):
        output = self.config.generation.output
        mkdir(output)

        labels = torch.arange(0, 1000).repeat_interleave(50).tolist()
        labels = list(enumerate(labels))
        labels = partition_by_groups(labels, get_world_size())[get_global_rank()]

        device = get_device()
        steps = self.config.generation.steps
        sampler = self.config.generation.get("sampler", "euler")

        for i, label in tqdm(labels, position=get_local_rank()):
            generator = torch.Generator(device=device).manual_seed(i + self.config.generation.seed)
            class_labels = torch.tensor([label], device=device, dtype=torch.long)
            noises = torch.randn(
                [1, *self.latent_shape],
                device=device,
                generator=generator,
            )

            if isinstance(self.pipeline, CAFMJiTPipeline):
                result = self.pipeline(
                    class_labels=class_labels,
                    num_inference_steps=steps,
                    sampler=sampler,
                    latents=noises,
                    output_type="pt",
                    generator=generator,
                )
                sample = result.images[0]
            else:
                result = self.pipeline(
                    class_labels=class_labels,
                    num_inference_steps=steps,
                    sampler=sampler,
                    latents=noises,
                    output_type="pt",
                    generator=generator,
                )
                sample = result.images[0]

            to_pil_image(sample.mul(0.5).add(0.5).clamp(0, 1)).save(os.path.join(output, f"{i:05}.png"))


def main():
    from sys import argv

    config = load_config(argv[1], argv[2:])
    entrypoint = create_object(config)
    assert isinstance(entrypoint, AdversarialFlowGenerator)
    entrypoint.entrypoint()


if __name__ == "__main__":
    main()
