from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from ..._hf import get_hf_attr
from ...models.cafm.sit.generator import Generator
from .._labels import (
    build_label2id,
    get_label_ids,
    normalize_class_labels,
    normalize_id2label,
    read_id2label_from_model_index,
)
from diffusers.schedulers.scheduling_continuous_flow import ContinuousFlowMatchScheduler

DiffusionPipeline = get_hf_attr("diffusers.pipelines.pipeline_utils.DiffusionPipeline")
ImagePipelineOutput = get_hf_attr("diffusers.pipelines.pipeline_utils.ImagePipelineOutput")
VaeImageProcessor = get_hf_attr("diffusers.image_processor.VaeImageProcessor")
randn_tensor = get_hf_attr("diffusers.utils.torch_utils.randn_tensor")


class CAFMSiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for continuous adversarial flow model (CAFM) sampling with a SiT generator.
    """

    model_cpu_offload_seq = "generator->vae"

    def __init__(
        self,
        generator: Generator,
        vae,
        scheduler: Optional[ContinuousFlowMatchScheduler] = None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(
            generator=generator,
            vae=vae,
            scheduler=scheduler or ContinuousFlowMatchScheduler(),
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self._id2label = normalize_id2label(id2label)
        self.labels = build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        return get_label_ids(label, self.labels)

    def _default_image_size(self) -> int:
        return int(self.generator.config.input_size) * self.vae_scale_factor

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        return randn_tensor(
            (batch_size, self.generator.config.in_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return latents
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        image = self.vae.decode(latents / scaling_factor).sample
        if output_type == "pt":
            return image
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 250,
        sampler: str = "heun",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        default_size = self._default_image_size()
        height = int(height or default_size)
        width = int(width or default_size)
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")
        if sampler not in {"euler", "heun"}:
            raise ValueError("sampler must be one of: 'euler', 'heun'.")

        device = getattr(self, "_execution_device", None) or next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype
        class_labels_tensor = normalize_class_labels(
            class_labels,
            device=device,
            label2id=self.labels,
        )
        batch_size = class_labels_tensor.shape[0]

        latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device, solver=sampler)
        timesteps = self.scheduler.timesteps

        self.generator.eval()
        for t_src, t_tgt in self.progress_bar(list(zip(timesteps[:-1], timesteps[1:]))):
            t_src_batch = t_src.expand(batch_size)
            t_tgt_batch = t_tgt.expand(batch_size)
            outputs = self.generator(latents, class_labels_tensor, t_src_batch)
            if sampler == "heun":
                latents_next = self.scheduler.step(
                    outputs,
                    t_src,
                    t_tgt,
                    latents,
                    prediction_type="v",
                ).prev_sample
                outputs_next = self.generator(latents_next, class_labels_tensor, t_tgt_batch)
                latents = self.scheduler.step(
                    outputs,
                    t_src,
                    t_tgt,
                    latents,
                    model_output_next=outputs_next,
                    prediction_type="v",
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    outputs,
                    t_src,
                    t_tgt,
                    latents,
                    prediction_type="v",
                ).prev_sample

        image = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
