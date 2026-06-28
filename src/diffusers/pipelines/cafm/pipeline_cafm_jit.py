from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from ..._hf import get_hf_attr
from ...models.cafm.jit.generator import Generator
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
randn_tensor = get_hf_attr("diffusers.utils.torch_utils.randn_tensor")


class CAFMJiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for continuous adversarial flow model (CAFM) sampling with a JiT generator.

    JiT operates in pixel space, so no VAE is required.
    """

    model_cpu_offload_seq = "generator"

    def __init__(
        self,
        generator: Generator,
        scheduler: Optional[ContinuousFlowMatchScheduler] = None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(
            generator=generator,
            scheduler=scheduler or ContinuousFlowMatchScheduler(),
        )
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
        return int(self.generator.sample_size)

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        return (
            randn_tensor(
                (batch_size, self.generator.in_channels, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            * noise_scale
        )

    def postprocess_pixels(self, pixels: torch.Tensor, output_type: str = "pil"):
        images_pt = ((pixels.float().clamp(-1, 1) + 1.0) / 2.0).cpu()
        if output_type == "pt":
            return images_pt
        if output_type == "np":
            return images_pt.permute(0, 2, 3, 1).numpy()
        return self.numpy_to_pil(images_pt.permute(0, 2, 3, 1).numpy())

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sampler: str = "heun",
        noise_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        default_size = self._default_image_size()
        height = int(height or default_size)
        width = int(width or default_size)
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt'.")
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
            noise_scale=noise_scale,
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

        image = self.postprocess_pixels(latents, output_type=output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
