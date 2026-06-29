"""Hub custom pipeline: AFMPipeline.

Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class AFMPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "generator->vae"

    @staticmethod
    def _coerce_scheduler(scheduler, generator):
        if scheduler is not None and not isinstance(scheduler, (list, tuple)):
            return scheduler
        variant_path = getattr(generator.config, "_name_or_path", None)
        if variant_path:
            scheduler_dir = Path(variant_path).resolve().parent / "scheduler"
            module_path = scheduler_dir / "scheduling_continuous_flow.py"
            config_path = scheduler_dir / "scheduler_config.json"
            if module_path.is_file() and config_path.is_file():
                spec = importlib.util.spec_from_file_location("scheduling_continuous_flow", module_path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module.ContinuousFlowMatchScheduler.from_pretrained(str(scheduler_dir))
        raise ValueError("AFMPipeline could not load ContinuousFlowMatchScheduler from the variant folder.")

    def __init__(
        self,
        generator,
        vae,
        scheduler=None,
        pred_type: str = "x",
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ) -> None:
        super().__init__()
        scheduler = self._coerce_scheduler(scheduler, generator)
        self.register_modules(generator=generator, vae=vae, scheduler=scheduler)
        self.register_to_config(pred_type=pred_type)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
        if not variant_path:
            return {}
        model_index_path = Path(variant_path).resolve() / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id: Dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def vae_scale_factor(self) -> int:
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels:
            return int(2 ** (len(block_out_channels) - 1))
        return 8

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        labels = [label] if isinstance(label, str) else label
        if not self.labels:
            raise ValueError("No id2label mapping is available in this checkpoint.")
        missing = [item for item in labels if item not in self.labels]
        if missing:
            preview = ", ".join(list(self.labels.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [self.labels[item] for item in labels]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        device: torch.device,
    ) -> torch.LongTensor:
        if torch.is_tensor(class_labels):
            return class_labels.to(device=device, dtype=torch.long).reshape(-1)
        if isinstance(class_labels, int):
            class_label_ids = [class_labels]
        elif isinstance(class_labels, str):
            class_label_ids = self.get_label_ids(class_labels)
        elif class_labels and isinstance(class_labels[0], str):
            class_label_ids = self.get_label_ids(class_labels)
        else:
            class_label_ids = list(class_labels)
        return torch.tensor(class_label_ids, device=device, dtype=torch.long).reshape(-1)

    def _default_image_size(self) -> int:
        transformer = getattr(self.generator, "transformer", None)
        if transformer is not None:
            return int(transformer.config.sample_size) * self.vae_scale_factor
        return 256

    def check_inputs(
        self,
        height: int,
        width: int,
        num_inference_steps: int,
        output_type: str,
        sampler: str,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be >= 1.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")
        if sampler not in {"euler", "heun"}:
            raise ValueError("sampler must be one of: 'euler', 'heun'.")
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor}, got ({height}, {width})."
            )

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
        in_channels = int(getattr(self.generator, "in_channels", 4))
        return randn_tensor(
            (batch_size, in_channels, latent_height, latent_width),
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

    def _run_denoising_loop(
        self,
        latents: torch.Tensor,
        class_labels_tensor: torch.Tensor,
        batch_size: int,
        device: torch.device,
        num_inference_steps: int,
        sampler: str,
        pred_type: str,
    ) -> torch.Tensor:
        self.scheduler.set_timesteps(num_inference_steps, device=device, solver=sampler)
        timesteps = self.scheduler.timesteps
        self.generator.eval()
        for t_src, t_tgt in self.progress_bar(list(zip(timesteps[:-1], timesteps[1:]))):
            t_src_batch = t_src.expand(batch_size)
            t_tgt_batch = t_tgt.expand(batch_size)
            outputs = self.generator(latents, class_labels_tensor, t_src_batch, t_tgt_batch)
            if pred_type == "v":
                if sampler == "heun":
                    latents_next = self.scheduler.step(
                        outputs, t_src, t_tgt, latents, prediction_type="v"
                    ).prev_sample
                    outputs_next = self.generator(latents_next, class_labels_tensor, t_tgt_batch, t_tgt_batch)
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
                        outputs, t_src, t_tgt, latents, prediction_type="v"
                    ).prev_sample
            else:
                latents = outputs
        return latents

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        sampler: str = "euler",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        default_size = self._default_image_size()
        height = int(height or default_size)
        width = int(width or default_size)
        if num_inference_steps is None:
            num_inference_steps = int(getattr(self.generator.config, "num_inference_steps", 1))
        self.check_inputs(height, width, num_inference_steps, output_type, sampler)

        device = getattr(self, "_execution_device", None) or next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype
        pred_type = self.config.pred_type
        class_labels_tensor = self._normalize_class_labels(class_labels, device=device)
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
        latents = self._run_denoising_loop(
            latents,
            class_labels_tensor,
            batch_size,
            device,
            num_inference_steps,
            sampler,
            pred_type,
        )
        image = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
