"""Hub custom pipeline: CAFMZImagePipeline.

Continuous adversarial flow sampling for Tongyi-MAI/Z-Image T2I checkpoints.
Load with native Hugging Face diffusers and trust_remote_code=True.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import AutoTokenizer, PreTrainedModel


class CAFMZImagePipelineOutput:
    def __init__(self, images):
        self.images = images


class CAFMZImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["scheduler"]

    def __init__(
        self,
        transformer,
        vae,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        scheduler=None,
        timestep_shift: float = 1.0 / 3.0,
        default_sampler: str = "euler",
    ) -> None:
        super().__init__()
        modules = {
            "transformer": transformer,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        if scheduler is not None and not isinstance(scheduler, (list, tuple)):
            modules["scheduler"] = scheduler
        self.register_modules(**modules)
        self.register_to_config(timestep_shift=float(timestep_shift))
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self.vae, "config") else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    @staticmethod
    def shift_timesteps(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
        return (timesteps * shift) / (1 + (shift - 1) * timesteps)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        max_sequence_length: int = 512,
    ) -> List[torch.Tensor]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        formatted = []
        for prompt_item in prompts:
            messages = [{"role": "user", "content": prompt_item}]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )
        text_inputs = self.tokenizer(
            formatted,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()
        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]
        return [prompt_embeds[i][prompt_masks[i]] for i in range(len(prompt_embeds))]

    def _latent_hw(self, height: int, width: int) -> tuple[int, int]:
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        return height, width

    def prepare_noises(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> List[torch.Tensor]:
        latent_height, latent_width = self._latent_hw(height, width)
        in_channels = int(self.transformer.config.in_channels)
        noises = []
        for index in range(batch_size):
            gen = generator[index] if isinstance(generator, list) else generator
            noise = randn_tensor(
                (in_channels, 1, latent_height, latent_width),
                generator=gen,
                device=device,
                dtype=dtype,
            )
            noises.append(noise)
        return noises

    def decode_latents(self, latents: List[torch.Tensor], output_type: str = "pil"):
        scale = self.vae.config.scaling_factor
        shift = self.vae.config.shift_factor
        device = getattr(self, "_execution_device", next(self.vae.parameters()).device)
        vae_dtype = self.vae.dtype
        images = []
        for latent in latents:
            latent = rearrange(latent, "c f h w -> f c h w").to(device=device, dtype=vae_dtype)
            latent = latent / scale + shift
            sample = self.vae.decode(latent).sample
            images.append(sample.squeeze(0).float())
        stacked = torch.stack(images, dim=0)
        if output_type == "pt":
            return stacked
        if output_type == "np":
            return self.image_processor.postprocess(stacked, output_type="np")
        return self.image_processor.postprocess(stacked, output_type="pil")

    def _run_denoising_loop(
        self,
        latents: List[torch.Tensor],
        text_embeds: List[torch.Tensor],
        num_inference_steps: int,
        sampler: str,
    ) -> List[torch.Tensor]:
        device = latents[0].device
        dtype = latents[0].dtype
        timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)
        timesteps = self.shift_timesteps(timesteps, self.config.timestep_shift)

        self.transformer.eval()
        for t_src, t_tgt in self.progress_bar(list(zip(timesteps[:-1], timesteps[1:]))):
            dt = t_tgt - t_src
            t_src_batch = t_src.expand(len(latents)).to(dtype=dtype)
            outputs = self.transformer(latents, t_src_batch, text_embeds, return_dict=True).sample
            if sampler == "heun":
                latents_next = [x_t + dt * velocity for x_t, velocity in zip(latents, outputs)]
                t_tgt_batch = t_tgt.expand(len(latents)).to(dtype=dtype)
                outputs_next = self.transformer(latents_next, t_tgt_batch, text_embeds, return_dict=True).sample
                latents = [
                    x_t + dt * 0.5 * (velocity + velocity_next)
                    for x_t, velocity, velocity_next in zip(latents, outputs, outputs_next)
                ]
            else:
                latents = [x_t + dt * velocity for x_t, velocity in zip(latents, outputs)]
        return latents

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        sampler: str = "euler",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_sequence_length: int = 512,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[CAFMZImagePipelineOutput, Tuple]:
        if sampler not in {"euler", "heun"}:
            raise ValueError("sampler must be one of: 'euler', 'heun'.")

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(f"height and width must be divisible by {vae_scale}, got ({height}, {width}).")

        device = getattr(self, "_execution_device", None) or next(self.transformer.parameters()).device
        dtype = getattr(self.transformer, "dtype", torch.float32)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        text_embeds = self.encode_prompt(prompts, device=device, max_sequence_length=max_sequence_length)
        latents = self.prepare_noises(
            batch_size=len(prompts),
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        latents = self._run_denoising_loop(latents, text_embeds, num_inference_steps, sampler)
        images = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return CAFMZImagePipelineOutput(images=images)
