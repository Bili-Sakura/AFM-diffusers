from typing import Optional

import torch

from .._hf import get_hf_attr


def _randn_tensor(*args, **kwargs):
    return get_hf_attr("diffusers.utils.torch_utils.randn_tensor")(*args, **kwargs)


@torch.no_grad()
def sample_dit_latents(
    transformer,
    scheduler,
    class_labels: torch.Tensor,
    latent_size: int,
    in_channels: int = 4,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 250,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    batch_size = class_labels.shape[0]
    latents = _randn_tensor(
        (batch_size, in_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    latent_model_input = torch.cat([latents, latents], dim=0) if guidance_scale > 1.0 else latents
    null_labels = torch.full((batch_size,), transformer.config.num_embeds_ada_norm, device=device, dtype=torch.long)
    class_labels_input = torch.cat([class_labels, null_labels], dim=0) if guidance_scale > 1.0 else class_labels

    scheduler.set_timesteps(num_inference_steps, device=device)
    for timestep in scheduler.timesteps:
        if guidance_scale > 1.0:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=latent_model_input.device, dtype=torch.long)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        timesteps = timesteps.expand(latent_model_input.shape[0])

        noise_pred = transformer(
            latent_model_input,
            timestep=timesteps,
            class_labels=class_labels_input,
        ).sample

        if guidance_scale > 1.0:
            eps, rest = noise_pred[:, :in_channels], noise_pred[:, in_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            noise_pred = torch.cat([eps, rest], dim=1)

        if transformer.config.out_channels // 2 == in_channels:
            model_output, _ = torch.split(noise_pred, in_channels, dim=1)
        else:
            model_output = noise_pred

        latent_model_input = scheduler.step(model_output, timestep, latent_model_input).prev_sample

    if guidance_scale > 1.0:
        latents, _ = latent_model_input.chunk(2, dim=0)
    else:
        latents = latent_model_input
    return latents
