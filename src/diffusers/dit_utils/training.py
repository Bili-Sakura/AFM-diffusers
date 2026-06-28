import torch
import torch.nn.functional as F

from .._hf import get_hf_attr


def create_training_scheduler(num_train_timesteps: int = 1000):
    DDPMScheduler = get_hf_attr("diffusers.schedulers.DDPMScheduler")
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )


def compute_dit_training_loss(
    model,
    scheduler,
    latents: torch.Tensor,
    class_labels: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(latents)

    batch_size = latents.shape[0]
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch_size,),
        device=latents.device,
        dtype=torch.long,
    )
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    model_output = model(noisy_latents, timestep=timesteps, class_labels=class_labels).sample

    in_channels = model.config.in_channels
    if model.config.out_channels // 2 == in_channels:
        model_output, _ = torch.split(model_output, in_channels, dim=1)

    return F.mse_loss(model_output.float(), noise.float())
