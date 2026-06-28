from pathlib import Path
from typing import Optional

import torch

from .._hf import get_hf_attr, get_hf_diffusers
from .config import DIT_MODEL_PRESETS, get_transformer_config
from .conversion import convert_original_state_dict


def _load_legacy_state_dict(checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("ema", "model", "state_dict", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_dit_pipeline(
    model_name: str,
    image_size: int,
    vae: str = "mse",
    checkpoint_path: Optional[str] = None,
    diffusers_dir: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float32,
):
    DiTPipeline = get_hf_attr("diffusers.pipelines.dit.pipeline_dit.DiTPipeline")
    DiTTransformer2DModel = get_hf_attr("diffusers.models.transformers.dit_transformer_2d.DiTTransformer2DModel")
    DDIMScheduler = get_hf_attr("diffusers.schedulers.DDIMScheduler")
    AutoencoderKL = get_hf_attr("diffusers.models.autoencoder_kl.AutoencoderKL")

    vae_id = f"stabilityai/sd-vae-ft-{vae}"

    if diffusers_dir is not None:
        return DiTPipeline.from_pretrained(diffusers_dir, torch_dtype=torch_dtype)

    if checkpoint_path is None and model_name == "DiT-XL/2" and image_size in (256, 512):
        return DiTPipeline.from_pretrained(f"facebook/DiT-XL-2-{image_size}", torch_dtype=torch_dtype)

    if checkpoint_path is None:
        raise ValueError(
            "Provide a legacy .pt checkpoint, a converted Diffusers directory, "
            "or use model DiT-XL/2 at 256/512 for automatic Hub download."
        )

    if Path(checkpoint_path).is_dir():
        return DiTPipeline.from_pretrained(checkpoint_path, torch_dtype=torch_dtype)

    if model_name not in DIT_MODEL_PRESETS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {sorted(DIT_MODEL_PRESETS)}")

    state_dict = _load_legacy_state_dict(checkpoint_path)
    converted = convert_original_state_dict(state_dict, model_name)
    transformer = DiTTransformer2DModel(**get_transformer_config(model_name, image_size))
    transformer.load_state_dict(converted, strict=True)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    vae_model = AutoencoderKL.from_pretrained(vae_id)
 
    # Ensure Hub helpers resolve to the PyPI package when saving/loading.
    get_hf_diffusers()
    return DiTPipeline(transformer=transformer, vae=vae_model, scheduler=scheduler)
