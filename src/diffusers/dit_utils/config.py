from typing import Any, Dict


DIT_MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "DiT-XL/2": {"num_layers": 28, "num_attention_heads": 16, "attention_head_dim": 72, "patch_size": 2},
    "DiT-XL/4": {"num_layers": 28, "num_attention_heads": 16, "attention_head_dim": 72, "patch_size": 4},
    "DiT-XL/8": {"num_layers": 28, "num_attention_heads": 16, "attention_head_dim": 72, "patch_size": 8},
    "DiT-L/2": {"num_layers": 24, "num_attention_heads": 16, "attention_head_dim": 64, "patch_size": 2},
    "DiT-L/4": {"num_layers": 24, "num_attention_heads": 16, "attention_head_dim": 64, "patch_size": 4},
    "DiT-L/8": {"num_layers": 24, "num_attention_heads": 16, "attention_head_dim": 64, "patch_size": 8},
    "DiT-B/2": {"num_layers": 12, "num_attention_heads": 12, "attention_head_dim": 64, "patch_size": 2},
    "DiT-B/4": {"num_layers": 12, "num_attention_heads": 12, "attention_head_dim": 64, "patch_size": 4},
    "DiT-B/8": {"num_layers": 12, "num_attention_heads": 12, "attention_head_dim": 64, "patch_size": 8},
    "DiT-S/2": {"num_layers": 12, "num_attention_heads": 6, "attention_head_dim": 64, "patch_size": 2},
    "DiT-S/4": {"num_layers": 12, "num_attention_heads": 6, "attention_head_dim": 64, "patch_size": 4},
    "DiT-S/8": {"num_layers": 12, "num_attention_heads": 6, "attention_head_dim": 64, "patch_size": 8},
}


def get_transformer_config(model_name: str, image_size: int, num_classes: int = 1000) -> Dict[str, Any]:
    if model_name not in DIT_MODEL_PRESETS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {sorted(DIT_MODEL_PRESETS)}")
    preset = DIT_MODEL_PRESETS[model_name]
    latent_size = image_size // 8
    return {
        "sample_size": latent_size,
        "num_layers": preset["num_layers"],
        "num_attention_heads": preset["num_attention_heads"],
        "attention_head_dim": preset["attention_head_dim"],
        "in_channels": 4,
        "out_channels": 8,
        "patch_size": preset["patch_size"],
        "attention_bias": True,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": num_classes,
        "norm_type": "ada_norm_zero",
        "norm_elementwise_affine": False,
        "dropout": 0.0,
        "norm_num_groups": 32,
        "norm_eps": 1e-5,
        "upcast_attention": False,
    }
