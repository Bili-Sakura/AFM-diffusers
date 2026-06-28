import copy
from typing import Dict

import torch

from .config import DIT_MODEL_PRESETS


def convert_original_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_name: str,
) -> Dict[str, torch.Tensor]:
    """
    Map a legacy facebookresearch/DiT checkpoint into Diffusers DiTTransformer2DModel keys.
    """
    if model_name not in DIT_MODEL_PRESETS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {sorted(DIT_MODEL_PRESETS)}")
    depth = DIT_MODEL_PRESETS[model_name]["num_layers"]
    converted = copy.deepcopy(state_dict)

    converted["pos_embed.proj.weight"] = converted.pop("x_embedder.proj.weight").clone().contiguous()
    converted["pos_embed.proj.bias"] = converted.pop("x_embedder.proj.bias").clone().contiguous()

    timestep_weights = {
        "linear_1.weight": converted.pop("t_embedder.mlp.0.weight"),
        "linear_1.bias": converted.pop("t_embedder.mlp.0.bias"),
        "linear_2.weight": converted.pop("t_embedder.mlp.2.weight"),
        "linear_2.bias": converted.pop("t_embedder.mlp.2.bias"),
    }
    class_embedding = converted.pop("y_embedder.embedding_table.weight")

    for block_idx in range(depth):
        for key, tensor in timestep_weights.items():
            converted[f"transformer_blocks.{block_idx}.norm1.emb.timestep_embedder.{key}"] = tensor.clone()
        converted[f"transformer_blocks.{block_idx}.norm1.emb.class_embedder.embedding_table.weight"] = (
            class_embedding.clone()
        )

        converted[f"transformer_blocks.{block_idx}.norm1.linear.weight"] = converted[
            f"blocks.{block_idx}.adaLN_modulation.1.weight"
        ]
        converted[f"transformer_blocks.{block_idx}.norm1.linear.bias"] = converted[
            f"blocks.{block_idx}.adaLN_modulation.1.bias"
        ]

        q, k, v = torch.chunk(converted[f"blocks.{block_idx}.attn.qkv.weight"], 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(converted[f"blocks.{block_idx}.attn.qkv.bias"], 3, dim=0)
        converted[f"transformer_blocks.{block_idx}.attn1.to_q.weight"] = q
        converted[f"transformer_blocks.{block_idx}.attn1.to_q.bias"] = q_bias
        converted[f"transformer_blocks.{block_idx}.attn1.to_k.weight"] = k
        converted[f"transformer_blocks.{block_idx}.attn1.to_k.bias"] = k_bias
        converted[f"transformer_blocks.{block_idx}.attn1.to_v.weight"] = v
        converted[f"transformer_blocks.{block_idx}.attn1.to_v.bias"] = v_bias
        converted[f"transformer_blocks.{block_idx}.attn1.to_out.0.weight"] = converted[
            f"blocks.{block_idx}.attn.proj.weight"
        ]
        converted[f"transformer_blocks.{block_idx}.attn1.to_out.0.bias"] = converted[
            f"blocks.{block_idx}.attn.proj.bias"
        ]
        converted[f"transformer_blocks.{block_idx}.ff.net.0.proj.weight"] = converted[
            f"blocks.{block_idx}.mlp.fc1.weight"
        ]
        converted[f"transformer_blocks.{block_idx}.ff.net.0.proj.bias"] = converted[
            f"blocks.{block_idx}.mlp.fc1.bias"
        ]
        converted[f"transformer_blocks.{block_idx}.ff.net.2.weight"] = converted[
            f"blocks.{block_idx}.mlp.fc2.weight"
        ]
        converted[f"transformer_blocks.{block_idx}.ff.net.2.bias"] = converted[
            f"blocks.{block_idx}.mlp.fc2.bias"
        ]

        for suffix in (
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
            "adaLN_modulation.1.weight",
            "adaLN_modulation.1.bias",
        ):
            converted.pop(f"blocks.{block_idx}.{suffix}", None)

    converted["proj_out_1.weight"] = converted.pop("final_layer.adaLN_modulation.1.weight")
    converted["proj_out_1.bias"] = converted.pop("final_layer.adaLN_modulation.1.bias")
    converted["proj_out_2.weight"] = converted.pop("final_layer.linear.weight")
    converted["proj_out_2.bias"] = converted.pop("final_layer.linear.bias")

    converted.pop("pos_embed", None)
    for block_idx in range(depth):
        for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            converted.pop(f"blocks.{block_idx}.{suffix}", None)

    return {key: tensor.detach().clone().contiguous() for key, tensor in converted.items()}
