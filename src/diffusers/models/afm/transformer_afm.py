"""Hub-ready AFM generator for Diffusers checkpoints."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

_LOCAL_SRC = Path(__file__).resolve().parents[2]


def _bootstrap_upstream():
    stashed = {}
    for name in list(sys.modules):
        if name == "diffusers" or name.startswith("diffusers."):
            stashed[name] = sys.modules.pop(name)
    original_path = sys.path[:]
    try:
        sys.path = [entry for entry in sys.path if Path(entry).resolve() != _LOCAL_SRC.resolve()]
        cfg = importlib.import_module("diffusers.configuration_utils")
        mdl = importlib.import_module("diffusers.models.modeling_utils")
        return cfg.ConfigMixin, cfg.register_to_config, mdl.ModelMixin
    finally:
        sys.path = original_path
        sys.modules.update(stashed)


ConfigMixin, register_to_config, ModelMixin = _bootstrap_upstream()

from .generator import Generator
from .generator_deep import GeneratorDeep


AFM_ARCH_PRESETS: Dict[str, Dict[str, int]] = {
    "AFM-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    "AFM-M/2": {"depth": 16, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "AFM-L/2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "AFM-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
}


class AFMGenerator2DModel(ModelMixin, ConfigMixin):
    """Diffusers wrapper around AFM Generator / GeneratorDeep backbones."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        model_type: str = "AFM-XL/2",
        architecture: str = "standard",
        repeat: int = 1,
        use_t_src: bool = False,
        use_t_tgt: bool = False,
        depth: int = 28,
        hidden_size: int = 1152,
        patch_size: int = 2,
        num_heads: int = 16,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.0,
        input_size: int = 32,
        num_classes: int = 1000,
        in_channels: int = 4,
        pred_type: str = "x",
        num_inference_steps: int = 1,
    ) -> None:
        super().__init__()
        backbone_kwargs = dict(
            depth=depth,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_heads=num_heads,
            learn_sigma=learn_sigma,
            class_dropout_prob=class_dropout_prob,
            input_size=input_size,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        if architecture == "deep":
            self.backbone = GeneratorDeep(repeat=repeat, **backbone_kwargs)
        else:
            self.backbone = Generator(
                use_t_src=use_t_src,
                use_t_tgt=use_t_tgt,
                **backbone_kwargs,
            )

    @property
    def in_channels(self) -> int:
        return int(self.backbone.in_channels)

    @property
    def transformer(self):
        return self.backbone.transformer

    def forward(self, x, y, t_src=None, t_tgt=None):
        return self.backbone(x, y, t_src, t_tgt)

    def load_state_dict(self, state_dict, strict: bool = True):
        if state_dict and any(key.startswith("backbone.") for key in state_dict):
            return super().load_state_dict(state_dict, strict=strict)
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.backbone.state_dict(*args, **kwargs)

    @classmethod
    def from_legacy_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        model_type: str,
        architecture: str = "standard",
        repeat: int = 1,
        use_t_src: bool = False,
        use_t_tgt: bool = False,
        pred_type: str = "x",
        num_inference_steps: int = 1,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Tuple["AFMGenerator2DModel", Dict[str, object]]:
        if model_type not in AFM_ARCH_PRESETS:
            raise ValueError(f"Unknown AFM preset '{model_type}'. Known: {list(AFM_ARCH_PRESETS)}")
        config = dict(AFM_ARCH_PRESETS[model_type])
        model = cls(
            model_type=model_type,
            architecture=architecture,
            repeat=repeat,
            use_t_src=use_t_src,
            use_t_tgt=use_t_tgt,
            pred_type=pred_type,
            num_inference_steps=num_inference_steps,
            **config,
        )
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        model.load_state_dict(state_dict, strict=strict)
        metadata = {
            "checkpoint_path": checkpoint_path,
            "model_type": model_type,
            "architecture": architecture,
            "repeat": repeat,
            "use_t_src": use_t_src,
            "use_t_tgt": use_t_tgt,
            "pred_type": pred_type,
            "num_inference_steps": num_inference_steps,
        }
        return model, metadata


__all__ = ["AFMGenerator2DModel", "AFM_ARCH_PRESETS"]
