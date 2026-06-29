import importlib
import sys
from pathlib import Path
from typing import NamedTuple

import torch

_LOCAL_SRC = Path(__file__).resolve().parents[3]


def _load_upstream_attr(module_path: str, attr_name: str):
    stashed = {}
    for name in list(sys.modules):
        if not (name == "diffusers" or name.startswith("diffusers.")):
            continue
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        mod_paths = getattr(mod, "__path__", None)
        is_local = f"{_LOCAL_SRC / 'diffusers'}" in mod_file.replace("\\", "/")
        if mod_paths is not None:
            is_local = is_local or any(f"{_LOCAL_SRC / 'diffusers'}" in str(path) for path in mod_paths)
        if is_local:
            stashed[name] = sys.modules.pop(name)
    original_path = sys.path[:]
    try:
        sys.path = [entry for entry in sys.path if Path(entry).resolve() != _LOCAL_SRC.resolve()]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    finally:
        sys.path = original_path
        sys.modules.update(stashed)


AutoencoderKL = _load_upstream_attr("diffusers", "AutoencoderKL")
DiagonalGaussianDistribution = _load_upstream_attr("diffusers.models.autoencoders.vae", "DiagonalGaussianDistribution")
apply_forward_hook = _load_upstream_attr("diffusers.utils.accelerate_utils", "apply_forward_hook")


class AutoencoderOutput(NamedTuple):
    sample: torch.Tensor
    posterior: DiagonalGaussianDistribution


class EncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: DiagonalGaussianDistribution


class DecoderOutput(NamedTuple):
    sample: torch.Tensor


class AutoencoderKLWrapper(AutoencoderKL):
    def load_state_dict(self, state_dict, *args, **kwargs):
        self._convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, sample_posterior: bool = True) -> EncoderOutput:
        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
        ):
            return self.tiled_encode(x, return_dict=False).latent_dist

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()
        return EncoderOutput(latent=latent, posterior=posterior)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor) -> DecoderOutput:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        x: torch.FloatTensor,
        sample_posterior: bool = False,
    ) -> AutoencoderOutput:
        latent, posterior = self.encode(x, sample_posterior=sample_posterior)
        sample = self.decode(latent).sample
        return AutoencoderOutput(sample=sample, posterior=posterior)
