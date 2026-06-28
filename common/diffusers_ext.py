"""Register AFM/CAFM model extensions with the upstream diffusers package."""

from __future__ import annotations

import sys
from pathlib import Path


def register_diffusers_extensions() -> None:
    src_diffusers = Path(__file__).resolve().parents[1] / "src" / "diffusers"
    src_root = str(src_diffusers.parent)
    if src_root not in sys.path:
        sys.path.append(src_root)

    import diffusers

    extension_path = str(src_diffusers)
    if extension_path not in diffusers.__path__:
        diffusers.__path__.append(extension_path)

    import diffusers.models

    models_extension = str(src_diffusers / "models")
    if models_extension not in diffusers.models.__path__:
        diffusers.models.__path__.append(models_extension)

    import diffusers.models.autoencoders
    import diffusers.models.transformers

    autoencoders_extension = str(src_diffusers / "models" / "autoencoders")
    transformers_extension = str(src_diffusers / "models" / "transformers")

    if autoencoders_extension not in diffusers.models.autoencoders.__path__:
        diffusers.models.autoencoders.__path__.append(autoencoders_extension)
    if transformers_extension not in diffusers.models.transformers.__path__:
        diffusers.models.transformers.__path__.append(transformers_extension)


register_diffusers_extensions()
