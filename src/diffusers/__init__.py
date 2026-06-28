from importlib.metadata import version as _package_version
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

try:
    __version__ = _package_version("diffusers")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "Classifier",
    "DIT_MODEL_PRESETS",
    "DiTPipeline",
    "DiTTransformer2DModel",
    "Discriminator",
    "DiscriminatorJVP",
    "Generator",
    "GeneratorDeep",
    "JiTPipeline",
    "JiTPipelineOutput",
    "JiTScheduler",
    "JiTTransformer2DModel",
    "SiTPipeline",
    "SiTTransformer2DModel",
    "ZImageTransformer2DModelDiscriminatorJVP",
    "compute_dit_training_loss",
    "convert_original_state_dict",
    "create_training_scheduler",
    "get_transformer_config",
    "load_dit_pipeline",
]


def __getattr__(name: str):
    if name in {"SiTTransformer2DModel", "SiTPipeline"}:
        if name == "SiTTransformer2DModel":
            from .models.transformers.transformer_sit import SiTTransformer2DModel

            return SiTTransformer2DModel
        from .pipelines.sit.pipeline_sit import SiTPipeline

        return SiTPipeline
    if name in {"JiTTransformer2DModel", "JiTPipeline", "JiTPipelineOutput", "JiTScheduler"}:
        if name == "JiTTransformer2DModel":
            from .models.transformers.transformer_jit import JiTTransformer2DModel

            return JiTTransformer2DModel
        if name == "JiTScheduler":
            from .schedulers.scheduling_jit import JiTScheduler

            return JiTScheduler
        from .pipelines.jit.pipeline_jit import JiTPipeline, JiTPipelineOutput

        return JiTPipeline if name == "JiTPipeline" else JiTPipelineOutput
    if name in {"DiTTransformer2DModel", "DiTPipeline"}:
        if name == "DiTTransformer2DModel":
            from .models.transformers.transformer_dit import DiTTransformer2DModel

            return DiTTransformer2DModel
        from .pipelines.dit.pipeline_dit import DiTPipeline

        return DiTPipeline
    if name in {"Generator", "GeneratorDeep", "Discriminator", "Classifier"}:
        import importlib

        module_name = "generator_deep" if name == "GeneratorDeep" else name.lower()
        afm = importlib.import_module(f"{__name__}.models.afm.{module_name}")
        return getattr(afm, name)
    if name == "DiscriminatorJVP":
        from .models.cafm.jvp.discriminator import DiscriminatorJVP

        return DiscriminatorJVP
    if name == "ZImageTransformer2DModelDiscriminatorJVP":
        from .models.cafm.zimage.discriminator import ZImageTransformer2DModelDiscriminatorJVP

        return ZImageTransformer2DModelDiscriminatorJVP
    if name in {
        "DIT_MODEL_PRESETS",
        "compute_dit_training_loss",
        "convert_original_state_dict",
        "create_training_scheduler",
        "get_transformer_config",
        "load_dit_pipeline",
    }:
        from .dit_utils import config, conversion, loading, training

        mapping = {
            "DIT_MODEL_PRESETS": config.DIT_MODEL_PRESETS,
            "convert_original_state_dict": conversion.convert_original_state_dict,
            "get_transformer_config": config.get_transformer_config,
            "load_dit_pipeline": loading.load_dit_pipeline,
            "compute_dit_training_loss": training.compute_dit_training_loss,
            "create_training_scheduler": training.create_training_scheduler,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
