from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .._hf import get_hf_attr

from .scheduling_continuous_flow import ContinuousFlowMatchScheduler
from .scheduling_jit import JiTScheduler

__all__ = ["ContinuousFlowMatchScheduler", "DDIMScheduler", "DDPMScheduler", "JiTScheduler"]


def __getattr__(name: str):
    if name == "DDIMScheduler":
        return get_hf_attr("diffusers.schedulers.DDIMScheduler")
    if name == "DDPMScheduler":
        return get_hf_attr("diffusers.schedulers.DDPMScheduler")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
