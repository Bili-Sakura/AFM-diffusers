__all__ = [
    "AFMPipeline",
    "CAFMJiTPipeline",
    "CAFMSiTPipeline",
    "DiTPipeline",
    "JiTPipeline",
    "JiTPipelineOutput",
    "SiTPipeline",
]


def __getattr__(name: str):
    if name == "AFMPipeline":
        from .afm.pipeline_afm import AFMPipeline

        return AFMPipeline
    if name == "CAFMSiTPipeline":
        from .cafm.pipeline_cafm_sit import CAFMSiTPipeline

        return CAFMSiTPipeline
    if name == "CAFMJiTPipeline":
        from .cafm.pipeline_cafm_jit import CAFMJiTPipeline

        return CAFMJiTPipeline
    if name == "DiTPipeline":
        from .dit.pipeline_dit import DiTPipeline

        return DiTPipeline
    if name == "SiTPipeline":
        from .sit.pipeline_sit import SiTPipeline

        return SiTPipeline
    if name in {"JiTPipeline", "JiTPipelineOutput"}:
        from .jit.pipeline_jit import JiTPipeline, JiTPipelineOutput

        return JiTPipeline if name == "JiTPipeline" else JiTPipelineOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
