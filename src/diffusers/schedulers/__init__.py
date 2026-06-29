from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .scheduling_continuous_flow import ContinuousFlowMatchScheduler

__all__ = ["ContinuousFlowMatchScheduler", "DDIMScheduler", "DDPMScheduler"]


def __getattr__(name: str):
    if name == "DDIMScheduler":
        import importlib
        import sys
        from pathlib import Path

        local_src = Path(__file__).resolve().parents[2]
        stashed = {}
        for mod_name in list(sys.modules):
            if not (mod_name == "diffusers" or mod_name.startswith("diffusers.")):
                continue
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            mod_file = getattr(mod, "__file__", "") or ""
            mod_paths = getattr(mod, "__path__", None)
            is_local = f"{local_src / 'diffusers'}" in mod_file.replace("\\", "/")
            if mod_paths is not None:
                is_local = is_local or any(f"{local_src / 'diffusers'}" in str(path) for path in mod_paths)
            if is_local:
                stashed[mod_name] = sys.modules.pop(mod_name)
        original_path = sys.path[:]
        try:
            sys.path = [entry for entry in sys.path if Path(entry).resolve() != local_src.resolve()]
            module = importlib.import_module("diffusers.schedulers.scheduling_ddim")
            return module.DDIMScheduler
        finally:
            sys.path = original_path
            sys.modules.update(stashed)
    if name == "DDPMScheduler":
        import importlib
        import sys
        from pathlib import Path

        local_src = Path(__file__).resolve().parents[2]
        stashed = {}
        for mod_name in list(sys.modules):
            if not (mod_name == "diffusers" or mod_name.startswith("diffusers.")):
                continue
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            mod_file = getattr(mod, "__file__", "") or ""
            mod_paths = getattr(mod, "__path__", None)
            is_local = f"{local_src / 'diffusers'}" in mod_file.replace("\\", "/")
            if mod_paths is not None:
                is_local = is_local or any(f"{local_src / 'diffusers'}" in str(path) for path in mod_paths)
            if is_local:
                stashed[mod_name] = sys.modules.pop(mod_name)
        original_path = sys.path[:]
        try:
            sys.path = [entry for entry in sys.path if Path(entry).resolve() != local_src.resolve()]
            module = importlib.import_module("diffusers.schedulers.scheduling_ddpm")
            return module.DDPMScheduler
        finally:
            sys.path = original_path
            sys.modules.update(stashed)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
