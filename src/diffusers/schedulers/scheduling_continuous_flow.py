from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from .._hf import get_hf_attr

ConfigMixin = get_hf_attr("diffusers.configuration_utils.ConfigMixin")
SchedulerMixin = get_hf_attr("diffusers.schedulers.scheduling_utils.SchedulerMixin")
BaseOutput = get_hf_attr("diffusers.utils.BaseOutput")
register_to_config = get_hf_attr("diffusers.configuration_utils.register_to_config")


@dataclass
class ContinuousFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor


class ContinuousFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """Flow-matching scheduler for AFM/CAFM models with time in [1, 0]."""

    order = 2

    @register_to_config
    def __init__(self, solver: str = "euler"):
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be one of: 'euler', 'heun'.")
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        self._step_index: Optional[int] = None

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device, None] = None,
        solver: Optional[str] = None,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        self.num_inference_steps = num_inference_steps
        if solver == "heun":
            grid_size = (num_inference_steps // 2) + 1
        else:
            grid_size = num_inference_steps + 1
        self.timesteps = torch.linspace(1.0, 0.0, grid_size, device=device, dtype=torch.float32)
        self._step_index = 0
        if solver is not None:
            self.register_to_config(solver=solver)

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        del timestep
        return sample

    def _resolve_step_index(self, timestep: Union[float, torch.Tensor, None]) -> int:
        if self._step_index is not None:
            return self._step_index
        if self.timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")
        if timestep is None:
            return 0
        t_value = float(timestep) if not isinstance(timestep, torch.Tensor) else float(timestep.flatten()[0])
        matches = (self.timesteps - t_value).abs() < 1e-6
        if matches.any():
            return int(matches.nonzero(as_tuple=False)[0].item())
        return 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep_src: Union[float, torch.Tensor],
        timestep_tgt: Union[float, torch.Tensor],
        sample: torch.Tensor,
        model_output_next: Optional[torch.Tensor] = None,
        prediction_type: str = "v",
        return_dict: bool = True,
    ) -> Union[ContinuousFlowMatchSchedulerOutput, Tuple[torch.Tensor]]:
        if prediction_type == "x":
            prev_sample = model_output
        else:
            t_src = torch.as_tensor(timestep_src, device=sample.device, dtype=sample.dtype)
            t_tgt = torch.as_tensor(timestep_tgt, device=sample.device, dtype=sample.dtype)
            while t_src.ndim < sample.ndim:
                t_src = t_src.unsqueeze(-1)
                t_tgt = t_tgt.unsqueeze(-1)
            dt = t_src - t_tgt
            if self.config.solver == "heun" and model_output_next is not None:
                prev_sample = sample - dt * 0.5 * (model_output + model_output_next)
            else:
                prev_sample = sample - dt * model_output

        step_index = self._resolve_step_index(timestep_src)
        self._step_index = step_index + 1

        if not return_dict:
            return (prev_sample,)
        return ContinuousFlowMatchSchedulerOutput(prev_sample=prev_sample)
