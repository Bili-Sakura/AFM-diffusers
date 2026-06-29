#!/usr/bin/env python3
"""Compare raw ByteDance AFM .pth checkpoints against converted diffusers variants."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_pil_image

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parents[1]
DEFAULT_SOURCE = REPO_ROOT / "models/ByteDance-Seed/Adversarial-Flow-Models"
DEFAULT_DIFFUSERS = REPO_ROOT / "models/BiliSakura/AFM-diffusers"
HF_CACHE = Path.home() / ".cache/huggingface/modules/diffusers_modules/local"

AFM_ARCH = {
    "AFM-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    "AFM-M/2": {"depth": 16, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "AFM-L/2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "AFM-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
}


@dataclass(frozen=True)
class CompareSpec:
    variant_name: str
    checkpoint: str
    model_type: str
    architecture: str = "standard"
    repeat: int = 1
    use_t_src: bool = False
    pred_type: str = "x"
    steps: int = 1


COMPARE_SPECS: tuple[CompareSpec, ...] = (
    CompareSpec("AFM-B-2-1NFE-guided", "b2_1nfe_guided_FID3.05.pth", "AFM-B/2"),
    CompareSpec("AFM-XL-2-1NFE-guided", "xl2_1nfe_guided_FID2.38.pth", "AFM-XL/2"),
    CompareSpec(
        "AFM-XL-2-2NFE-guided",
        "xl2_2nfe_guided_FID2.11.pth",
        "AFM-XL/2",
        use_t_src=True,
        pred_type="v",
        steps=2,
    ),
    CompareSpec(
        "AFM-XL-2-4NFE-guided",
        "xl2_4nfe_guided_FID2.03.pth",
        "AFM-XL/2",
        use_t_src=True,
        pred_type="v",
        steps=4,
    ),
    CompareSpec(
        "AFM-XL-2-56layer-1NFE-guided",
        "xl2_56layer_1nfe_guided_FID2.08.pth",
        "AFM-XL/2",
        architecture="deep",
        repeat=2,
    ),
    CompareSpec(
        "AFM-XL-2-112layer-1NFE-guided",
        "xl2_112layer_1nfe_guided_FID1.94.pth",
        "AFM-XL/2",
        architecture="deep",
        repeat=4,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--diffusers-dir", type=Path, default=DEFAULT_DIFFUSERS)
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/afm_raw_vs_diffusers")
    return parser.parse_args()


def _load_hub_modeling(variant_dir: Path):
    module_path = variant_dir / "generator" / "modeling_afm.py"
    spec = importlib.util.spec_from_file_location("modeling_afm", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["modeling_afm"] = module
    spec.loader.exec_module(module)
    return module


def _build_raw_generator(module, spec: CompareSpec, ckpt_path: Path, device: torch.device):
    arch = AFM_ARCH[spec.model_type]
    if spec.architecture == "deep":
        model = module.AFMGeneratorDeep2DModel(
            model_type=spec.model_type,
            architecture="deep",
            repeat=spec.repeat,
            pred_type=spec.pred_type,
            num_inference_steps=spec.steps,
            **arch,
        )
    else:
        model = module.AFMGenerator2DModel(
            model_type=spec.model_type,
            architecture="standard",
            use_t_src=spec.use_t_src,
            use_t_tgt=False,
            pred_type=spec.pred_type,
            num_inference_steps=spec.steps,
            **arch,
        )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def _load_diffusers_pipeline(variant_dir: Path, device: torch.device) -> DiffusionPipeline:
    if HF_CACHE.exists():
        shutil.rmtree(HF_CACHE)
    return DiffusionPipeline.from_pretrained(
        str(variant_dir),
        local_files_only=True,
        custom_pipeline=str(variant_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)


def _build_raw_pipeline(module, spec: CompareSpec, ckpt_path: Path, variant_dir: Path, device: torch.device):
    generator = _build_raw_generator(module, spec, ckpt_path, device)
    pipe = _load_diffusers_pipeline(variant_dir, device)
    pipe.generator = generator
    return pipe


def _tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "mse": float(torch.mean((a - b) ** 2).item()),
    }


def _run_pipeline(pipe, *, class_label: int, steps: int, sampler: str, latents: torch.Tensor):
    return pipe(
        class_labels=class_label,
        num_inference_steps=steps,
        sampler=sampler,
        latents=latents.clone(),
        output_type="pt",
    ).images


def compare_variant(
    spec: CompareSpec,
    source_dir: Path,
    diffusers_dir: Path,
    *,
    class_label: int,
    seed: int,
    sampler: str,
    atol: float,
    output_dir: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = source_dir / "afm/models" / spec.checkpoint
    variant_dir = diffusers_dir / spec.variant_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    if not variant_dir.is_dir():
        raise FileNotFoundError(variant_dir)

    module = _load_hub_modeling(variant_dir)
    raw_pipe = _build_raw_pipeline(module, spec, ckpt_path, variant_dir, device)
    diff_pipe = _load_diffusers_pipeline(variant_dir, device)

    latent_shape = (1, 4, 32, 32)
    latents = torch.randn(latent_shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))

    with torch.inference_mode():
        raw_out = _run_pipeline(raw_pipe, class_label=class_label, steps=spec.steps, sampler=sampler, latents=latents)
        diff_out = _run_pipeline(diff_pipe, class_label=class_label, steps=spec.steps, sampler=sampler, latents=latents)

    image_stats = _tensor_stats(raw_out, diff_out)
    passed = image_stats["max_abs"] <= atol

    out_dir = output_dir / spec.variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil_image(raw_out[0].mul(0.5).add(0.5).clamp(0, 1)).save(out_dir / "raw.png")
    to_pil_image(diff_out[0].mul(0.5).add(0.5).clamp(0, 1)).save(out_dir / "diffusers.png")
    diff_vis = (raw_out - diff_out).abs().mul(10.0).clamp(0, 1)
    to_pil_image(diff_vis[0].mul(0.5).add(0.5).clamp(0, 1)).save(out_dir / "abs_diff_x10.png")

    result = {
        "variant": spec.variant_name,
        "checkpoint": spec.checkpoint,
        "class_label": class_label,
        "seed": seed,
        "steps": spec.steps,
        "pred_type": spec.pred_type,
        "sampler": sampler,
        "image": image_stats,
        "passed": passed,
        "atol": atol,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del raw_pipe, diff_pipe
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    specs = COMPARE_SPECS
    if args.variant:
        lookup = {spec.variant_name: spec for spec in COMPARE_SPECS}
        specs = tuple(lookup[name] for name in args.variant)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in specs:
        print(f"Comparing {spec.variant_name}...", flush=True)
        result = compare_variant(
            spec,
            args.source_dir,
            args.diffusers_dir,
            class_label=args.class_label,
            seed=args.seed,
            sampler=args.sampler,
            atol=args.atol,
            output_dir=args.output_dir,
        )
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        stats = result["image"]
        print(
            f"  {status}: max_abs={stats['max_abs']:.2e}, mean_abs={stats['mean_abs']:.2e}, mse={stats['mse']:.2e}",
            flush=True,
        )

    summary = {
        "class_label": args.class_label,
        "seed": args.seed,
        "sampler": args.sampler,
        "atol": args.atol,
        "results": results,
        "all_passed": all(item["passed"] for item in results),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
