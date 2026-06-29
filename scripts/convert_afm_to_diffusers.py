#!/usr/bin/env python3
"""Convert ByteDance AFM .pth checkpoints into BiliSakura/AFM-diffusers layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusers.models.afm.generator import Generator  # noqa: E402
from diffusers.models.afm.generator_deep import GeneratorDeep  # noqa: E402

AFM_ARCH_PRESETS: Dict[str, Dict[str, int]] = {
    "AFM-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    "AFM-M/2": {"depth": 16, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "AFM-L/2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "AFM-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
}

REPO_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = REPO_ROOT / "models/ByteDance-Seed/Adversarial-Flow-Models"
DEFAULT_OUTPUT = REPO_ROOT / "models/BiliSakura/AFM-diffusers"
TEMPLATES = ROOT / "hub_templates"
LABELS_PATH = REPO_ROOT / "src/labels/id2label_en.json"
DEFAULT_VAE_DIR = REPO_ROOT / "models/BiliSakura/iMF-diffusers/iMF-XL-2/vae"

SIZE_TO_MODEL = {
    "b2": "AFM-B/2",
    "m2": "AFM-M/2",
    "l2": "AFM-L/2",
    "xl2": "AFM-XL/2",
}


@dataclass(frozen=True)
class CheckpointSpec:
    filename: str
    variant_name: str
    model_type: str
    architecture: str = "standard"
    repeat: int = 1
    use_t_src: bool = False
    use_t_tgt: bool = False
    pred_type: str = "x"
    num_inference_steps: int = 1
    scheduler_solver: str = "euler"


CHECKPOINT_SPECS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec("b2_1nfe_guided_FID3.05.pth", "AFM-B-2-1NFE-guided", "AFM-B/2"),
    CheckpointSpec("b2_1nfe_noguide_FID6.07.pth", "AFM-B-2-1NFE-noguide", "AFM-B/2"),
    CheckpointSpec("m2_1nfe_guided_FID2.82.pth", "AFM-M-2-1NFE-guided", "AFM-M/2"),
    CheckpointSpec("m2_1nfe_noguide_FID5.21.pth", "AFM-M-2-1NFE-noguide", "AFM-M/2"),
    CheckpointSpec("l2_1nfe_guided_FID2.63.pth", "AFM-L-2-1NFE-guided", "AFM-L/2"),
    CheckpointSpec("l2_1nfe_noguide_FID4.36.pth", "AFM-L-2-1NFE-noguide", "AFM-L/2"),
    CheckpointSpec("xl2_1nfe_guided_FID2.38.pth", "AFM-XL-2-1NFE-guided", "AFM-XL/2"),
    CheckpointSpec("xl2_1nfe_noguide_FID3.98.pth", "AFM-XL-2-1NFE-noguide", "AFM-XL/2"),
    CheckpointSpec(
        "xl2_2nfe_guided_FID2.11.pth",
        "AFM-XL-2-2NFE-guided",
        "AFM-XL/2",
        use_t_src=True,
        pred_type="v",
        num_inference_steps=2,
    ),
    CheckpointSpec(
        "xl2_2nfe_noguide_FID2.36.pth",
        "AFM-XL-2-2NFE-noguide",
        "AFM-XL/2",
        use_t_src=True,
        pred_type="v",
        num_inference_steps=2,
    ),
    CheckpointSpec(
        "xl2_4nfe_guided_FID2.03.pth",
        "AFM-XL-2-4NFE-guided",
        "AFM-XL/2",
        use_t_src=True,
        pred_type="v",
        num_inference_steps=4,
    ),
    CheckpointSpec(
        "xl2_56layer_1nfe_guided_FID2.08.pth",
        "AFM-XL-2-56layer-1NFE-guided",
        "AFM-XL/2",
        architecture="deep",
        repeat=2,
    ),
    CheckpointSpec(
        "xl2_112layer_1nfe_guided_FID1.94.pth",
        "AFM-XL-2-112layer-1NFE-guided",
        "AFM-XL/2",
        architecture="deep",
        repeat=4,
    ),
)


# Paper benchmark rows (ImageNet 256x256, LDM latent space). Metrics: FID, sFID, IS, Precision, Recall.
BENCHMARK_ROWS: tuple[tuple[str, ...], ...] = (
    ("Model", "Params", "Guidance", "NFE", "FID", "sFID", "IS", "Prec.", "Recall", "Checkpoint"),
    ("AFM-B/2", "130M", "None", "1", "6.07", "5.31", "169.51", "0.72", "0.49", "AFM-B-2-1NFE-noguide"),
    ("AFM-M/2", "306M", "None", "1", "5.21", "5.60", "178.48", "0.75", "0.54", "AFM-M-2-1NFE-noguide"),
    ("AFM-L/2", "457M", "None", "1", "4.36", "5.39", "186.21", "0.77", "0.53", "AFM-L-2-1NFE-noguide"),
    ("AFM-XL/2", "673M", "None", "1", "3.98", "5.40", "201.85", "0.78", "0.52", "AFM-XL-2-1NFE-noguide"),
    ("AFM-XL/2", "673M", "None", "2", "2.36", "4.35", "235.77", "0.81", "0.52", "AFM-XL-2-2NFE-noguide"),
    ("AFM-B/2", "130M", "CG+DA", "1", "3.05", "5.32", "269.18", "0.81", "0.51", "AFM-B-2-1NFE-guided"),
    ("AFM-M/2", "306M", "CG+DA", "1", "2.82", "5.20", "279.12", "0.81", "0.50", "AFM-M-2-1NFE-guided"),
    ("AFM-L/2", "457M", "CG+DA", "1", "2.63", "5.10", "277.96", "0.81", "0.52", "AFM-L-2-1NFE-guided"),
    ("AFM-XL/2", "673M", "CG+DA", "1", "2.38", "4.87", "284.18", "0.81", "0.52", "AFM-XL-2-1NFE-guided"),
    ("AFM-XL/2", "675M", "CG+DA", "2", "2.11", "4.33", "273.84", "0.82", "0.55", "AFM-XL-2-2NFE-guided"),
    ("AFM-XL/2 (2× deep, 56-layer)", "675M", "CG+DA", "1", "2.08", "4.79", "298.33", "0.79", "0.56", "AFM-XL-2-56layer-1NFE-guided"),
    ("AFM-XL/2", "675M", "CG+DA", "4", "2.03", "4.59", "259.66", "0.78", "0.59", "AFM-XL-2-4NFE-guided"),
    ("AFM-XL/2 (4× deep, 112-layer)", "675M", "CG+DA", "1", "1.94", "4.54", "292.20", "0.79", "0.56", "AFM-XL-2-112layer-1NFE-guided"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Directory containing afm/models/*.pth and misc/sd-vae-ft-mse.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output BiliSakura/AFM-diffusers directory",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Convert only this checkpoint filename (repeatable). Defaults to all AFM specs.",
    )
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE conversion/copy.")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def _write_generator_bundle(generator_dir: Path) -> None:
    generator_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATES / "modeling_afm.py", generator_dir / "modeling_afm.py")


def _load_legacy_model(spec: CheckpointSpec, ckpt_path: Path) -> torch.nn.Module:
    arch = dict(AFM_ARCH_PRESETS[spec.model_type])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if spec.architecture == "deep":
        model = GeneratorDeep(repeat=spec.repeat, **arch)
    else:
        model = Generator(use_t_src=spec.use_t_src, use_t_tgt=spec.use_t_tgt, **arch)
    model.load_state_dict(state, strict=True)
    return model


def _save_generator(model: torch.nn.Module, generator_dir: Path, spec: CheckpointSpec) -> None:
    from safetensors.torch import save_file

    arch = AFM_ARCH_PRESETS[spec.model_type]
    config = {
        "_class_name": "AFMGeneratorDeep2DModel" if spec.architecture == "deep" else "AFMGenerator2DModel",
        "_diffusers_version": "0.36.0",
        "model_type": spec.model_type,
        "architecture": spec.architecture,
        "repeat": spec.repeat,
        "use_t_src": spec.use_t_src,
        "use_t_tgt": spec.use_t_tgt,
        "pred_type": spec.pred_type,
        "num_inference_steps": spec.num_inference_steps,
        "learn_sigma": False,
        "class_dropout_prob": 0.0,
        "input_size": 32,
        "num_classes": 1000,
        "in_channels": 4,
        **arch,
    }
    (generator_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file(model.state_dict(), generator_dir / "diffusion_pytorch_model.safetensors")


def _materialize_vae(vae_dir: Path, template_vae_dir: Path) -> None:
    if (vae_dir / "diffusion_pytorch_model.safetensors").exists():
        return
    if not template_vae_dir.is_dir():
        raise FileNotFoundError(
            f"Bundled VAE not found at {template_vae_dir}. "
            "Convert iMF-diffusers first or provide sd-vae-ft-mse weights."
        )
    shutil.copytree(template_vae_dir, vae_dir)


def _write_scheduler(scheduler_dir: Path, solver: str) -> None:
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATES / "scheduling_continuous_flow.py", scheduler_dir / "scheduling_continuous_flow.py")
    config = {
        "_class_name": "ContinuousFlowMatchScheduler",
        "_diffusers_version": "0.36.0",
        "solver": solver,
    }
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _write_model_index(
    variant_dir: Path,
    id2label: Dict[str, str],
    pred_type: str,
    num_inference_steps: int,
    architecture: str = "standard",
) -> None:
    generator_class = "AFMGeneratorDeep2DModel" if architecture == "deep" else "AFMGenerator2DModel"
    model_index = {
        "_class_name": ["pipeline", "AFMPipeline"],
        "_diffusers_version": "0.36.0",
        "generator": ["modeling_afm", generator_class],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["scheduling_continuous_flow", "ContinuousFlowMatchScheduler"],
        "pred_type": pred_type,
        "id2label": id2label,
    }
    (variant_dir / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")


def convert_checkpoint(
    spec: CheckpointSpec,
    source_dir: Path,
    output_dir: Path,
    id2label: Dict[str, str],
    template_vae_dir: Path,
    skip_vae: bool,
    dtype: torch.dtype,
) -> Path:
    ckpt_path = source_dir / "afm/models" / spec.filename
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    variant_dir = output_dir / spec.variant_name
    generator_dir = variant_dir / "generator"
    scheduler_dir = variant_dir / "scheduler"
    vae_dir = variant_dir / "vae"

    if generator_dir.exists():
        shutil.rmtree(generator_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    _write_generator_bundle(generator_dir)
    model = _load_legacy_model(spec, ckpt_path)
    model = model.to(dtype=dtype)
    _save_generator(model, generator_dir, spec)

    _write_scheduler(scheduler_dir, spec.scheduler_solver)
    if not skip_vae:
        _materialize_vae(vae_dir, template_vae_dir)
    _write_model_index(
        variant_dir, id2label, spec.pred_type, spec.num_inference_steps, spec.architecture
    )
    shutil.copy2(TEMPLATES / "pipeline.py", variant_dir / "pipeline.py")
    return variant_dir


def _selected_specs(filenames: Optional[Iterable[str]]) -> list[CheckpointSpec]:
    if not filenames:
        return list(CHECKPOINT_SPECS)
    lookup = {spec.filename: spec for spec in CHECKPOINT_SPECS}
    selected = []
    for name in filenames:
        if name not in lookup:
            raise ValueError(f"Unknown checkpoint '{name}'. Known: {sorted(lookup)}")
        selected.append(lookup[name])
    return selected


def _write_repo_readme(output_dir: Path, variants: list[str]) -> None:
    lines = [
        "---",
        "license: mit",
        "library_name: diffusers",
        "pipeline_tag: text-to-image",
        "tags:",
        "- diffusers",
        "- afm",
        "- adversarial-flow-models",
        "- class-conditional",
        "- imagenet",
        "inference: true",
        "widget:",
        "- output:",
        "    url: AFM-XL-2-56layer-1NFE-guided/demo.png",
        "language:",
        "- en",
        "---",
        "",
        "# BiliSakura/AFM-diffusers",
        "",
        "Self-contained [Adversarial Flow Models](https://arxiv.org/abs/2511.22475) checkpoints for Hugging Face diffusers.",
        "",
        "Converted from `ByteDance-Seed/Adversarial-Flow-Models` using `libs/AFM-diffusers/scripts/convert_afm_to_diffusers.py`.",
        "",
        "All models use LDM (Rombach et al., 2022) latent space with `sd-vae-ft-mse`. Guidance abbreviations: **CG** = classifier guidance (Dhariwal & Nichol, 2021), **DA** = data augmentation (Karras et al., 2020a).",
        "",
        "## Demo",
        "",
        "`AFM-XL-2-2NFE-noguide` — class **207** (*golden retriever*), seed **0**, 2 NFE:",
        "",
        '<p align="center">',
        '  <img src="AFM-XL-2-2NFE-noguide/demo.png" alt="AFM-XL-2-2NFE-noguide demo (class 207, seed 0)" width="256"/>',
        "</p>",
        "",
        "Each variant folder includes `demo.png` generated with the same prompt settings.",
        "",
        "## Benchmark results (ImageNet 256×256)",
        "",
        "| " + " | ".join(BENCHMARK_ROWS[0]) + " |",
        "| " + " | ".join(["---"] * len(BENCHMARK_ROWS[0])) + " |",
    ]
    for row in BENCHMARK_ROWS[1:]:
        checkpoint_cell = f"`{row[-1]}/`" if row[-1] in variants else row[-1]
        cells = list(row[:-1]) + [checkpoint_cell]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Available checkpoints",
            "",
            "| Variant | Model | Steps | Guidance |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for spec in CHECKPOINT_SPECS:
        if spec.variant_name not in variants:
            continue
        guide = "guided" if "guided" in spec.filename else "noguide"
        lines.append(f"| `{spec.variant_name}/` | {spec.model_type} | {spec.num_inference_steps} | {guide} |")
    lines.extend(
        [
            "",
            "## Inference",
            "",
            "```python",
            "from pathlib import Path",
            "import torch",
            "from diffusers import DiffusionPipeline",
            "",
            'model_dir = Path("./AFM-XL-2-1NFE-guided")',
            "pipe = DiffusionPipeline.from_pretrained(",
            "    str(model_dir),",
            "    local_files_only=True,",
            "    custom_pipeline=str(model_dir / \"pipeline.py\"),",
            "    trust_remote_code=True,",
            "    torch_dtype=torch.bfloat16,",
            ").to(\"cuda\")",
            "",
            "image = pipe(class_labels=\"golden retriever\", num_inference_steps=1).images[0]",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    id2label = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    specs = _selected_specs(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for spec in specs:
        print(f"Converting {spec.filename} -> {spec.variant_name}")
        out = convert_checkpoint(
            spec,
            args.source_dir,
            args.output_dir,
            id2label,
            DEFAULT_VAE_DIR,
            args.skip_vae,
            dtype,
        )
        converted.append(spec.variant_name)
        print(f"  saved to {out}")

    _write_repo_readme(args.output_dir, converted)
    print(f"Done. Converted {len(converted)} variant(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
