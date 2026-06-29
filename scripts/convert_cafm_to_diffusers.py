#!/usr/bin/env python3
"""Convert ByteDance CAFM .pth checkpoints into BiliSakura/CAFM-diffusers layout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = REPO_ROOT / "models/ByteDance-Seed/Adversarial-Flow-Models"
DEFAULT_ZIMAGE_SOURCE = REPO_ROOT / "models/Tongyi-MAI/Z-Image"
DEFAULT_OUTPUT = REPO_ROOT / "models/BiliSakura/CAFM-diffusers"
TEMPLATES = ROOT / "hub_templates"
LABELS_PATH = REPO_ROOT / "src/labels/id2label_en.json"
DEFAULT_VAE_DIR = REPO_ROOT / "models/BiliSakura/SiT-diffusers/SiT-XL-2-256/vae"
JIT_BACKBONE = REPO_ROOT / "models/BiliSakura/JiT-diffusers/JiT-H-16/transformer/jit_transformer_2d.py"
SIT_BACKBONE = REPO_ROOT / "models/BiliSakura/SiT-diffusers/SiT-XL-2-256/transformer/transformer_sit.py"
ZIMAGE_COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")


@dataclass(frozen=True)
class CheckpointSpec:
    filename: str
    variant_name: str
    backbone: str
    num_inference_steps: int
    scheduler_solver: str = "heun"


CHECKPOINT_SPECS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        "jit_h16_cafm.pth",
        "CAFM-JiT-H-16-256",
        "jit",
        num_inference_steps=100,
    ),
    CheckpointSpec(
        "sit_xl2_cafm.pth",
        "CAFM-SiT-XL-2-256",
        "sit",
        num_inference_steps=250,
    ),
)

ZIMAGE_VARIANT = "CAFM-Z-Image-T2I"
ZIMAGE_DEMO_PROMPT = "A golden retriever sitting in a sunny park, photo realistic."
ZIMAGE_DEMO_STEPS = 25

BENCHMARK_ROWS: tuple[tuple[str, ...], ...] = (
    ("Model", "Space", "NFE", "FID", "Checkpoint"),
    ("CAFM JiT-H/16", "pixel", "100", "1.80", "CAFM-JiT-H-16-256"),
    ("CAFM SiT-XL/2", "latent", "250", "1.53", "CAFM-SiT-XL-2-256"),
    ("CAFM Z-Image", "latent T2I", "25", "—", "CAFM-Z-Image-T2I"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--zimage-source-dir",
        type=Path,
        default=DEFAULT_ZIMAGE_SOURCE,
        help="Tongyi-MAI/Z-Image diffusers source directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Convert only this checkpoint filename (repeatable). Use 'zimage' for Z-Image T2I. Defaults to all specs.",
    )
    parser.add_argument(
        "--include-zimage",
        action="store_true",
        help="Also convert Tongyi-MAI/Z-Image into CAFM-Z-Image-T2I.",
    )
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        default=None,
        help="Optional CAFM finetuned .pth to load into the Z-Image transformer.",
    )
    parser.add_argument(
        "--keep-zimage-source",
        action="store_true",
        help="Copy Z-Image weight shards instead of moving them out of --zimage-source-dir.",
    )
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE copy for SiT variants.")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--demo", action="store_true", help="Generate demo.png per variant after conversion.")
    return parser.parse_args()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remap_jit_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace(".adaLN_modulation.1.", ".adaLN_modulation.")
        if new_key.startswith("final_layer."):
            new_key = new_key.replace("final_layer.norm_final", "norm_final")
            new_key = new_key.replace("final_layer.linear", "linear_final")
            new_key = new_key.replace("final_layer.adaLN_modulation", "adaLN_modulation_final")
        remapped[new_key] = value
    return remapped


def _jit_config() -> dict:
    return {
        "_class_name": "CAFMJiTGenerator2DModel",
        "_diffusers_version": "0.36.0",
        "sample_size": 256,
        "patch_size": 16,
        "hidden_size": 1280,
        "num_layers": 32,
        "num_attention_heads": 16,
        "bottleneck_dim": 256,
        "in_context_len": 32,
        "in_context_start": 10,
        "attention_dropout": 0.0,
        "dropout": 0.2,
        "num_classes": 1000,
        "in_channels": 3,
        "mlp_ratio": 4.0,
        "norm_eps": 1e-6,
    }


def _sit_config(num_inference_steps: int) -> dict:
    return {
        "_class_name": "CAFMSiTGenerator2DModel",
        "_diffusers_version": "0.36.0",
        "input_size": 32,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 1152,
        "depth": 28,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "learn_sigma": True,
        "class_dropout_prob": 0.0,
        "num_classes": 1001,
    }


def _write_generator_bundle(generator_dir: Path, spec: CheckpointSpec) -> None:
    generator_dir.mkdir(parents=True, exist_ok=True)
    if spec.backbone == "jit":
        bundled = JIT_BACKBONE.read_text(encoding="utf-8")
        bundled += "\n\n" + (TEMPLATES / "modeling_cafm_jit_wrapper.py").read_text(encoding="utf-8")
        (generator_dir / "modeling_cafm_jit.py").write_text(bundled, encoding="utf-8")
    else:
        bundled = SIT_BACKBONE.read_text(encoding="utf-8")
        bundled += "\n\n" + (TEMPLATES / "modeling_cafm_sit_wrapper.py").read_text(encoding="utf-8")
        (generator_dir / "modeling_cafm_sit.py").write_text(bundled, encoding="utf-8")


def _load_and_convert_weights(spec: CheckpointSpec, ckpt_path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if spec.backbone == "jit":
        jit_mod = _load_module("jit_transformer_2d", JIT_BACKBONE)
        model = jit_mod.JiTTransformer2DModel(
            sample_size=256,
            patch_size=16,
            hidden_size=1280,
            num_layers=32,
            num_attention_heads=16,
            bottleneck_dim=256,
            in_context_len=32,
            in_context_start=10,
            attention_dropout=0.0,
            dropout=0.2,
            num_classes=1000,
            in_channels=3,
        )
        remapped = remap_jit_state_dict(state)
        model.load_state_dict(remapped, strict=True)
        return model.state_dict()

    sit_mod = _load_module("transformer_sit", SIT_BACKBONE)
    model = sit_mod.SiTTransformer2DModel(
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,
        learn_sigma=True,
        class_dropout_prob=0.0,
        num_classes=1001,
        input_size=32,
        in_channels=4,
    )
    model.load_state_dict(state, strict=True)
    return model.state_dict()


def _save_generator(state_dict: Dict[str, torch.Tensor], generator_dir: Path, config: dict) -> None:
    from safetensors.torch import save_file

    (generator_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file(state_dict, generator_dir / "diffusion_pytorch_model.safetensors")


def _materialize_vae(vae_dir: Path, template_vae_dir: Path) -> None:
    if (vae_dir / "diffusion_pytorch_model.safetensors").exists():
        return
    if not template_vae_dir.is_dir():
        raise FileNotFoundError(f"Bundled VAE not found at {template_vae_dir}.")
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


def _write_model_index(variant_dir: Path, spec: CheckpointSpec, id2label: Dict[str, str]) -> None:
    if spec.backbone == "jit":
        generator_entry = ["modeling_cafm_jit", "CAFMJiTGenerator2DModel"]
        pipeline_class = "CAFMJiTPipeline"
        components = {
            "generator": generator_entry,
            "scheduler": ["scheduling_continuous_flow", "ContinuousFlowMatchScheduler"],
        }
    else:
        generator_entry = ["modeling_cafm_sit", "CAFMSiTGenerator2DModel"]
        pipeline_class = "CAFMSiTPipeline"
        components = {
            "generator": generator_entry,
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["scheduling_continuous_flow", "ContinuousFlowMatchScheduler"],
        }

    model_index = {
        "_class_name": ["pipeline", pipeline_class],
        "_diffusers_version": "0.36.0",
        "id2label": id2label,
        **components,
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
    ckpt_path = source_dir / "cafm" / spec.filename
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    variant_dir = output_dir / spec.variant_name
    generator_dir = variant_dir / "generator"
    scheduler_dir = variant_dir / "scheduler"
    vae_dir = variant_dir / "vae"

    if generator_dir.exists():
        shutil.rmtree(generator_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    _write_generator_bundle(generator_dir, spec)
    state_dict = _load_and_convert_weights(spec, ckpt_path)
    if dtype != torch.float32:
        state_dict = {key: value.to(dtype=dtype) for key, value in state_dict.items()}

    if spec.backbone == "jit":
        config = _jit_config()
    else:
        config = _sit_config(spec.num_inference_steps)
    _save_generator(state_dict, generator_dir, config)

    _write_scheduler(scheduler_dir, spec.scheduler_solver)
    if spec.backbone == "sit" and not skip_vae:
        if vae_dir.exists():
            shutil.rmtree(vae_dir)
        _materialize_vae(vae_dir, template_vae_dir)

    _write_model_index(variant_dir, spec, id2label)

    pipeline_template = TEMPLATES / (
        "pipeline_cafm_jit.py" if spec.backbone == "jit" else "pipeline_cafm_sit.py"
    )
    shutil.copy2(pipeline_template, variant_dir / "pipeline.py")
    return variant_dir


def _materialize_component(src: Path, dst: Path, *, move: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Missing component directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copytree(src, dst)


def _remove_zimage_source(zimage_source: Path) -> None:
    if not zimage_source.exists():
        return
    shutil.rmtree(zimage_source)
    parent = zimage_source.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def _write_zimage_scheduler(scheduler_dir: Path) -> None:
    return


def _write_zimage_model_index(variant_dir: Path) -> None:
    model_index = {
        "_class_name": ["pipeline", "CAFMZImagePipeline"],
        "_diffusers_version": "0.38.0",
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "text_encoder": ["transformers", "Qwen3ForCausalLM"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    (variant_dir / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")


def _maybe_apply_zimage_checkpoint(transformer_dir: Path, checkpoint: Path, dtype: torch.dtype) -> None:
    from diffusers import ZImageTransformer2DModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = ZImageTransformer2DModel.from_pretrained(str(transformer_dir), local_files_only=True, torch_dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.save_pretrained(str(transformer_dir), safe_serialization=True)


def convert_zimage(
    zimage_source: Path,
    output_dir: Path,
    *,
    keep_source: bool,
    generator_checkpoint: Optional[Path],
    dtype: torch.dtype,
) -> Path:
    if not (zimage_source / "transformer").is_dir():
        raise FileNotFoundError(f"Missing Z-Image transformer at {zimage_source / 'transformer'}")

    variant_dir = output_dir / ZIMAGE_VARIANT
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    move_source = not keep_source
    for component in ZIMAGE_COMPONENTS:
        _materialize_component(
            zimage_source / component,
            variant_dir / component,
            move=move_source,
        )

    if generator_checkpoint is not None:
        if not generator_checkpoint.is_file():
            raise FileNotFoundError(f"Missing generator checkpoint: {generator_checkpoint}")
        print(f"  applying CAFM checkpoint {generator_checkpoint.name}")
        _maybe_apply_zimage_checkpoint(variant_dir / "transformer", generator_checkpoint, dtype)

    _write_zimage_model_index(variant_dir)
    shutil.copy2(TEMPLATES / "pipeline_cafm_zimage.py", variant_dir / "pipeline.py")

    if move_source:
        _remove_zimage_source(zimage_source)
        print(f"  removed source directory {zimage_source}")

    return variant_dir


def _selected_specs(filenames: Optional[Iterable[str]]) -> tuple[list[CheckpointSpec], bool]:
    include_zimage = False
    if not filenames:
        return list(CHECKPOINT_SPECS), False
    lookup = {spec.filename: spec for spec in CHECKPOINT_SPECS}
    selected = []
    for name in filenames:
        if name == "zimage":
            include_zimage = True
            continue
        if name not in lookup:
            raise ValueError(f"Unknown checkpoint '{name}'. Known: {sorted(lookup)} + ['zimage']")
        selected.append(lookup[name])
    return selected, include_zimage


def _discovered_variants(output_dir: Path) -> list[str]:
    variants: list[str] = []
    for spec in CHECKPOINT_SPECS:
        if (output_dir / spec.variant_name).is_dir():
            variants.append(spec.variant_name)
    if (output_dir / ZIMAGE_VARIANT).is_dir():
        variants.append(ZIMAGE_VARIANT)
    return variants


def _write_repo_readme(output_dir: Path, variants: list[str]) -> None:
    lines = [
        "---",
        "license: mit",
        "library_name: diffusers",
        "pipeline_tag: text-to-image",
        "tags:",
        "- diffusers",
        "- cafm",
        "- continuous-adversarial-flow-models",
        "- class-conditional",
        "- imagenet",
        "- text-to-image",
        "- z-image",
        "inference: true",
        "widget:",
        "- output:",
        "    url: CAFM-JiT-H-16-256/demo.png",
        "language:",
        "- en",
        "---",
        "",
        "# BiliSakura/CAFM-diffusers",
        "",
        "Self-contained [Continuous Adversarial Flow Models](https://arxiv.org/abs/2604.11521) checkpoints for Hugging Face diffusers.",
        "",
        "Converted from `ByteDance-Seed/Adversarial-Flow-Models` using `libs/AFM-diffusers/scripts/convert_cafm_to_diffusers.py`.",
        "Z-Image weights are bundled self-contained under `CAFM-Z-Image-T2I/`.",
        "",
        "## Demo",
        "",
        "`CAFM-JiT-H-16-256` — class **207** (*golden retriever*), seed **0**, 100 NFE (Heun):",
        "",
        '<p align="center">',
        '  <img src="CAFM-JiT-H-16-256/demo.png" alt="CAFM-JiT-H-16-256 demo (class 207, seed 0)" width="256"/>',
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
            "| Variant | Backbone | Steps | Solver |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for spec in CHECKPOINT_SPECS:
        if spec.variant_name not in variants:
            continue
        lines.append(
            f"| `{spec.variant_name}/` | {spec.backbone.upper()} | {spec.num_inference_steps} | {spec.scheduler_solver} |"
        )
    if ZIMAGE_VARIANT in variants:
        lines.append(f"| `{ZIMAGE_VARIANT}/` | Z-IMAGE | {ZIMAGE_DEMO_STEPS} | euler |")
    lines.extend(
        [
            "",
            "## Inference",
            "",
            "### ImageNet class-conditional (JiT / SiT)",
            "",
            "```python",
            "from pathlib import Path",
            "import torch",
            "from diffusers import DiffusionPipeline",
            "",
            'model_dir = Path("./CAFM-SiT-XL-2-256")',
            "pipe = DiffusionPipeline.from_pretrained(",
            "    str(model_dir),",
            "    local_files_only=True,",
            "    custom_pipeline=str(model_dir / \"pipeline.py\"),",
            "    trust_remote_code=True,",
            "    torch_dtype=torch.bfloat16,",
            ").to(\"cuda\")",
            "",
            "image = pipe(class_labels=\"golden retriever\", num_inference_steps=250, sampler=\"heun\").images[0]",
            "```",
            "",
            "### Text-to-image (Z-Image)",
            "",
            "```python",
            "from pathlib import Path",
            "import torch",
            "from diffusers import DiffusionPipeline",
            "",
            f'model_dir = Path("./{ZIMAGE_VARIANT}")',
            "pipe = DiffusionPipeline.from_pretrained(",
            "    str(model_dir),",
            "    local_files_only=True,",
            "    custom_pipeline=str(model_dir / \"pipeline.py\"),",
            "    trust_remote_code=True,",
            "    torch_dtype=torch.bfloat16,",
            ")",
            "pipe.enable_model_cpu_offload()  # recommended for single-GPU inference",
            "",
            "image = pipe(",
            f"    prompt=\"{ZIMAGE_DEMO_PROMPT}\",",
            "    height=512,",
            "    width=512,",
            f"    num_inference_steps={ZIMAGE_DEMO_STEPS},",
            "    sampler=\"euler\",",
            ").images[0]",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _maybe_generate_demo(variant_dir: Path, spec: Optional[CheckpointSpec] = None, *, is_zimage: bool = False) -> None:
    if not torch.cuda.is_available():
        name = ZIMAGE_VARIANT if is_zimage else spec.variant_name
        print(f"  skip demo for {name}: CUDA unavailable")
        return
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        str(variant_dir),
        local_files_only=True,
        custom_pipeline=str(variant_dir / "pipeline.py"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if is_zimage else torch.float32,
    )
    if is_zimage:
        pipe.enable_model_cpu_offload()
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)
    else:
        pipe = pipe.to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(0)
    if is_zimage:
        image = pipe(
            prompt=ZIMAGE_DEMO_PROMPT,
            height=512,
            width=512,
            num_inference_steps=ZIMAGE_DEMO_STEPS,
            sampler="euler",
            generator=generator,
        ).images[0]
        print(f"  wrote demo.png for {ZIMAGE_VARIANT}")
    else:
        image = pipe(
            class_labels="golden retriever",
            num_inference_steps=spec.num_inference_steps,
            sampler=spec.scheduler_solver,
            generator=generator,
        ).images[0]
        print(f"  wrote demo.png for {spec.variant_name}")
    image.save(variant_dir / "demo.png")


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    id2label = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    specs, include_zimage = _selected_specs(args.checkpoint)
    if args.include_zimage:
        include_zimage = True
    convert_all = args.checkpoint is None and not args.include_zimage
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
        if args.demo:
            _maybe_generate_demo(out, spec)

    if include_zimage or convert_all:
        variant_dir = args.output_dir / ZIMAGE_VARIANT
        source_ready = (args.zimage_source_dir / "transformer").is_dir()
        if source_ready:
            print(f"Converting Z-Image -> {ZIMAGE_VARIANT}")
            out = convert_zimage(
                args.zimage_source_dir,
                args.output_dir,
                keep_source=args.keep_zimage_source,
                generator_checkpoint=args.generator_checkpoint,
                dtype=dtype,
            )
            converted.append(ZIMAGE_VARIANT)
            print(f"  saved to {out}")
        elif variant_dir.is_dir():
            out = variant_dir
            print(f"Using existing {ZIMAGE_VARIANT} at {out}")
        else:
            raise FileNotFoundError(
                f"Missing Z-Image source at {args.zimage_source_dir / 'transformer'} "
                f"and no existing variant at {variant_dir}"
            )
        if args.demo:
            _maybe_generate_demo(out, is_zimage=True)

    _write_repo_readme(args.output_dir, _discovered_variants(args.output_dir))
    print(f"Done. Converted {len(converted)} variant(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
