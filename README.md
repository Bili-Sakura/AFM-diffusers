# Adversarial Flow Models for Diffusers

This repository is refactored to a Diffusers-style layout and API surface, following the [SiT-diffusers](https://github.com/Bili-Sakura/SiT-diffusers) pattern.

The old standalone model code under `models/` (with placeholder files requiring manual downloads from DiT/SiT/JiT repos) has been removed. All architectures are now implemented natively under `src/diffusers`.

## Package Layout

- `src/diffusers/models/transformers/transformer_dit.py`: `DiTTransformer2DModel` (native DiT)
- `src/diffusers/models/transformers/transformer_afm.py`: AFM generator, discriminator, classifier, and deep generator
- `src/diffusers/models/transformers/transformer_sit.py`: `SiTTransformer2DModel` (native SiT)
- `src/diffusers/models/transformers/transformer_cafm_sit.py`: CAFM SiT generator and discriminator
- `src/diffusers/models/transformers/transformer_jit.py`: `JiTTransformer2DModel` (native JiT)
- `src/diffusers/models/transformers/transformer_cafm_jit.py`: CAFM JiT generator and discriminator
- `src/diffusers/models/transformers/transformer_cafm_zimage.py`: Z-Image CAFM JVP discriminator
- `src/diffusers/models/autoencoders/autoencoder_kl_wrapper.py`: `AutoencoderKLWrapper`
- `src/diffusers/models/discriminators/jvp_discriminator.py`: `DiscriminatorJVP`
- `common/diffusers_ext.py`: registers extensions into the upstream `diffusers` namespace

## Install

```bash
pip install -e .
```

## AFMs

### Train

1. Install the package: `pip install -e .`
2. Download VAE and other misc [checkpoints](https://huggingface.co/ByteDance-Seed/Adversarial-Flow-Models/tree/main/misc) to the root directory.
3. Configure your dataset. Instruction is provided in the next section.
4. Run the training configurations provided in `configs/train/afm`.

```bash
torchrun main.py configs/train/afm/train_afm_1nfe.yaml
```

### Evaluate

1. Download [pre-trained AFM checkpoints](https://huggingface.co/ByteDance-Seed/Adversarial-Flow-Models), or use your own.
2. Generate 50K samples for FID evaluation.

```bash
torchrun main.py configs/generate/afm/generate_1nfe.yaml
```

3. Use `misc/pack_npz.py` to pack npz.
4. Use the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to evaluate FID.

## CAFMs

### Train

1. Install the package: `pip install -e .`
2. Download VAE and other misc [checkpoints](https://huggingface.co/ByteDance-Seed/Adversarial-Flow-Models/tree/main/misc) to the root directory.
3. Configure your dataset. Instruction is provided in the next section.
4. Download pre-trained checkpoints for initialization.
    * [SiT-XL/2 checkpoint](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt)
    * [JiT-H/16 checkpoint](https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/) — use `misc/convert_from_jit_format.py` to convert
    * [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) under `./Z-Image`
5. Run the training configurations provided in `configs/train/cafm`.

```bash
torchrun main.py configs/train/cafm/train_cafm_sit.yaml
torchrun main.py configs/train/cafm/train_cafm_jit.yaml
torchrun main.py configs/train/cafm/train_cafm_zimage.yaml
```

### Evaluate

1. Download [pre-trained CAFM checkpoints](https://huggingface.co/ByteDance-Seed/Adversarial-Flow-Models), or use your own.
2. Generation/evaluation uses external SiT/JiT sampling code. You may need `misc/convert_to_jit_format.py` for JiT checkpoint format conversion.

## Using Models Programmatically

After `pip install -e .`, models are available through the upstream `diffusers` namespace:

```python
from diffusers.models.transformers.transformer_afm import AFMGeneratorModel
from diffusers.models.transformers.transformer_cafm_sit import CAFMSiTGeneratorModel
from diffusers.models.autoencoders.autoencoder_kl_wrapper import AutoencoderKLWrapper
```

Training scripts auto-register extensions via `common/diffusers_ext.py`.

## Tests

```bash
pytest tests/test_afm_diffusers.py -q
```

## Dataloading

For ImageNet, implement an `IterableDataset` with a forever loop returning `image` and `label` tensors. For CAFM SiT, offline latents with shape `(4, 32, 32)` are also supported. For T2I, return `image` and `text`. The dataset must partition by rank and worker id for distributed training.

## Citation

```bibtex
@article{afm,
  title={Adversarial Flow Models},
  author={Lin, Shanchuan and Yang, Ceyuan and Lin, Zhijie and Chen, Hao and Fan, Haoqi},
  journal={arXiv preprint arXiv:2511.22475},
  year={2025}
}

@article{cafm,
  title={Continuous Adversarial Flow Models},
  author={Lin, Shanchuan and Yang, Ceyuan and Lin, Zhijie and Chen, Hao and Fan, Haoqi},
  journal={arXiv preprint arXiv:2604.11521},
  year={2026}
}
```

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models.
