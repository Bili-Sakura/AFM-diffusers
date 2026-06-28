We provide native diffusers pipelines for CAFM sampling:

- `CAFMSiTPipeline` for SiT-based CAFM checkpoints (latent space + VAE)
- `CAFMJiTPipeline` for JiT-based CAFM checkpoints (pixel space)

Example:

```python
from diffusers.pipelines.cafm import CAFMSiTPipeline
from diffusers.models.cafm.sit.generator import Generator
from diffusers.models.cafm.sit.vae import AutoencoderKLWrapper

generator = Generator(...)
generator.load_state_dict(torch.load("cafm_sit.pth"))
vae = AutoencoderKLWrapper(...)
pipe = CAFMSiTPipeline(generator=generator, vae=vae).to("cuda")
image = pipe(class_labels=207, num_inference_steps=250, sampler="heun").images[0]
```

Or use the YAML configs in this folder with a thin generation entrypoint.
