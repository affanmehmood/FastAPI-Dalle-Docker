"""make variations of inputs image"""
import os
import PIL
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if image.size == (256, 256):
        print(f"loaded inputs image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
        print(f"inputs image size too small resizing to 512x512px")
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.
    else:
        print(f"loaded inputs image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.


def main(prompt, initimg, outdir, model, device, ddim_steps=200, plms=False,
         ddim_eta=0.0, n_iter=1, n_samples=5, scale=5.0, strength=0.55, from_file=None,
         precision="autocast", task_id=''):
    # opt = parser.parse_args()
    #     seed_everything(opt.seed)

    if plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    outpath = os.path.join(outdir, task_id)
    os.makedirs(outpath, exist_ok=True)

    batch_size = n_samples
    if not from_file:
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    init_image = load_img(initimg).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(n_samples * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(n_samples * [prompt])

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, )

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for index, x_sample in enumerate(x_samples):
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(outpath, f"{index}.png"))
                        all_samples.append(x_samples)

    # del model
    # del pl_sd
    # del init_latent
    # del sampler
    # gc.collect()
    # torch.cuda.empty_cache()

    print("Stable Done")

