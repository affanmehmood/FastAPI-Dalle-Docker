"""make variations of inputs image"""
import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from os import listdir
import shutil
from os.path import isfile, join
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import time
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import random
import gc


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    #    if "global_step" in pl_sd:
    #        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model, pl_sd


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


def main(prompt, initimg, outdir, ckpt, embedding_path, ddim_steps=200, plms=False,
         ddim_eta=0.0, n_iter=1, n_samples=2, n_rows=0, scale=5.0, strength=0.55, from_file=None,
         precision="autocast", task_id=''):
    # opt = parser.parse_args()
    #     seed_everything(opt.seed)

    config = OmegaConf.load(
        "/app/STABLE_DOCKER/configs/stable-diffusion/v1-inference.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model, pl_sd = load_model_from_config(config, ckpt)  # TODO: check path
    model.embedding_manager.load(embedding_path)

    # config = OmegaConf.load(f"{opt.config}")
    # model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    if not from_file:
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    init_image = load_img(initimg).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

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

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(outpath, f"{task_id}.png"))
                        all_samples.append(x_samples)

    del model
    del pl_sd
    del init_latent
    del sampler
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


def resize_image(src_img, size=(64, 64), bg_color="white"):
    from PIL import Image

    # rescale the image so the longest edge is the right size
    src_img.thumbnail(size, Image.ANTIALIAS)

    # Create a new image of the right shape
    new_image = Image.new("RGB", size, bg_color)

    # Paste the rescaled image onto the new centered background
    new_image.paste(src_img, (int((size[0] - src_img.size[0]) / 2), int((size[1] - src_img.size[1]) / 2)))

    # return the resized image
    return new_image


if __name__ == "__main__":
    while True:
        if len(os.listdir('/app/dalle_tmp/')) == 0:
            time.sleep(2)
        else:
            time.sleep(1)

            subfolders = [f.path for f in os.scandir('/app/dalle_tmp/') if f.is_dir()]  # yields full path
            task_id = os.path.basename(os.path.normpath(subfolders[0]))
            onlyfiles = [f for f in listdir(subfolders[0]) if isfile(join(subfolders[0], f))]  # yields only filename
            print('Stable Diff Triggered', onlyfiles[0].split('.')[0], task_id)
            if len(onlyfiles) < 1:
                print('No file inside ', task_id)
                time.sleep(2)
                continue
            # resize

            size = (512, 512)
            background_color = "white"
            img = Image.open('/app/dalle_tmp/{}/{}'.format(task_id, onlyfiles[0]))

            # resize the image
            resized_img = np.array(resize_image(img, size, background_color))
            im = Image.fromarray(resized_img)
            im.save('/app/dalle_tmp/{}/resized.png'.format(task_id))
            main(prompt=onlyfiles[0].split('.')[0], initimg='/app/dalle_tmp/{}/resized.png'.format(task_id),
                     outdir='/app/stable_tmp/', ckpt='/app/STABLE_DOCKER/models/sd-v1-4.ckpt',
                     embedding_path='/app/STABLE_DOCKER/models/embeddings.pt', ddim_eta=0.0,
                     n_samples=1, n_iter=1, scale=10.0, ddim_steps=50, strength=0.55, task_id=task_id)

            shutil.rmtree('/app/dalle_tmp/{}/'.format(task_id))
