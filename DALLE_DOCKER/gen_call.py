from pathlib import Path
from tqdm import tqdm
import os
# torch

import torch

from einops import repeat

# vision imports

from PIL import Image
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import __version__
from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, YttmTokenizer, ChineseTokenizer


def exists(val):
    return val is not None


# tokenizer
def main(dalle_path, text, vqgan_model_path='', vqgan_config_path='', num_images=128, batch_size=4, top_k=0.9,
         outputs_dir='./outputs', bpe_path='', hug=False, chinese=False, taming=False, gentxt=False):

    if exists(bpe_path):
        klass = HugTokenizer if hug else YttmTokenizer
        tokenizer = klass(bpe_path)
    elif chinese:
        tokenizer = ChineseTokenizer()

    # load DALL-E

    dalle_path = Path(dalle_path)

    assert dalle_path.exists(), 'trained DALL-E must exist'
    load_obj = torch.load(str(dalle_path))
    dalle_params, vae_params, weights, vae_class_name, version = load_obj.pop('hparams'), load_obj.pop(
        'vae_params'), load_obj.pop('weights'), load_obj.pop('vae_class_name', None), load_obj.pop('version', None)

    # friendly print


    if exists(version):
        print(f'Loading a model trained with DALLE-pytorch version {version}', flush=True)
    else:
        print(
            'You are loading a model trained on an older version of DALL-E pytorch - it may not be compatible with the most recent version')

    # load VAE

    if taming:
        vae = VQGanVAE(vqgan_model_path, vqgan_config_path)
    elif vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    else:
        vae = OpenAIDiscreteVAE()

    assert not (exists(
        vae_class_name) and vae.__class__.__name__ != vae_class_name), f'you trained DALL-E using {vae_class_name} but are trying to generate with {vae.__class__.__name__} - please make sure you are passing in the correct paths and settings for the VAE to use for generation'

    # reconstitute DALL-E

    dalle = DALLE(vae=vae, **dalle_params).cuda()

    dalle.load_state_dict(weights)

    print(f'Loaded state dict', flush=True)
    # generate images

    # image_size = vae.image_size

    texts = text.split('|')

    for text in texts:
        if gentxt:
            text_tokens, gen_texts = dalle.generate_texts(tokenizer, text=text, filter_thres=top_k)
            text = gen_texts[0]
        else:
            from dalle_pytorch.tokenizer import tokenizer
            print(f'No gentxt else', flush=True)
            print('text=', text, 'dalle.text_seq_len=',dalle.text_seq_len)
            text_tokens = tokenizer.tokenize([text]).cuda()
            print(f'No gentxt else complete', flush=True)

        print(f'Generation complete', flush=True)
        text_tokens = repeat(text_tokens, '() n -> b n', b=num_images)

        outputs = []

        for text_chunk in tqdm(text_tokens.split(batch_size), desc=f'generating images for - {text}'):
            output = dalle.generate_images(text_chunk, filter_thres=top_k)
            outputs.append(output)

        outputs = torch.cat(outputs)

        print(f'off to saving')
        # save all images

        outputs_dir = Path(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(outputs):
            save_image(image, outputs_dir / f'{text}.png', normalize=True)

        print(f'created {num_images} images at "{str(outputs_dir)}"')
    del dalle
    del dalle_params
    del load_obj
