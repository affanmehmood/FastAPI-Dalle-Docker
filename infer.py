import time
import json
from os import listdir
import shutil
from os.path import isfile, join
import os
from PIL import Image
from STABLE_DOCKER.img2img import main
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch


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
    return model


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


def update_running_state_stable(state):
    json_file = {
        'stable': state,
        'dalle': False
    }

    with open("mutex.json", "w") as jsonFile:
        json.dump(json_file, jsonFile)


def check_dalle_running_state():
    if not os.path.exists("/app/mutex.json"):
        return False
    with open("/app/mutex.json", "r") as jsonFile:
        data = json.load(jsonFile)
    return data['dalle']


if __name__ == "__main__":
    config = OmegaConf.load(
        "/app/STABLE_DOCKER/configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, '/app/STABLE_DOCKER/models/sd-v1-4.ckpt')
    model.embedding_manager.load('/app/STABLE_DOCKER/models/embeddings.pt')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    while True:
        if check_dalle_running_state():
            time.sleep(2)
            continue
        if len(os.listdir('/app/dalle_tmp/')) == 0:
            time.sleep(2)
            continue
        else:
            time.sleep(1)

            subfolders = [f.path for f in os.scandir('/app/dalle_tmp/') if f.is_dir()]  # yields full path
            if len(subfolders) < 1:
                print('No folder inside')
                time.sleep(2)
                continue
            task_id = os.path.basename(os.path.normpath(subfolders[0]))
            if isfile('/app/dalle_tmp/{}/{}'.format(task_id, 'resized.png')):
                print('deleting trash')
                shutil.rmtree('/app/dalle_tmp/{}/'.format(task_id))
                continue
            onlyfiles = [f for f in listdir(subfolders[0]) if isfile(join(subfolders[0], f))]  # yields only filename
            print('Stable Diff Triggered', onlyfiles[0].split('.')[0], task_id)
            if len(onlyfiles) < 1:
                print('No file inside ', task_id)
                time.sleep(2)
                continue

            update_running_state_stable(True)
            # resize
            size = (512, 512)
            background_color = "white"
            img = Image.open('/app/dalle_tmp/{}/{}'.format(task_id, onlyfiles[0]))

            # resize the image
            resized_img = np.array(resize_image(img, size, background_color))
            im = Image.fromarray(resized_img)
            im.save('/app/dalle_tmp/{}/resized.png'.format(task_id))
            main(prompt=onlyfiles[0].split('.')[0], initimg='/app/dalle_tmp/{}/resized.png'.format(task_id),
                 outdir='/app/stable_tmp/', model=model, device=device, ddim_eta=0.0,
                 n_samples=1, n_iter=1, scale=10.0, ddim_steps=50, strength=0.55, task_id=task_id)
            update_running_state_stable(False)
            shutil.rmtree('/app/dalle_tmp/{}/'.format(task_id))
            # exit()
