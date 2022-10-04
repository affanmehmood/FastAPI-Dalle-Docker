import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
# import wandb
from app.DALLE_DOCKER.gen_call import main as generate_img

import torch

torch.multiprocessing.set_start_method('forkserver')

paths = ['/app/dalle_tmp/', '/app/stable_tmp/']
for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)


# run = wandb.init(anonymous="must")
# artifact_uri = 'img2dataset'  # @param ["img2dataset", "kaggle dataset"] {allow-input: false}
# if artifact_uri == 'img2dataset':
#     artifact_uri = 'divedeepai/img2dataset_train_transformer/trained-dalle:latest'
# else:
#     artifact_uri = "divedeepai/dalle_train_transformer/trained-dalle:v51"
# artifact = run.use_artifact(artifact_uri, type='model')
# artifact_dir = artifact.download(root='/output/')

# @markdown # **3** Try out the model. @markdown #### Results will be saved in the outputs directory. Refresh (right
# click the folder -> refresh) if you dont see the result inside the folder.


# input_img_path = "/input/img.jpg"
# !python /content/DALLE-pytorch/generate.py --dalle_path=$checkpoint_path --text="$text" --num_images=$num_images
# --batch_size=$batch_size --outputs_dir="$_folder" ; wait;


if __name__ == '__main__':
    def main(text, outputs_dir='/app/dalle_tmp/', batch_size=1, num_images=1,
             checkpoint_path="/app/DALLE_DOCKER/models/dalle.pt"):
        generate_img(dalle_path=checkpoint_path, text=text, num_images=num_images,
                     batch_size=batch_size, outputs_dir=outputs_dir)
