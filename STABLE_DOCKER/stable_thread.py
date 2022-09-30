from threading import Thread
from infer import main
import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from os import listdir
import shutil
import time
from os.path import isfile, join


class EngineRunner:
    def __init__(self):
        # A list of threads that have been running in the past or are running now
        self.runningStableEngines = []

    # Halts any other thread executing and launches a new one.
    def startStable(self, args):
        self.killStable()

        newInstanceContainer = ThreadWithTrace(target=main, args=args)
        self.runningStableEngines.append(newInstanceContainer)

        newInstanceContainer.start()

    def killStable(self):
        for thread in self.runningStableEngines:
            if thread.is_alive():
                thread.kill()
                thread.join()


class ThreadWithTrace(threading.Thread):
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True


engineRunner = EngineRunner()

if __name__ == "__main__":
    while True:
        if len(os.listdir('/app/dalle_tmp/')) == 0:
            time.sleep(2)
        else:
            time.sleep(1)
            print('Stable Diff Triggered')
            subfolders = [f.path for f in os.scandir('/app/dalle_tmp/') if f.is_dir()]  # yields full path
            task_id = os.path.basename(os.path.normpath(subfolders[0]))
            onlyfiles = [f for f in listdir(subfolders[0]) if isfile(join(subfolders[0], f))]  # yields only filename

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
            if os.path.isfile('/app/dalle_tmp/{}/resized.png'.format(task_id)):
                engineRunner.startStable([])

                main(prompt=onlyfiles[0].split('.')[0], initimg='/app/dalle_tmp/{}/resized.png'.format(task_id),
                     outdir='/app/stable_tmp/', ckpt='/app/STABLE_DOCKER/models/sd-v1-4.ckpt',
                     embedding_path='/app/STABLE_DOCKER/models/embeddings.pt', ddim_eta=0.0,
                     n_samples=1, n_iter=1, scale=10.0, ddim_steps=50, strength=0.55, task_id=task_id)

            if os.path.isfile('/app/stable_tmp/{}.png'.format(task_id)):
                shutil.rmtree('/app/dalle_tmp/{}/'.format(task_id))
                print('Killing process')
                engineRunner.killStable()
