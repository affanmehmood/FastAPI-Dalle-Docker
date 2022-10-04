import glob
import os
import time
from app.DALLE_DOCKER.infer import main
from celery import current_task
from time import sleep
import shutil

from .celery_app import celery_app


def update_running_state_dalle(state):
    json_file = {
        'stable': False,
        'dalle': state
    }

    with open("mutex.json", "w") as jsonFile:
        json.dump(json_file, jsonFile)


def check_stable_running_state():
    with open("/app/mutex.json", "r") as jsonFile:
        data = json.load(jsonFile)
    return data['stable']


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    while check_stable_running_state():
        sleep(2)

    update_running_state_dalle(True)

    print('GEN Started', word, test_celery.request.id)

    current_task.update_state(state='GENERATING',
                              meta={'Status': 'Dalle Running'})

    main(word, outputs_dir='/app/dalle_tmp/' + test_celery.request.id + '/')
    update_running_state_dalle(False)
    return f"Generation completed for {word}"
