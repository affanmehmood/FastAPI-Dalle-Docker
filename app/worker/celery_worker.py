import glob
import os
import time
from app.DALLE_DOCKER.infer import main
from celery import current_task
from time import sleep
import shutil

from .celery_app import celery_app


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    for trash in [name for name in os.listdir('/app/dalle_tmp/')]:
        shutil.rmtree('/app/dalle_tmp/' + trash)

    print('GEN Started')
    current_task.update_state(state='GENERATING',
                              meta={'Status': 'Dalle Running'})

    main(word, outputs_dir='/app/dalle_tmp/' + test_celery.request.id + '/')

    return f"Generation completed for {word}"
