import glob
import os
import time
from app.DALLE_DOCKER.infer import main
from celery import current_task
from time import sleep

from .celery_app import celery_app


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    files = glob.glob('/app/dalle_tmp/*')
    for f in files:
        os.remove(f)
    print('GEN Started')
    current_task.update_state(state='GENERATING',
                              meta={'Status': 'Dalle Running'})

    main(word, outputs_dir='/app/dalle_tmp/')
    return f"Generation completed {word}"
