import glob
import os
import time
from app.DALLE_DOCKER.infer import main
from celery import current_task
from time import sleep

from .celery_app import celery_app


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    for path in ['/app/dalle_tmp/*', '/app/stable_tmp/*']:
        files = glob.glob(path)
        for f in files:
            os.remove(f)
    print('GEN Started')
    current_task.update_state(state='GENERATING',
                              meta={'Status': 'Dalle Running'})
    if 'task_id' in current_task:
        print('Task id inside')
        print(current_task.task_id)
    else:
        print('task id not')
    # print(current_task.task_id)

    # main(word, outputs_dir='/app/dalle_tmp/', task_id=current_task)
    return f"Generation completed {word}"
