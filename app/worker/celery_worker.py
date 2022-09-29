from time import sleep
from celery import current_task
from .celery_app import celery_app
from app.DALLE_DOCKER.infer import main
import time


@celery_app.task(acks_late=True)
def test_celery(word: str) -> str:
    print('GEN Started')
    current_task.update_state(state='GENERATING',
                              meta={'Duration': 'unknown'})
    main(word, outputs_dir='/app/dalle_tmp/')

    return f"Generation completed {word} {current_task.task_id}"
