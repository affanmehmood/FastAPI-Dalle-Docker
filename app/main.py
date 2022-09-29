import os
import logging
from threading import Thread
from celery.result import AsyncResult
from fastapi import FastAPI, BackgroundTasks

from worker.celery_app import celery_app


log = logging.getLogger(__name__)

app = FastAPI()


def celery_on_message(body):
    log.warn(body)


def background_on_message(task):
    log.warn(task.get(on_message=celery_on_message, propagate=False))


@app.get("/{word}")
async def root(word: str, background_task: BackgroundTasks):
    # set correct task name based on the way you run the example
    if not bool(os.getenv('DOCKER')):
        task_name = "app.worker.celery_worker.test_celery"
    else:
        task_name = "app.app.worker.celery_worker.test_celery"

    task = celery_app.send_task(task_name, args=[word])
    background_task.add_task(background_on_message, task)

    return {"task_id": task.task_id}


@app.get("/get_progress/{task_id}")
async def get_status(task_id: str):
    res = AsyncResult(task_id)
    print('res', res)
    return {"progress": {
        'state': res.state,
        'meta': res.result
    }}
