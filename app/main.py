import logging
from celery.result import AsyncResult
from fastapi import FastAPI, BackgroundTasks, Response
from worker.celery_app import celery_app
import os
import zipfile
import json
import io
from os import listdir
from os.path import isfile, join

log = logging.getLogger(__name__)
app = FastAPI()


def zipfiles(filenames):
    zip_filename = "archive.zip"

    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for fpath in filenames:
        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)

        # Add file, at correct path
        zf.write(fpath, fname)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp


def celery_on_message(body):
    log.warn(body)


def background_on_message(task):
    log.warn(task.get(on_message=celery_on_message, propagate=False))


@app.get("/{word}/{ddim_eta}/{n_samples}/{n_iter}/{scale}/{ddim_steps}/{strength}")
async def root(word: str, ddim_eta: float, n_samples: int, n_iter: int,
               scale: float, ddim_steps: int, strength: float, background_task: BackgroundTasks):
    ddim_eta = ddim_eta if ddim_eta else 0.0
    n_samples = n_samples if n_samples else 5
    n_iter = n_iter if n_iter else 1
    scale = scale if scale else 5.0
    ddim_steps = ddim_steps if ddim_steps else 200
    strength = strength if strength else 0.50
    stored_params = {
        'stable': {
            'n_iter': n_iter,
            'ddim_eta': ddim_eta,
            'n_samples': n_samples,
            'scale': scale,
            'ddim_steps': ddim_steps,
            'strength': strength
        },
        'dalle': {}
    }

    # set correct task name based on the way you run the example
    if not bool(os.getenv('DOCKER')):
        task_name = "app.worker.celery_worker.test_celery"
    else:
        task_name = "app.app.worker.celery_worker.test_celery"

    task = celery_app.send_task(task_name, args=[word])
    background_task.add_task(background_on_message, task)

    with open("app/param_tmp/{}.json".format(task.task_id), "w") as jsonFile:
        json.dump(stored_params, jsonFile)

    return {"task_id": task.task_id}


@app.get("/get_progress/{task_id}")
async def get_status(task_id: str):
    res = AsyncResult(task_id)

    if os.path.isfile('/app/stable_tmp/{}/dalle.png'.format(task_id)):
        stable_files = [join('/app/stable_tmp/{}/'.format(task_id), f)
                        for f in listdir('/app/stable_tmp/{}/'.format(task_id))
                        if isfile(join('/app/stable_tmp/{}/'.format(task_id), f))]
        return zipfiles(stable_files)
    elif res.state == 'SUCCESS':
        return {"progress": {
            'state': 'Running Stable Diffusion',
            'meta': 'Generating'
        }}
    else:
        return {"progress": {
            'state': res.state,
            'meta': res.result
        }}
