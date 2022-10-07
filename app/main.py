import os
import logging
from threading import Thread
from celery.result import AsyncResult
from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.responses import FileResponse
from worker.celery_app import celery_app
import os
import zipfile
import io
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


from os import listdir
from os.path import isfile, join


@app.get("/get_progress/{task_id}/{img_type}")
async def get_status(task_id: str):
    res = AsyncResult(task_id)

    # if os.path.isdir('/app/dalle_tmp/{}/'.format(task_id)):
    #     return FileResponse('/app/dalle_tmp/{}'.format([f for f in listdir('/app/dalle_tmp/{}/'.format(task_id)) if isfile(join('/app/dalle_tmp/{}/'.format(task_id), f))][0]))
    # elif res.state != 'SUCCESS':
    #     return {"progress": {
    #         'state': 'Running DALLE',
    #         'meta': 'Generating'
    #     }}
    # else:
    #     return {"progress": {
    #         'state': res.state,
    #         'meta': res.result
    #     }}

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
