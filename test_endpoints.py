import requests
import time
import json

prompts = ['nike air jordan', 'adidas ultraboost', 'Puma T-shirt', 'Gucci hangbag', 'Nike black shorts']
task_ids = []
for prompt in prompts:
    r = requests.get('http://35.208.240.104/{}'.format(prompt))
    print(json.loads(r.text)['task_id'])
    task_ids.append(json.loads(r.text)['task_id'])

time.sleep(2)

while True:
    for task_id in task_ids:
        try:
            r = requests.get('http://35.208.240.104/get_progress/{}'.format(task_id))
            print(json.loads(r.text))
            time.sleep(3)
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)
