import base64
import json

from io import BytesIO
from PIL import Image
from urllib import request

from captcha.captcha_recognizer import read_captcha
from captcha.helpers import get_model, get_project_root

model = get_model(1188)

all_solutions = []


def solve_captcha(images):
    solutions = [{'solution': read_captcha(model, Image.open(BytesIO(base64.b64decode(image['jpg_base64'])))),
                  'name': image['name']} for image in images]

    clean_solutions = [solution for solution in solutions if len(solution['solution']) == 4]

    for solution in clean_solutions:
        all_solutions.append(solution)
    return len(clean_solutions)


print('Solving CAPTCHAS')

count = 0
while count < 14800:
    images_json = request.urlopen('https://captcha.delorean.codes/u/byliuyang/challenge').read()
    images = json.loads(images_json)['images']

    num_saved = solve_captcha(images)
    count += num_saved
    print('Solved %d/%d captcha' % (count, 15000))

solution_json = json.dumps({'solutions': all_solutions})
with open('%s/solution.json' % get_project_root(), 'w') as solutions_json_file:
    solutions_json_file.write(solution_json)
