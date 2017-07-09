import base64
import json

from io import BytesIO

from PIL import Image
from keras.models import model_from_json

from captcha.captcha_recognizer import read_captcha


def get_model(training_set_size):
    with open('models/%d.model.json' % training_set_size) as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights('models/%d.weights.hdf5' % training_set_size)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


model = get_model(160)


def solve_captcha(image_file, solution_file):
    print('Solving %s' % image_file)
    with open(image_file) as file:
        images_json = file.read()

    images = json.loads(images_json)['images']

    solutions = [{'solution': read_captcha(model, Image.open(BytesIO(base64.b64decode(image['jpg_base64'])))),
                  'name': image['name']} for image in images]

    clean_solutions = [solution for solution in solutions if len(solution['solution']) == 4]

    solutions = {
        'solutions': clean_solutions
    }

    solution_json = json.dumps(solutions)
    with open(solution_file, 'w') as solutions_json_file:
        solutions_json_file.write(solution_json)
        return len(solutions['solutions'])

count = 11259
for i in range(20, 26):
    num_saved = solve_captcha('challenge%d.json' % i, 'solution%d.json' % i)
    count += num_saved
    print('Saved %d/%d solutions to solution_file' % (num_saved, count))
