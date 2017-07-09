import json

import numpy as np

import cv2
from keras.models import model_from_json

from captcha.captcha_recognizer import read_captcha


def get_model(training_set_size):
    with open('../models/%d.model.json' % training_set_size) as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights('../models/%d.weights.hdf5' % training_set_size)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


model = get_model(160)

with open('../examples/40.examples.json') as example_file:
    examples_json = example_file.read()
raw_examples = json.loads(examples_json)
examples = [(raw_example['captcha_image'], raw_example['label']) for raw_example in raw_examples]

X = []
Y = []

for example in examples:
    X.append(cv2.imread('../%s' % example[0]))
    Y.append(example[1])

captcha_list = [read_captcha(model, x) for x in X]

print('Expected     :', Y)
print('Predicted    :', captcha_list)

values, counts = np.unique([tuple[0] == tuple[1] for tuple in zip(captcha_list, Y)], return_counts=True)
correct_count = counts[np.nonzero(values)[0]]
print('Accuracy     :', correct_count / len(Y))
