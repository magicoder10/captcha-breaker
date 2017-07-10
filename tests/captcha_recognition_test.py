import json

import numpy as np

import cv2
from PIL import Image
from keras.models import model_from_json

from captcha.captcha_recognizer import read_captcha
from captcha.helpers import get_project_root, get_model

with open('%s/examples/%d.examples.json' % (get_project_root(), 330)) as example_file:
    examples_json = example_file.read()
raw_examples = json.loads(examples_json)
examples = [(raw_example['captcha_image'], raw_example['label']) for raw_example in raw_examples]

model = get_model(1188)

X = []
Y = []

for example in examples:
    X.append(Image.open(example[0]))
    Y.append(example[1])

num_examples = len(X)
num_examples_train = int(num_examples * 0.9)

Y = np.array(Y)

X_train = X[:num_examples_train]
Y_train = Y[:num_examples_train]
X_test = X[num_examples_train:]
Y_test = Y[num_examples_train:]

# print(Image.fromarray(np.array(X_train * 255, dtype=np.int8)))
predictions = [read_captcha(model, x) for x in X_test]


print('Expected     :', Y_test)
print('Predicted    :', predictions)

values, counts = np.unique([tuple[0] == tuple[1] for tuple in zip(predictions, Y_test)], return_counts=True)
correct_count = counts[np.nonzero(values)[0]]
print('Accuracy     :', correct_count / len(Y_test))
