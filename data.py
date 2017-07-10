import base64
import json

from io import BytesIO

import cv2
from PIL import Image
import numpy as np

from captcha.helpers import get_project_root
from captcha.preprocessor import clean_up

project_root = get_project_root()

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def load_images(i):
    with open('%s/challenges/challenge%d.json' % (project_root, i)) as file:
        images_json = file.read()

    images = json.loads(images_json)['images']
    base64_images = [image_info['jpg_base64'] for image_info in images]
    images = [Image.open(BytesIO(base64.b64decode(base64_image))) for base64_image in base64_images]
    for index in range(len(images)):
        images[index].save('%s/images/captchas/%d/%d.png' % (project_root, i, index))
    return images


def get_clean_images(i):
    images = load_images(i)
    clean_images = [Image.fromarray(clean_up(image)) for image in images]
    for index in range(len(clean_images)):
        clean_images[index].save('%s/images/clean-captchas/%d/%d.png' % (project_root, i, index))
    return images, clean_images


def get_example_data(i, percent_training):
    with open('%s/examples/%d.examples.json' % (project_root, i)) as example_file:
        examples_json = example_file.read()
    raw_examples = json.loads(examples_json)
    examples = [(raw_example['char_images'], list(raw_example['label'])) for raw_example in raw_examples]

    X = []
    Y = []

    for example in examples:
        for char_image in example[0]:
            X.append(cv2.imread(char_image))
        for label in example[1]:
            prediction = np.zeros(len(classes))
            prediction[classes.index(label)] = 1
            Y.append(prediction)

    num_examples = len(X)
    num_examples_train = int(num_examples * percent_training)

    X = np.array(X) / 255.0
    Y = np.array(Y)

    X_train = X[:num_examples_train]
    Y_train = Y[:num_examples_train]
    X_test = X[num_examples_train:]
    Y_test = Y[num_examples_train:]

    return X_train, Y_train, X_test, Y_test