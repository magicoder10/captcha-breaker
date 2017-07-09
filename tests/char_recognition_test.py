import json
import numpy as np
import cv2
from keras.models import model_from_json

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


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
examples = [(raw_example['digit_images'], list(raw_example['label'])) for raw_example in raw_examples]

X = []
Y = []

for example in examples:
    for char_image in example[0]:
        X.append(cv2.imread('../%s' % char_image))
    for label in example[1]:
        prediction = np.zeros(len(classes))
        prediction[classes.index(label)] = 1
        Y.append(prediction)

X = np.array(X)
Y = np.array(Y)

predictions = model.predict(X)
predicted_classes = [classes[np.argmax(prediction)] for prediction in predictions]
target_classes = [classes[np.argmax(y)] for y in Y]

print('Prediction : ', predicted_classes)
print()
print('Target     : ', target_classes)

evaluation = model.evaluate(X, Y, batch_size=32, verbose=False, sample_weight=None)

print()
print('Evaluation : \n  %s => %f\n  %s => %f' % (model.metrics_names[0],
                                                 evaluation[0],
                                                 model.metrics_names[1],
                                                 evaluation[1]))
