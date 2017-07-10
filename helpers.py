import os

from keras.models import model_from_json


def get_project_root():
    return os.path.dirname(os.path.realpath(__file__))

def get_model(training_set_size):
    with open('%s/models/%d.model.json' % (get_project_root(), training_set_size)) as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights('%s/models/%d.weights.hdf5' % (get_project_root(), training_set_size))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model