import json

from keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Conv2D
from keras.models import Sequential, model_from_json
import cv2
import numpy as np

image_width = 50
image_height = 50
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

model = Sequential()
model.add(
    Conv2D(50, (5, 5), activation='relu', batch_input_shape=(None, image_height, image_width, 3),
           data_format="channels_last", name='conv_1'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='max_pool_1'))
model.add(Conv2D(25, (3, 3), activation='relu', data_format='channels_last', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='max_pool_2'))
model.add(Dropout(0.2, name='dropout_1'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(128, activation='relu', name='dense_1'))
model.add(Dense(50, activation='relu', name='dense_2'))
model.add(Dense(len(classes), activation='softmax', name='prediction'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

with open('examples/300.examples.json') as example_file:
    examples_json = example_file.read()
raw_examples = json.loads(examples_json)
examples = [(raw_example['digit_images'], list(raw_example['label'])) for raw_example in raw_examples]

X = []
Y = []

for example in examples:
    for char_image in example[0]:
        X.append(cv2.imread(char_image))
    for label in example[1]:
        prediction = np.zeros(len(classes))
        prediction[classes.index(label)] = 1
        Y.append(prediction)

X = np.array(X) / 255.0
Y = np.array(Y)

num_examples = len(X)
num_examples_train = int(num_examples * 0.8)

X_train = X[:num_examples_train]
Y_train = Y[:num_examples_train]
X_test = X[num_examples_train:]
Y_test = Y[num_examples_train:]

history_callback = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, epochs=50, verbose=2,
                             initial_epoch=0)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model_json = model.to_json()
with open('models/%d.model.json' % len(X), 'w') as model_json_file:
    model_json_file.write(model_json)
model.save_weights('models/%d.weights.hdf5' % len(X))
print('Saved model for training set %d' % len(X))

loss_history = history_callback.history['loss']
np.savetxt('logs/%d.log.txt' % len(X), np.array(loss_history))
