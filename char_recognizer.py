from keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Conv2D
from keras.models import Sequential

from captcha.data import classes, get_example_data

image_width = 40
image_height = 40

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

X_train, Y_train, X_test, Y_test = get_example_data(330, 0.9)

history_callback = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=100, epochs=100, verbose=1,
                             initial_epoch=0)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

model_json = model.to_json()
with open('models/%d.model.json' % len(X_train), 'w') as model_json_file:
    model_json_file.write(model_json)
model.save_weights('models/%d.weights.hdf5' % len(X_train))
print('Saved model for training set %d' % len(X_train))

acc_history = history_callback.history['acc']
loss_history = history_callback.history['loss']
val_acc_history = history_callback.history['val_acc']
val_loss_history = history_callback.history['val_loss']

history = zip(loss_history, acc_history, val_loss_history, val_acc_history)
history_lines = ['loss: %f  acc: %f  val_loss: %f  val_acc: 0%f' % line for line in history]

with open('logs/%d.log.txt' % len(X_train), 'w') as file:
    for line in history_lines:
        file.write("%s\n" % line)
