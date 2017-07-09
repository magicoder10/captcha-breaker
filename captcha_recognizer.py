import cv2
import numpy as np

from captcha.preprocessor import get_digits
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def read_captcha(model, image):
    first_character, second_character, third_character, fourth_character = get_digits(image)
    first_character = cv2.cvtColor(first_character, cv2.COLOR_GRAY2BGR)
    second_character = cv2.cvtColor(second_character, cv2.COLOR_GRAY2BGR)
    third_character = cv2.cvtColor(third_character, cv2.COLOR_GRAY2BGR)
    fourth_character = cv2.cvtColor(fourth_character, cv2.COLOR_GRAY2BGR)
    if first_character is not None:
        predictions = model.predict(np.array([first_character, second_character, third_character, fourth_character]))
        predicted_classes = [classes[np.argmax(prediction)] for prediction in predictions]
        return ''.join(predicted_classes)
    return ''
