import cv2
import numpy as np

from captcha.data import classes
from captcha.preprocessor import get_characters


def read_captcha(model, image):
    first_character, second_character, third_character, fourth_character = get_characters(image)
    if first_character is not None:
        first_character = cv2.cvtColor(first_character, cv2.COLOR_GRAY2BGR)
        second_character = cv2.cvtColor(second_character, cv2.COLOR_GRAY2BGR)
        third_character = cv2.cvtColor(third_character, cv2.COLOR_GRAY2BGR)
        fourth_character = cv2.cvtColor(fourth_character, cv2.COLOR_GRAY2BGR)
        predictions = model.predict(np.array(
            [first_character / 255.0, second_character / 255.0, third_character / 255.0, fourth_character / 255.0]))
        predicted_classes = [classes[np.argmax(prediction)] for prediction in predictions]
        return ''.join(predicted_classes)
    return ''
