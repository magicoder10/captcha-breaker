import json

import cv2
from PIL import Image

from captcha.preprocessor import get_digits

image_names = [25]
examples = []

for i in range(len(image_names)):
    image = Image.open('../images/%d.png' % image_names[i])
    first_character, second_character, third_character, fourth_character = get_digits(image)
    if first_character is not None:
        captcha_filename = 'images/%d.png' % i
        first_filename = 'digits/%d_%d.png' % (i, 0)
        second_filename = 'digits/%d_%d.png' % (i, 1)
        third_filename = 'digits/%d_%d.png' % (i, 2)
        fourth_filename = 'digits/%d_%d.png' % (i, 3)

        example = {
            'captcha_image': captcha_filename,
            'digit_images': [
                first_filename,
                second_filename,
                third_filename,
                fourth_filename
            ]
        }
        examples.append(example)
        image.save(captcha_filename)
        cv2.imwrite(first_filename, first_character)
        cv2.imwrite(second_filename, second_character)
        cv2.imwrite(third_filename, third_character)
        cv2.imwrite(fourth_filename, fourth_character)