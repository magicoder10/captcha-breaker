import cv2
import numpy as np


def show_contour(white_image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(white_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def get_rect_info(contour):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    return {'contour': contour, 'left': x, 'top': y, 'width': w, 'height': h, 'area': area}


def get_character_contours_info(contours, min_area):
    contours_info = [get_rect_info(contour) for contour in contours]

    contours_info = [contour_info for contour_info in contours_info if min_area < contour_info['area']]
    contours_info.sort(key=lambda contour_info: contour_info['left'])

    non_overlapping_contours = []
    last_contour_info = None
    min_left = 0
    for i in range(len(contours_info)):
        contour_info = contours_info[i]
        if contour_info['left'] > min_left + 3:
            non_overlapping_contours.append(contour_info)
            min_left = contour_info['left'] + contour_info['width']
            last_contour_info = contour_info
        elif last_contour_info is not None and (contour_info['left'] < last_contour_info['left']
                                                or contour_info['top'] < last_contour_info['top']
                                                or contour_info['area'] > last_contour_info['area']):
            return []
    return non_overlapping_contours


def crop(image, left, width, top, height):
    return image[top:top + height, left:left + width]


def crop_character(image, character_contours_info, index):
    top = character_contours_info[index]['top'] - 2
    bottom = character_contours_info[index]['top'] + character_contours_info[index]['height'] + 2
    left = character_contours_info[index]['left'] - 2
    right = character_contours_info[index]['left'] + character_contours_info[index]['width'] + 2
    return image[top: bottom, left: right]


def resize(width, height, image):
    black_image = np.zeros([height, width], dtype=np.uint8)
    black_image.fill(0)
    x_offset = int((black_image.shape[1] - image.shape[1]) / 2)
    y_offset = int((black_image.shape[0] - image.shape[0]) / 2)
    black_image[y_offset: y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    return black_image


def remove_bg(image):
    open_cv_image = np.array(image)
    grey_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    grey_image[grey_image <= 200] = 0
    grey_image[grey_image > 200] = 255
    return grey_image


def remove_lines(image):
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    return cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)


def clean_up(image):
    return remove_lines(remove_bg(image))


def get_characters(image):
    open_cv_image = np.array(image)
    grey_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    grey_image[grey_image <= 200] = 0
    grey_image[grey_image > 200] = 255

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(grey_image, kernel, iterations=1)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

    ret, thresh = cv2.threshold(closing, 200, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    character_contours_info = get_character_contours_info(contours, 11)

    max_area = 155
    example_width = 40
    example_height = 40

    has_combined_chars = len(
        [contour_info for contour_info in character_contours_info if contour_info['area'] > max_area]) > 0

    infos = []
    for contour_info in character_contours_info:
        right = contour_info['left'] + contour_info['width'] + 2
        bottom = contour_info['top'] + contour_info['height'] + 4
        if contour_info['left'] > 2 and right < 100 and contour_info['top'] > 4 and bottom < 50:
            infos.append(contour_info)

    if len(character_contours_info) == 4 and len(infos) == 4 and not has_combined_chars:

        first_character = crop_character(closing, character_contours_info, 0)
        first_character = resize(example_width, example_height, first_character)

        second_character = crop_character(closing, character_contours_info, 1)
        second_character = resize(example_width, example_height, second_character)

        third_character = crop_character(closing, character_contours_info, 2)
        third_character = resize(example_width, example_height, third_character)

        fourth_character = crop_character(closing, character_contours_info, 3)
        fourth_character = resize(example_width, example_height, fourth_character)

        return first_character, second_character, third_character, fourth_character

    return [None, None, None, None]
