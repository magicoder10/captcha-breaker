import cv2
import numpy as np


def show_contour(white_image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(white_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def get_rect_info(contour):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    return {'counter': contour, 'left': x, 'width': w, 'area': area}


def get_character_contours_info(contours, num_digits, min_area):
    contours_info = [get_rect_info(contour) for contour in contours]
    contours_info.sort(key=lambda contour_info: contour_info['left'])

    contours_info = [contour_info for contour_info in contours_info if contour_info['area'] > min_area]
    non_overlapping_contours = []
    size = len(contours_info)
    min_left = 0
    for i in range(len(contours_info)):
        contour_info = contours_info[i]
        if contour_info['left'] > min_left:
            non_overlapping_contours.append(contour_info)
            min_left = contour_info['left'] + contour_info['width']
        elif contour_info['left'] == min_left and contour_info['area'] > min_area + 5 and len(
                non_overlapping_contours) + size > num_digits:
            last_contour_info = non_overlapping_contours.pop()
            last_contour_info['width'] += contour_info['width']
            last_contour_info['area'] += contour_info['area']
            non_overlapping_contours.append(last_contour_info)
            min_left = contour_info['left'] + contour_info['width']
        size -= 1

    return non_overlapping_contours


def crop(image, left, width, top, height):
    return image[top:top + height, left:left + width]


def crop_character(image, character_contours_info, index):
    return image[0: image.shape[0], character_contours_info[index]['left']:character_contours_info[index]['left'] +
                                                                           character_contours_info[index]['width']]


def resize(width, height, image):
    black_image = np.zeros([height, width], dtype=np.uint8)
    black_image.fill(0)
    x_offset = int((black_image.shape[1] - image.shape[1]) / 2)
    y_offset = int((black_image.shape[0] - image.shape[0]) / 2)
    black_image[y_offset: y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    return black_image


def get_digits(image):
    open_cv_image = np.array(image)
    grey_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    grey_image[grey_image <= 200] = 0
    grey_image[grey_image > 200] = 255

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(grey_image, kernel, iterations=1)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

    ret, thresh = cv2.threshold(closing, 200, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    character_contours_info = get_character_contours_info(contours, 4, 18)

    # infos = [(info['left'], info['width'], info['area']) for info in character_contours_info]
    # print(infos)
    # print(len(character_contours_info))

    # white_image = np.zeros([50, 100, 3], dtype=np.uint8)
    # white_image.fill(255)
    #
    # for contour in contours[5:6]:
    #     show_contour(white_image, contour)
    # cv2.imshow('image', white_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if len(character_contours_info) > 3:
        first_character = crop_character(closing, character_contours_info, 0)
        first_character = resize(100, 100, first_character)

        second_character = crop_character(closing, character_contours_info, 1)
        second_character = resize(100, 100, second_character)

        third_character = crop_character(closing, character_contours_info, 2)
        third_character = resize(100, 100, third_character)

        fourth_character = crop_character(closing, character_contours_info, 3)
        fourth_character = resize(100, 100, fourth_character)

        return first_character, second_character, third_character, fourth_character

    return [None, None, None, None]
