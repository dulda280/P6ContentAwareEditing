import cv2

import numpy as np
import argparse
import keyboard

import os
parser = argparse.ArgumentParser(description='hej')
parser.add_argument('--input', help='500Billeder/', default='Billeder/IMG_2121.jpg')
args = parser.parse_args()
image = cv2.imread(cv2.samples.findFile(args.input))


# new_image = np.zeros(image.shape, image.dtype)


def rescale_image(image, res_x, res_y):
    rescale_dimensions = (res_y, res_x)
    rescaled_image = cv2.resize(image, rescale_dimensions, interpolation=cv2.INTER_AREA)
    # rescaled_image = cv.cvtColor(rescaled_image, cv.COLOR_BGR2RGB)
    return rescaled_image


def adjust_gamma(image, gamma=1.1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.9) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


x = int(image.shape[0] / 4)
y = int(image.shape[1] / 4)

new_image = rescale_image(image, x, y)
gamma_image = adjust_gamma(new_image, gamma=1.1)

cv2.imshow('OG rescaled', new_image)
cv2.imshow('gamma', gamma_image)

# Wait until user press some key
cv2.waitKey()
cv2.imwrite('C:/Users/krell/PycharmProjects/P6ContentAwareEditing/KristianNN/redigeret/115.jpg', gamma_image)

print('hej')


