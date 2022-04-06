from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
import keyboard


parser = argparse.ArgumentParser(description='hej')
parser.add_argument('--input', help='500Billeder/',default='dataset/DSC00778.jpg')
args = parser.parse_args()
image = cv.imread(cv.samples.findFile(args.input))



#new_image = np.zeros(image.shape, image.dtype)


def rescale_image(image, res_x, res_y):
    rescale_dimensions = (res_y, res_x)
    rescaled_image = cv.resize(image, rescale_dimensions, interpolation=cv.INTER_AREA)
    # rescaled_image = cv.cvtColor(rescaled_image, cv.COLOR_BGR2RGB)
    return rescaled_image

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.9) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv.LUT(image,table)

x = int(image.shape[0] / 4)
y = int(image.shape[1] / 4)

new_image = rescale_image(image,x,y)
gamma_image = adjust_gamma(new_image,gamma=1.3)



cv.imshow('OG rescaled', new_image)
cv.imshow('gamma', gamma_image)





# Wait until user press some key
cv.waitKey()
#cv.imwrite('C:/Users/krell/PycharmProjects/Convulutionalneurnalenrtnenrewo/Redigeret/brightness.jpg', new_image)
cv.imwrite('C:/Users/krell/PycharmProjects/Convulutionalneurnalenrtnenrewo/Redigeret/61.jpg', gamma_image)
print('hej')
