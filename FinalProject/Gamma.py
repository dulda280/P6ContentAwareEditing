from __future__ import print_function
from builtins import input

import cv2
import cv2 as cv
import numpy as np

import os
# parser = argparse.ArgumentParser(description='hej')
# parser.add_argument('--input', help='500Billeder/', default='dataset/DSC00778.jpg')
# args = parser.parse_args()
# image = cv.imread(cv.samples.findFile(args.input))


# new_image = np.zeros(image.shape, image.dtype)

def rescale_image(image, res_x, res_y):
    rescale_dimensions = (res_y, res_x)
    rescaled_image = cv.resize(image, rescale_dimensions, interpolation=cv.INTER_AREA)
    # rescaled_image = cv.cvtColor(rescaled_image, cv.COLOR_BGR2RGB)
    return rescaled_image


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.9) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)


if __name__ == "__main__":
    doGammaStuff = False
    if doGammaStuff:
        x = int(image.shape[0] / 4)
        y = int(image.shape[1] / 4)

        new_image = rescale_image(image, x, y)
        gamma_image = adjust_gamma(new_image, gamma=1.2)

        cv.imshow('OG rescaled', new_image)
        cv.imshow('gamma', gamma_image)

        # Wait until user press some key
        cv.waitKey()
        # cv.imwrite('C:/Users/krell/PycharmProjects/Convulutionalneurnalenrtnenrewo/Redigeret/brightness.jpg', new_image)
        cv.imwrite('C:\\Users\\sebbe\\Desktop\\MED-local\\P6ContentAwareEditing\\trainingData\\trainedImage1.jpg',
                   gamma_image)
        print('hej')

    path = os.getcwd()
    path = path + '\\data'
    print('her er path: ', path)
    pathDir = os.listdir(path)
    print('her er directory: ', pathDir)
    print("hvad:_ ", path + "\\" + pathDir[1])
    imgP = str(path + "\\" + pathDir[59])


    image = cv2.imread(imgP)

    new_image = rescale_image(image, int(image.shape[0]*3/4), int(image.shape[1]*3/4))
    gamma_image = adjust_gamma(new_image, gamma=1.)

    cv.imshow(f'{pathDir[3]}', new_image)
    cv.imshow('gamma', gamma_image)
    cv.waitKey(0)
    # Wait until user press some key

    # cv.imwrite('C:/Users/krell/PycharmProjects/Convulutionalneurnalenrtnenrewo/Redigeret/brightness.jpg', new_image)
    cv.imwrite('C:\\Users\\sebbe\\Desktop\\MED-local\\P6ContentAwareEditing\\Gamma og resize\\correctedData\\corrected60.jpg', gamma_image)

