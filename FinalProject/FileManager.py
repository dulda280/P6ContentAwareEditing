import os
import sys
import cv2 as cv
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


class FileManager:

    def save_image(self, path, image, keyword, index):
        cv.imwrite(os.path.join(path, str(index) + "_" + str(keyword) + ".jpg"), image)

    def save_array(self, path, image, keyword, index):
        with open(os.path.join(path, str(index) + "_" + str(keyword) + ".txt"), 'w') as file:
            file.write(str(image))