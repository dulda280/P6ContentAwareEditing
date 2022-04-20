import os
import sys
import cv2 as cv
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


class FileManager:

    def save_image(self, path, image, keyword, index):
        cv.imwrite(os.path.join(path, str(index) + "_" + str(keyword) + ".jpg"), image)

    def save_image_nokeyword(self, path, image, index):
        cv.imwrite(os.path.join(path, str(index) + ".jpg"), image)

    def save_array(self, path, image, keyword, index):
        with open(os.path.join(path, str(index) + "_" + str(keyword) + ".txt"), 'w') as file:
            file.write(str(image))

    def save_text_file(self, path, value):
        with open(os.path.join(path, "Gamma_Values.txt"), "a+") as file:
            file.write(str(value) + "\n")