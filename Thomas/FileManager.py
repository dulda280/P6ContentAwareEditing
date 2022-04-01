import os
import cv2 as cv


class FileManager:

    def save_image(self, path, image, keyword, index):
        cv.imwrite(os.path.join(path, str(index) + "_" + str(keyword) + ".jpg"), image)
