import os
from os import listdir
import cv2 as cv
import numpy as np
from scipy.ndimage import convolve


class ImageProcessing:
    folder_dir = "car_pictures"  # image path/directory
    original_img = []
    resized_img = []
    scale_percent = 20  # percent of original size

    def import_images(self):
        for img in os.listdir(self.folder_dir):
            img = cv.imread(os.path.join(self.folder_dir, img))
            if img is not None:
                self.original_img.append(img)

        return self.original_img

    def rescale_images(self, print_debug: bool):
        for img in self.original_img:
            if print_debug:
                print('Original Dimensions : ', img.shape)

            # calculate new dimensions
            width = int(img.shape[1] * self.scale_percent / 100)
            height = int(img.shape[0] * self.scale_percent / 100)
            dimensions = (400, 200)

            # resize image
            img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.reshape((img.shape[1] * img.shape[0], 3))
            print(img)
            self.resized_img.append(img)
            if print_debug:
                print('Resized Dimensions : ', img.shape)

            # show image
            # cv.imshow("Resized image", img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        return self.resized_img

    def rescale_image(self, image, res_x, res_y):
        rescale_dimensions = (res_y, res_x)
        rescaled_image = cv.resize(image, rescale_dimensions, interpolation=cv.INTER_AREA)
        rescaled_image = cv.cvtColor(rescaled_image, cv.COLOR_BGR2RGB)
        return rescaled_image

    def count_bw_pixels(self, image, index):
        number_of_white_pix = np.sum(image == 255)
        number_of_black_pix = np.sum(image == 0)
        return f"Picture_{index} (  {number_of_white_pix} White pixels  ||  {number_of_black_pix} Black pixels  )"

    def group_pixels(self, image):
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
        convolve(image, kernel)
        np.where(convolve(image, kernel) < 0, 1, 0)
        group = np.sum(np.where(convolve(image, kernel) < 0, 1, 0))
        return group
