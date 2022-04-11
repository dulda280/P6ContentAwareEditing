import math

import cv2
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
import tqdm
from EdgeDetection import *
from FileManager import *
from ImageProcessing import *
from collections import Counter

# Paths for directories
path = "images/image_directory"
directory = "images/image_directory"
originals_directory = "images/originals_directory"
save_directory = "images/save_directory"
cca_directory = "images/cca_directory"
image_arrays_directory = "images/image_arrays"

ed = EdgeDetection()
fm = FileManager()
ip = ImageProcessing()

"""""
img = cv.imread('The_Photo.jpg', cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 100, 200)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
"""""

if __name__ == '__main__':
    index = 1
    for image in os.listdir(directory):
        if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg"):
            img = cv.imread(directory + "/" + image, cv.IMREAD_COLOR)
            rescaled_image = ip.rescale_image(img, 64, 64)
            fm.save_image(originals_directory, rescaled_image, "original", index)

            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            resized = ip.rescale_image(imgGray, 256, 256)
            fm.save_image(save_directory, resized, "resized", index)
            gaussian = cv2.GaussianBlur(imgGray, (3, 3), cv2.BORDER_DEFAULT)
            image_edges = cv.Canny(gaussian, 127, 255)
            reshaped_edges = image_edges.reshape(1, -1)
            fm.save_array(image_arrays_directory, reshaped_edges, "array", index)
            fm.save_image(save_directory, image_edges, "edges", index)

            print("=======================================================================")
            pixel_groups = ip.count_bw_pixels(image_edges, index)
            cca, num_edge_groups = ip.connected_component_labelling(image_edges)
            # cca_cleaned = ip.find_all_pixels_ignore_black(cca)

            fm.save_array(image_arrays_directory, cca, "cca_array", index)
            # fm.save_array(image_arrays_directory, cca_cleaned, "cca_array_cleaned", index)
            fm.save_image(cca_directory, cca, "cca", index)
            # fm.save_image(cca_directory, cca_cleaned, "cca_cleaned", index)

            contours, hierarchy = cv2.findContours(image_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=len)
            largest_connected_edge = len(contour)

            contourImg = cv2.drawContours(cca, contour, -1, (0, 0, 255), 1)
            fm.save_image(cca_directory, contourImg, "contour", index)
            print(pixel_groups)
            print("Number of Edge Groups: " + str(num_edge_groups))
            print("Largest Connected Edge (px): " + str(largest_connected_edge))
            dom_color, hex = ip.most_frequent_color(cca)
            pixels = np.sum(np.all(cca == [round(dom_color[0]), round(dom_color[1]), round(dom_color[2])], axis=2))
            print("R:", round(dom_color[0]), " || " "G:", round(dom_color[1]), " || " "B:", round(dom_color[2]))
            print("Pixel occurrences of RGB:", pixels)
            unique, counts = np.unique(cca, return_counts=True)
            print(dict(zip(unique, counts)))
            index += 1
        else:
            raise ValueError("Picture is not of file extension *.jpg or *.png")
