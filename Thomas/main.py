import cv2
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
import tqdm
from EdgeDetection import *
from FileManager import *
from ImageProcessing import *

path = "images/image_directory"
directory = "images/image_directory"
save_directory = "images/save_directory"
originals_directory = "images/originals_directory"

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
        if image.endswith(".jpg") or image.endswith(".png"):
            img = cv.imread(directory + "/" + image, cv.IMREAD_GRAYSCALE)
            rescaled_image = ip.rescale_image(img, 64, 64)
            fm.save_image(originals_directory, rescaled_image, "original", index)
            image_edges = cv.Canny(rescaled_image, 100, 200)
            fm.save_image(save_directory, image_edges, "edges", index)
            pixel_groups = ip.count_bw_pixels(image_edges, index)
            group = ip.group_pixels(image_edges)
            print(group)
            print(pixel_groups)
            index += 1
        else:
            raise ValueError("Picture is not of file extension *.jpg or *.png")
