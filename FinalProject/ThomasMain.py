import cv2 as cv
import numpy as np
import binascii
import scipy
import scipy.misc
import scipy.cluster

from Downsampling import *
from EdgeDetection import *
from FileManager import *
from ImageProcessing import *

# Class for finding the two features:
#   - Antal af forskellige grupper af pixels
#   - Længste kant (antal pixels i længste kant).

class CannyEdge:
    # import images
    img = Downsampling()
    images = img.rescale_images()

    ed = EdgeDetection()
    fm = FileManager()
    ip = ImageProcessing()

    def main(self):

        # Paths for directories
        cca_directory = "images/cca_directory"
        image_arrays_directory = "images/image_arrays"

        index = 1
        for img in self.images:
            # Convert to grayscale
            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Decrease noise
            gaussian = cv.GaussianBlur(imgGray, (3, 3), cv.BORDER_DEFAULT)
            # Detect edges
            image_edges = cv.Canny(gaussian, 127, 255)

            print("=======================================================================")
            pixel_groups = self.ip.count_bw_pixels(image_edges, index)
            cca, num_edge_groups = self.ip.connected_component_labelling(image_edges)
            # cca_cleaned = ip.find_all_pixels_ignore_black(cca)

            self.fm.save_array(image_arrays_directory, cca, "cca_array", index)
            # fm.save_array(image_arrays_directory, cca_cleaned, "cca_array_cleaned", index)
            self.fm.save_image(cca_directory, cca, "cca", index)
            # fm.save_image(cca_directory, cca_cleaned, "cca_cleaned", index)

            contours, hierarchy = cv.findContours(image_edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            contour = max(contours, key=len)
            largest_connected_edge = len(contour)

            contourImg = cv.drawContours(cca, contour, -1, (0, 0, 255), 1)
            self.fm.save_image(cca_directory, contourImg, "contour", index)
            print(pixel_groups)
            print("Number of Edge Groups: " + str(num_edge_groups))
            print("Largest Connected Edge (px): " + str(largest_connected_edge))
            dom_color, hex = self.ip.most_frequent_color(cca)
            pixels = np.sum(
                np.all(cca == [round(dom_color[0]), round(dom_color[1]), round(dom_color[2])], axis=2))
            print("R:", round(dom_color[0]), " || " "G:", round(dom_color[1]), " || " "B:", round(dom_color[2]))
            print("Pixel occurrences of RGB:", pixels)
            unique, counts = np.unique(cca, return_counts=True)
            print(dict(zip(unique, counts)))
            index += 1
