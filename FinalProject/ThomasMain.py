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

class ThomasMain:
    # import images

    def __init__(self, images):
        self.images = images

    ed = EdgeDetection()
    fm = FileManager()
    ip = ImageProcessing()

    largest_edge = []
    number_of_edges = []

    def main(self, verbose: bool):
        # Paths for directories
        cca_directory = "images/cca_directory"
        image_arrays_directory = "images/image_arrays"
        save_directory = "images/save_directory"

        index = 1
        for img in self.images:
            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gaussian = cv.GaussianBlur(imgGray, (3, 3), cv.BORDER_DEFAULT)
            image_edges = cv.Canny(gaussian, 127, 255)
            # self.fm.save_image(save_directory, image_edges, "edges", index)

            pixel_groups = self.ip.count_bw_pixels(image_edges, index)
            cca, num_edge_groups = self.ip.connected_component_labelling(image_edges)
            # self.fm.save_array(image_arrays_directory, cca, "cca_array", index)
            # self.fm.save_image(cca_directory, cca, "cca", index)

            contours, hierarchy = cv.findContours(image_edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # finder kanter
            if len(contours) != 0:
                contour = max(contours, key=len)  # længste kant
            largest_connected_edge = len(contour)

            contourImg = cv.drawContours(cca, contour, -1, (0, 0, 255), 2)
            # self.fm.save_image(cca_directory, contourImg, "contour", index)

            if verbose:
                print("=======================================================================")
                print(pixel_groups)
                print("Number of Edge Groups: " + str(num_edge_groups))
                print("Largest Connected Edge (px): " + str(largest_connected_edge))
            dom_color, hex = self.ip.most_frequent_color(cca)
            pixels = np.sum(np.all(cca == [round(dom_color[0]), round(dom_color[1]), round(dom_color[2])], axis=2))
            if verbose:
                print("R:", round(dom_color[0]), " || " "G:", round(dom_color[1]), " || " "B:", round(dom_color[2]))
                print("Pixel occurrences of RGB:", pixels)
            unique, counts = np.unique(cca, return_counts=True)
            if verbose:
                print(dict(zip(unique, counts)))
            index += 1

            self.largest_edge.append([largest_connected_edge])
            self.number_of_edges.append([num_edge_groups])

        if verbose:
            print("*************************************************")
            print("Number of edges list", self.number_of_edges)
            print("Largest edges list", self.largest_edge)
            print("*************************************************")
        return self.largest_edge, self.number_of_edges
