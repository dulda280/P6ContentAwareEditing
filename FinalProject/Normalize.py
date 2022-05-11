import numpy as np
from sklearn import preprocessing

from KmeansFeature import *     # Dominant colours
from AverageColors import *     # Average colours
from ThomasMain import *        # Edges x2
from CornerDetection import *   # Corners x2
from HOG import *               # HOG


class Normalize:

    # import features
    def __init__(self, dominantColours, averageColours, largestEdge, edgeGroups, corners, cornerBox, hog):
        self.dominantColours = dominantColours
        self.averageColours = averageColours
        self.edge_groups = edgeGroups
        self.largest_edge = largestEdge
        self.corner_data = corners
        self.cornerBoxed = cornerBox
        self.hog = hog

    def normalise_data(self, data, i):
        (data[i] - np.min(data)) / (np.max(data) - np.min(data))
        return data

    def merge_data(self, img):
        debug = False
        if debug == True:
            print("FEATURES")
            print("dominant colours", self.dominantColours)
            print("average colours", self.averageColours)
            print("edge groups", self.edge_groups)
            print("largest edge", self.largest_edge)
            print("corners", self.corner_data)
            print("cornerBox", self.cornerBoxed)
            print("hog", self.hog)

        # (data – np.min(data)) / (np.max(data) – np.min(data))

        self.largest_edge = (self.largest_edge - np.min(self.largest_edge)) / (
                np.max(self.largest_edge) - np.min(self.largest_edge))
        self.edge_groups = (self.edge_groups - np.min(self.edge_groups)) / (
                np.max(self.edge_groups) - np.min(self.edge_groups))
        self.corner_data = (self.corner_data - np.min(self.corner_data)) / (
                np.max(self.corner_data) - np.min(self.corner_data))
        #self.cornerBoxed = (self.cornerBoxed - np.min(self.cornerBoxed)) / (
        #        np.max(self.cornerBoxed) - np.min(self.cornerBoxed))
        self.dominantColours = (self.dominantColours - np.min(self.dominantColours)) / (
                np.max(self.dominantColours) - np.min(self.dominantColours))
        self.averageColours = (self.averageColours - np.min(self.averageColours)) / (
                np.max(self.averageColours) - np.min(self.averageColours))
        self.hog = (self.hog - np.min(self.hog)) / (
                np.max(self.hog) - np.min(self.hog))

        # data = []
        # Creates a list containing 5 lists, each of 8 items, all set to 0
        w, h = 7, len(img)
        data = [[] for y in range(h)]
        print("Data initially:", data)

        for i in range(0, len(data)):
            data[i].append(self.largest_edge[i][0])
            data[i].append(self.edge_groups[i][0])  # append number of edges
            data[i].append(self.corner_data[i][0])
            data[i].append(self.cornerBoxed[i][0])
            for j in range(0, len(self.dominantColours[i])):
                data[i].append(self.dominantColours[i][j])  # Dominant colours
            for j in range(0, len(self.averageColours[i])):
                data[i].append(self.averageColours[i][j])   # Average colours
            for j in range(0, len(self.hog[i])):
                data[i].append(self.hog[i][j])
        #print("NORMALISED DATA", data[0])
        #print("LENGTH DATA", len(data[0]))
        return data


# Corners i box er allerede normaliseret, siger Sebber
