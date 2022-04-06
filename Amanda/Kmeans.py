import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from sklearn.cluster import KMeans

from Downsampling import *

# Colors: https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a
# Edges:  https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

# HSV: 360, 180, 180


class Kmeans:
    # Import images
    images = Downsampling()
    data = images.BGR2HSV()

    def clustering(self):
        # Number of clusters
        clusters = KMeans(n_clusters=5)

        # Making the clustering process illumination invariant by setting saturation and value to 180.
        for img in self.data:

            # Making the clustering process illumination invariant by setting saturation and value to 180.
            img[:, :, 1] = 180
            img[:, :, 2] = 180
            # print("Image shape:", img.shape)
            # print("Image hue:", img)
            # cv.imshow("Max sat and val", img)
            # cv.waitKey(0)

            # Perform clustering
            clusters.fit(img.reshape(-1, 3))

            # Create palette with clusters
            width = 300
            palette = np.zeros((50, width, 3), np.uint8)
            steps = width / clusters.cluster_centers_.shape[0]

            for idx, centers in enumerate(clusters.cluster_centers_):
                palette[:, int(idx * steps):(int((idx + 1) * steps)), :] = centers

            print("Cluster centers: \n", clusters.cluster_centers_)

            # Show image and corresponding palette
            img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            palette = cv.cvtColor(palette, cv.COLOR_HSV2BGR)
            cv.imshow("Image BGR", img)
            cv.waitKey(0)
            cv.imshow("Palette", palette)
            cv.waitKey(0)
            cv.destroyAllWindows()

