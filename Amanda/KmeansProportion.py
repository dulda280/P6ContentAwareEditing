import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from sklearn.cluster import KMeans
from collections import Counter


from Downsampling import *

# Colors: https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a
# Edges:  https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

# HSV range: H 360, S 180, V 180


class KmeansProportion:
    # Import images
    images = Downsampling()
    data = images.BGR2HSV()

    debug = True

    def clustering(self):
        # Number of clusters
        clusters = KMeans(n_clusters=5)

        results = []

        for img in self.data:
            # The hue for each image is stored in a list
            hue_values = []

            # Making the clustering process illumination invariant by setting saturation and value to 180.
            img[:, :, 1] = 180
            img[:, :, 2] = 180

            # Perform clustering
            clt = clusters.fit(img.reshape(-1, 3))
############
            # Create palette with clusters
            width = 300
            palette = np.zeros((50, width, 3), np.uint8)
            n_pixels = len(clusters.labels_)
            counter = Counter(clusters.labels_)  # count how many pixels per cluster
            perc = {}
            for i in counter:
                perc[i] = np.round(counter[i] / n_pixels, 2)
            perc = dict(sorted(perc.items()))

            # for logging purposes
            print(perc)
            print(clusters.cluster_centers_)

            step = 0

            for idx, centers in enumerate(clusters.cluster_centers_):
                palette[:, step:int(step + perc[idx] * width + 1), :] = centers
                step += int(perc[idx] * width + 1)

            # Extracts the hue values
            for color in clusters.cluster_centers_:
                hue_values.append(round(color[0], 2))

            results.append(hue_values)

            if self.debug:
                # Print results
                print("Cluster centers: \n", clusters.cluster_centers_)
                print("Most dominant hue values", results)

                # Show image and corresponding palette
                img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
                palette = cv.cvtColor(palette, cv.COLOR_HSV2BGR)
                cv.imshow("Image BGR", img)
                cv.waitKey(0)
                cv.imshow("Palette", palette)
                cv.waitKey(0)
                cv.destroyAllWindows()

        print("Hue clusters for all images, aka. results:", len(results), results)
        return results

