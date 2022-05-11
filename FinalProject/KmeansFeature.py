
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


from Downsampling import *

# Colors: https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a
# Edges:  https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

# HSV range: H 360, S 180, V 180


class KmeansFeature:

    results = []

    debug = False
    def __init__(self, images):
        self.data = images

    def clustering(self):
        # Number of clusters
        clusters = KMeans(n_clusters=5)
        index = 0
        #print("Starting clustering loop...")
        for img in self.data:
            # The hue for each image is stored in a list
            hue_values = []

            # Making the clustering process illumination invariant by setting saturation and value to 180.
            img[:, :, 1] = 180
            img[:, :, 2] = 180

            # Perform clustering
            clusters.fit(img.reshape(-1, 3))

            # Create palette with clusters
            width = 300
            palette = np.zeros((50, width, 3), np.uint8)
            steps = width / clusters.cluster_centers_.shape[0]
            for idx, centers in enumerate(clusters.cluster_centers_):
                palette[:, int(idx * steps):(int((idx + 1) * steps)), :] = centers

            for color in clusters.cluster_centers_:
                hue_values.append(round(color[0], 2))

            n_pixels = len(clusters.labels_)
            counter = Counter(clusters.labels_)  # count how many pixels per cluster
            arr = []

            for i in counter:
                arr.append((counter[i], hue_values[i]))

            arr.sort(reverse=True)
            #print("arr, sorted", arr)
            hue_values = [hue[1] for hue in arr]

            #print("hue values", hue_values)
            self.results.append(hue_values)

            # Debugging:
            if self.debug:
                # Print results
                print("Cluster centers: \n", clusters.cluster_centers_)
                print("Most dominant hue values", self.results)

                # Show image and corresponding palette
                img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
                palette = cv.cvtColor(palette, cv.COLOR_HSV2BGR)
                # cv.imshow("Image BGR", img)
                # cv.waitKey(0)
                # cv.imshow("Palette", palette)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                index += 1
                print("Clustered image number: ", index)

                # print("Hue clusters for all images, aka. results:", len(self.results), self.results)
        return self.results

