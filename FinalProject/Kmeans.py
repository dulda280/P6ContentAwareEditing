
import numpy as np
from sklearn.cluster import KMeans

from Downsampling import *

# Colors: https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a
# Edges:  https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

# HSV range: H 360, S 180, V 180


class Kmeans:
    # Import images
    images = Downsampling()
    data = images.BGR2HSV()
    results = []

    debug = False

    def clustering(self):
        # Number of clusters
        clusters = KMeans(n_clusters=5)

        for img in self.data:
            # The hue for each image is stored in a list
            hue_values = []

            # Making the clustering process illumination invariant by setting saturation and value to 180.
            img[:, :, 1] = 180
            img[:, :, 2] = 180

            # Perform clustering
            clusters.fit(img.reshape(-1, 3))
            #print("img reshape", img.reshape(-1, 3))
            #print("img", img)

            # Create palette with clusters
            width = 300
            palette = np.zeros((50, width, 3), np.uint8)
            steps = width / clusters.cluster_centers_.shape[0]

            for idx, centers in enumerate(clusters.cluster_centers_):
                palette[:, int(idx * steps):(int((idx + 1) * steps)), :] = centers

            for color in clusters.cluster_centers_:
                hue_values.append(round(color[0], 2))

            self.results.append(hue_values)

            if self.debug:
                # Print results
                print("Cluster centers: \n", clusters.cluster_centers_)
                print("Most dominant hue values", self.results)

                # Show image and corresponding palette
                img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
                palette = cv.cvtColor(palette, cv.COLOR_HSV2BGR)
                cv.imshow("Image BGR", img)
                cv.waitKey(0)
                cv.imshow("Palette", palette)
                cv.waitKey(0)
                cv.destroyAllWindows()

        print("Hue clusters for all images, aka. results:", len(self.results), self.results)
        return self.results

