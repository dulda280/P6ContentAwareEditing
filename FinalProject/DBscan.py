import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

from Kmeans import *
from ThomasMain import *
from CornerDetection import *

class DBscan:

    # import data
    kmeans = Kmeans()
    data = kmeans.clustering()
    canny = ThomasMain() #return self.largest_edge, self.number_of_edges
    edge_data = canny.main() #return self.largest_edge, self.number_of_edges
    corner = CornerDetection()
    corner_data = corner.main()

    def merge_data(self):
        for i in range(0, len(self.data)):
            self.data[i].append(float(self.edge_data[0][i]))  # append largest edge
            self.data[i].append(float(self.edge_data[1][i]))  # append number of edges
            self.data[i].append(float(self.corner_data[i]))   # number of corners
            #self.data[i].append(float(self.corner_data[1][i]))

        print("corner_data 0 =", self.corner_data)
        print("DATA: [HUE, HUE, HUE, HUE, HUE, LARGEST_EDGE, N_EDGES] \n", self.data)
        return self.data

    def classify(self):
        # Compute DBSCAN
        db = DBSCAN(eps=50, min_samples=2).fit(self.data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Labels on clusters \n", labels)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(self.data, labels))

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self.data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14, )

            xy = self.data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6, )

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()
