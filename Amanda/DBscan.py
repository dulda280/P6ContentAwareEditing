import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

from Kmeans import *

class DBscan:

    # import data
    hue_values = Kmeans()
    data = hue_values.clustering()

    data = [[100.77668952007835, 16.742201294879344, 170.63788300835654, 124.46987951807229, 4.423177083333336],
            [2.9232175502742166, 138.56993006993008, 171.78308026030368, 103.72849462365592, 12.642285714285713],
            [6.627039627039629, 133.5816326530612, 101.35800807537012, 170.4865424430642, 16.710886806056237],
            [112.98875000000001, 8.32081447963801, 172.14968152866243, 102.08333333333333, 134.39130434782606],
            [9.439350180505418, 165.88235294117646, 106.73821989528795, 67.04382470119522, 36.08097165991903]]

    '''
    data = [[16.742201294879344, 124.46987951807229, 170.63788300835654, 100.77668952007835, 4.423177083333336],
            [3.2312535775615387, 137.72826086956522, 171.5674518201285, 103.63513513513513, 13.28218465539662],
            [17.342600163532296, 133.5816326530612, 101.35800807537012, 170.4865424430642, 7.234321157822194],
            [134.39130434782606, 8.32081447963801, 102.08333333333333, 172.14968152866243, 112.98875000000001],
            [9.439350180505418, 106.73821989528795, 165.88235294117646, 67.04382470119522, 36.08097165991903]]'''

    labels_true = ["1", "2", "3", "4", "5"]

    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels)
    )
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14, )

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6, )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
