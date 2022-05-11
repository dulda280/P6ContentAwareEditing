import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py


class DBscan:

    def classify(self, feature_array):
        # Compute DBSCAN00
        db = DBSCAN(eps=0.8, min_samples=2).fit(feature_array)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("=======================================================================")
        print("Labels on clusters \n", labels)
        print("-----------------------------------------------------------------------")
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print("=======================================================================")
        # print("Confusion matrix", metrics.confusion_matrix())
        # print("Precision", metrics.precision_score())
        # print("Recall", metrics.recall_score())
        # print("Precision and recall curve", metrics.plot_precision_recall_curve)
        # print("F-score", metrics.f1_score())
        # print("Accuracy", metrics.accuracy_score())
        # print("Mean squared error", metrics.mean_squared_error())

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k
            # xy = self.data[class_member_mask & core_samples_mask]
            # print(xy)
            # plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14, )
            #
            # xy = self.data[int(class_member_mask) & int(~core_samples_mask)]
            # plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6, )

        # plt.title("Estimated number of clusters: %d" % n_clusters_)
        # plt.show()
