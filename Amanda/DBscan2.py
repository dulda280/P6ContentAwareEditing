from sklearn.cluster import DBSCAN
import numpy as np

class DBscan2:

    data =[
        [6.05, 103.25, 31.13, 154.96, 53.5], [10.6, 80.64, 42.1, 109.19, 166.81], [12.31, 109.48, 78.9, 154.38, 30.88],
         [41.63, 12.21, 116.49, 169.4, 88.65], [32.62, 113.07, 74.85, 12.56, 157.88], [9.64, 160.5, 28.19, 15.58, 115.28],
         [0.63, 178.87, 21.79, 138.16, 73.96], [178.77, 21.31, 136.03, 78.2, 1.31], [0.26, 178.95, 20.53, 102.59, 43.42],
         [44.56, 179.0, 0.32, 21.04, 29.68],
         [10.35, 172.73, 57.22, 27.01, 111.1], [16.74, 124.47, 170.64, 4.42, 100.78], [171.78, 3.23, 103.73, 138.57, 13.28],
         [7.23, 101.1, 170.26, 132.05, 17.34], [102.08, 8.32, 172.15, 112.99, 134.39], [106.74, 9.44, 165.88, 67.04, 36.08],
         [22.75, 35.11, 12.44, 141.0, 16.78], [106.52, 13.3, 70.03, 120.5, 156.75], [176.87, 17.45, 102.91, 65.63, 3.52],
         [39.75, 175.8, 115.48, 146.26, 9.37], [9.13, 107.06, 171.78, 94.35, 39.62], [100.15, 11.47, 175.78, 4.22, 58.07],
         [11.17, 94.4, 173.07, 64.57, 4.22], [26.95, 117.2, 9.64, 165.86, 89.13], [117.92, 5.31, 174.51, 96.98, 13.48],
         [16.42, 105.47, 169.23, 8.89, 133.22], [18.23, 106.11, 164.78, 125.63, 9.27], [107.9, 17.97, 168.43, 7.32, 129.2],
         [6.57, 118.17, 170.29, 104.93, 14.34], [93.27, 13.37, 172.96, 108.0, 5.77], [15.7, 123.9, 170.9, 6.98, 105.81],
         [15.36, 107.74, 140.39, 7.04, 172.71], [94.33, 14.45, 168.95, 5.85, 36.05], [14.05, 162.22, 94.66, 6.76, 32.82],
         [6.23, 124.59, 162.5, 96.56, 15.8], [4.56, 172.19, 92.76, 12.15, 136.21], [135.45, 11.96, 93.61, 171.2, 3.94],
         [5.76, 173.56, 93.58, 12.47, 136.02], [9.63, 104.98, 168.75, 38.36, 120.65], [7.91, 107.28, 172.18, 34.41, 141.74],
         [10.51, 106.18, 166.38, 122.84, 58.27], [7.22, 107.71, 172.85, 28.31, 141.42],
         [169.02, 10.04, 105.64, 41.09, 125.25], [109.58, 9.99, 171.07, 38.93, 141.06],
         [170.55, 9.53, 109.37, 59.25, 137.35], [110.44, 14.92, 170.0, 136.31, 3.87], [3.85, 137.1, 108.0, 171.11, 19.13],
         [4.0, 171.59, 109.02, 18.6, 137.88], [3.32, 139.2, 169.78, 108.11, 19.98], [3.36, 111.2, 169.31, 18.14, 139.75],
         [110.51, 22.29, 167.82, 4.65, 139.88], [19.65, 110.62, 168.08, 3.79, 138.89],
         [10.86, 107.96, 169.83, 45.24, 135.19], [22.98, 139.83, 170.12, 4.8, 109.88], [36.65, 122.8, 164.61, 7.14, 106.84],
         [110.79, 8.35, 171.05, 138.44, 83.36], [5.47, 147.27, 111.93, 171.54, 79.01], [108.9, 10.02, 165.0, 128.49, 82.72],
         [8.73, 127.99, 163.9, 86.58, 108.82], [107.62, 8.28, 170.5, 43.11, 139.76], [110.56, 9.04, 139.94, 42.37, 169.13],
         [9.35, 110.41, 169.95, 43.81, 140.38], [131.82, 19.8, 163.46, 111.98, 5.6], [110.91, 14.39, 169.11, 134.38, 2.99],
         [7.57, 171.65, 113.79, 80.47, 143.63], [6.21, 112.56, 171.66, 142.64, 85.28], [66.28, 17.26, 172.64, 86.92, 45.21],
         [16.54, 118.26, 158.44, 104.56, 2.92], [99.8, 21.05, 107.71, 153.25, 75.13], [107.14, 17.71, 81.28, 150.3, 101.58],
         [101.64, 17.28, 70.75, 106.91, 137.0], [106.52, 72.33, 19.03, 100.81, 143.7],
         [106.01, 72.74, 20.08, 99.55, 141.59], [108.04, 51.1, 65.55, 93.23, 37.8], [107.22, 40.47, 71.67, 97.37, 51.59],
         [26.7, 108.39, 59.59, 85.68, 152.68], [23.56, 88.89, 63.89, 108.26, 44.73], [105.05, 11.57, 163.66, 54.12, 87.93],
         [106.21, 58.21, 96.89, 163.21, 12.5], [23.63, 103.88, 162.97, 58.86, 11.77], [2.61, 177.54, 30.96, 16.49, 6.51],
         [20.63, 107.37, 56.64, 46.44, 77.56], [80.27, 105.3, 54.0, 12.26, 169.64], [96.64, 20.14, 158.49, 108.63, 54.84],
         [27.58, 99.55, 17.58, 173.4, 54.84], [60.0, 60.0, 60.0, 60.0, 60.0], [65.01, 117.08, 33.47, 101.75, 145.03]]

    X = np.array(data)
    clustering = DBSCAN(eps=len(data)/2, min_samples=2).fit(X)

    labels = clustering.labels_

    print("clustering labels", labels)
    print("clustering results", clustering)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)