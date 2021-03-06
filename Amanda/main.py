from Downsampling import *
from Kmeans import *
from DBscan import *
from KmeansProportion import *

if __name__ == '__main__':
    # Import images, rescale, convert to HSV
    images = Downsampling()
    images.hsv_images()

    # Cluster hue values in images
    kmeans = Kmeans()
    kmeans.clustering()

    # Cluster images based on edges and hue
    db = DBscan()
    db.classify()
