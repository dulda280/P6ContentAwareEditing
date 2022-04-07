from Downsampling import *
from Kmeans import *
#from DBscan import *

if __name__ == '__main__':
    images = Downsampling()
    images.import_images()
    images.rescale_images()
    images.recolor_images()
    images.BGR2HSV()

    clusters = Kmeans()
    clusters.clustering()

    #db = DBscan()

    # cluster = Clustering()
    # classes = Classification()

