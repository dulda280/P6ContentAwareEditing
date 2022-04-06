from Downsampling import *
#from Classification import *
from Kmeans import *


if __name__ == '__main__':
    images = Downsampling()
    images.import_images()
    images.rescale_images()
    images.recolor_images()
    images.BGR2HSV()

    clusters = Kmeans()
    clusters.clustering()

    # cluster = Clustering()
    # classes = Classification()

