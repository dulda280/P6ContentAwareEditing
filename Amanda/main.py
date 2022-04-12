from Downsampling import *
from Kmeans import *
from DBscan import *
from KmeansProportion import *

if __name__ == '__main__':
    images = Downsampling()
    images.import_images()
    images.rescale_images()
    images.recolor_images()
    images.BGR2HSV()

    kmeans = Kmeans()
    kmeans.clustering()

    db = DBscan()
    db.classify()
