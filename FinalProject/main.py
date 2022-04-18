from Downsampling import *
from Kmeans import *
from DBscan import *
from ThomasMain import *
from CornerDetection import *

if __name__ == '__main__':
    # Import images, rescale, convert to HSV
    images = Downsampling()
    images.hsv_images()

    # Cluster hue values in images
    # kmeans = Kmeans()
    # kmeans.clustering()

    # canny = ThomasMain()
    # canny.main()

    # corner = CornerDetection()
    # corner.main()

    # Cluster images based on edges and hue
    db = DBscan()
    db.merge_data()
    db.classify()

    # neural network: Kristian
