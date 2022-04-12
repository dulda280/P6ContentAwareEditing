from Downsampling import *
from Kmeans import *
from DBscan import *
from ThomasMain import *

if __name__ == '__main__':
    # Import images, rescale, convert to HSV
    images = Downsampling()
    images.hsv_images()

    # Cluster hue values in images
    kmeans = Kmeans()
    kmeans.clustering()

    canny = CannyEdge()
    canny.main()
    # edge detection 1: Thomas
    # edge detection 2: Seb

    # Cluster images based on edges and hue
    db = DBscan()
    db.classify()

    # neural network: Kristian
