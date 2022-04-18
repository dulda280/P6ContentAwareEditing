from Downsampling import *
from Kmeans import *
from DBscan import *
from ThomasMain import *
from CornerDetection import *

if __name__ == '__main__':
    # Import images, rescale, convert to HSV
    images = Downsampling()
    img = images.rescale_images()
    hsvImg = images.BGR2HSV()
    # Cluster hue values in images
    # kmeans = Kmeans()
    # kmeans.clustering()

    # canny = ThomasMain()
    # canny.main()

    # corner = CornerDetection()
    # corner.main()
    kmeans = Kmeans(hsvImg)
    data = kmeans.clustering()
    print("data: ", data)

    canny = ThomasMain(img)  # return self.largest_edge, self.number_of_edges
    edge_data = canny.main()  # return self.largest_edge, self.number_of_edges
    print("edge_data: ", edge_data)

    corner = CornerDetection(img)
    corner_data = corner.main()
    print("corner_data: ", corner_data)

    # Cluster images based on edges and hue
    db = DBscan(data, edge_data[0], edge_data[1], corner_data)
    db.merge_data()
    db.classify()

    # neural network: Kristian
