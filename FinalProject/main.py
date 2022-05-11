from Downsampling import *
from AverageColors import *
from KmeansFeature import *
from DBscan import *
from Kmeans import *
from ThomasMain import *
from CornerDetection import *
from FileManager import *
from Normalize import *
from HOG import *
import math

if __name__ == '__main__':
    # Class Instantiations
    fileManager = FileManager()

    # Import images, rescale, convert to HSV
    images = Downsampling()
    img = images.rescale_images()
    hsvImg = images.BGR2HSV()

    # Calculate average colors
    avg = AverageColors()
    avgHue = avg.main(hsvImg)

    # Cluster hue values in images, aka. dominant colours
    kmeans = KmeansFeature(hsvImg)
    dominantColours = kmeans.clustering()

    # Edge detection x2
    canny = ThomasMain(img)  # return self.largest_edge, self.number_of_edges
    largest_edge, number_of_edges = canny.main(False)  # return self.largest_edge, self.number_of_edges

    # Corner detection x2
    corner = CornerDetection(img)
    corner_data, cornerBoxData = corner.main()

    # HOG feature
    hogs = HOG(img)

    # Normalise the data and create an array of features
    normalize = Normalize(dominantColours, avgHue, largest_edge, number_of_edges, corner_data, cornerBoxData, hogs)
    feature_array = normalize.merge_data(img)
    print("feature_array", feature_array)

    # Cluster images based on edges and hue
    #db = DBscan()
    #db.classify(feature_array)

    kmeans_cluster = Kmeans()
    kmeans_cluster.clustering(feature_array, img)

    print("*************************************************")
    print("Five most dominant colours", dominantColours)
    print("Dominant colours mean", np.mean(dominantColours))
    print("Dominant colours length", len(dominantColours))
    print("Corner Data", corner_data)
    print("Corner Data Mean", np.mean(corner_data))
    print("Corner Data Length", len(corner_data))
    print("*************************************************")
    print("Number of edges list", number_of_edges)
    print("Number of edges list Mean", np.mean(number_of_edges))
    print("Number of edges list Length", len(number_of_edges))
    print("Largest edges list", largest_edge)
    print("Largest edges list Mean", np.mean(largest_edge))
    print("Largest edges list Length", len(largest_edge))
    print("*************************************************")
    print("HOGGERSS:", print(len(hogs)))
    print("*************************************************")


    # neural network: Kristian
