from Downsampling import *
from AverageColors import *
from Kmeans import *
from DBscan import *
from ThomasMain import *
from CornerDetection import *
from FileManager import *
from Normalize import *
from HOG import *
import math

if __name__ == '__main__':
    # Class Instantiations
    fileManager = FileManager()
    normalize = Normalize()

    # Import images, rescale, convert to HSV
    images = Downsampling()
    img = images.rescale_images()
    hsvImg = images.BGR2HSV()
    freshIMG = images.import_images()
    rscImages = rescale_image(freshIMG, 128, 64)

    # Calculate average colors
    avg = AverageColors()
    avgHue = avg.main(hsvImg)

    # Cluster hue values in images
    kmeans = Kmeans(hsvImg)
    fiveColours = kmeans.clustering()

    # Edge detection
    canny = ThomasMain(img)  # return self.largest_edge, self.number_of_edges
    edge_data = canny.main(False)  # return self.largest_edge, self.number_of_edges

    # Corner detection
    corner = CornerDetection(img)
    corner_data, cornerBoxData = corner.main()
    #print("corner_data: ", corner_data)
    #print(len(corner_data))
    hogs = HOG(rscImages)
    #matrix = (normalize.norm(data), normalize.norm(edge_data[0]), normalize.norm(edge_data[1]), normalize.norm(corner_data), cornerBoxData)
    #print(matrix)
    # Cluster images based on edges and hue
    db = DBscan(avgHue, edge_data[0], edge_data[1], corner_data, cornerBoxData)
    mergeSample = db.merge_data()
    db.classify()

    print("*************************************************")
    print("Merge data sample: ", mergeSample[3])
    print("Data", fiveColours)
    print("Data Mean", np.mean(fiveColours))
    print("Data Length", len(fiveColours))
    print("Corner Data", corner_data)
    print("Corner Data Mean", np.mean(corner_data))
    print("Corner Data Length", len(corner_data))
    print("*************************************************")
    print("Number of edges list", edge_data[0])
    print("Number of edges list Mean", np.mean(edge_data[0]))
    print("Number of edges list Length", len(edge_data))
    print("Largest edges list", edge_data[1])
    print("Largest edges list Mean", np.mean(edge_data[1]))
    print("Largest edges list Length", len(edge_data))
    print("*************************************************")
    print("HOGGERSS:", print(len(hogs)))
    print("*************************************************")


    # neural network: Kristian
