import glob
import shutil
from sklearn.cluster import KMeans

from FileManager import *


class Kmeans:
    fileManager = FileManager()

    # save images to:
    imdir = "Thomas//images//image_directory"
    targetdir = "FinalProject//images//save_directory//"

    filelist = glob.glob(os.path.join(imdir, ('*.jpg' or '*.jpeg')))
    filelist.sort()

    def clustering(self, features, img):

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++").fit(np.array(features))

        # Save image to directory
        print("\n")
        for i, clusterGroup in enumerate(kmeans.labels_):
            print("    Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
            # fileManager.save_image(targetdir, filelist[i], "cluster_group", str(i))
            if clusterGroup == 0:
                shutil.copy(self.filelist[i],
                            self. targetdir + "//Cluster_Group_01//" + str(i) + "_" + "cgroup_" + str(clusterGroup) + "_" + ".jpg")
            elif clusterGroup == 1:
                shutil.copy(self.filelist[i],
                            self.targetdir + "//Cluster_Group_02//" + str(i) + "_" + "cgroup_" + str(clusterGroup) + "_" + ".jpg")
            else:
                continue
