import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import numpy as np
import cv2
import os, shutil, glob, os.path
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from Evaluation import *


def HOG(imageDir):
    featureVector = []
    for index in range(0, len(imageDir)):
        fd, hogImage = hog(imageDir[index], pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True,
                           multichannel=True, feature_vector=True)

        # fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        # axis1.axis('off')
        # axis1.imshow(imageDir[index], cmap=plt.cm.gray)
        # axis1.set_title('Input')

        # hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 10))
        featureVector.append(fd)
        # axis2.axis('off')
        # axis2.imshow(hogImage, cmap=plt.cm.gray)
        # axis2.set_title('Histogram of Oriented Gradients')
        # plt.show()

    return featureVector


def makeImageFolder(path=os.getcwd() + '\\Input_Directory_Landscape'):
    print('her er path: ', path)
    pathDir = os.listdir(path)
    print('her er directory: ', pathDir)
    images = []

    for image in range(0, len(pathDir)):
        # print("image: ", pathDir[image])½1½
        # print("path: ", str(path) + '\\' + str(pathDir[image]))
        # print("next image: ", pathDir[image])
        # print("len", len(pathDir))

        temp = cv2.imread(str(path) + '/' + str(pathDir[image]), cv2.IMREAD_COLOR)
        images.append(temp)

    return images, pathDir


def makeImagesGrayscale(imageDir):
    grayImages = []

    for image in range(0, len(imageDir)):
        print("IMAGE: ", imageDir[image])
        grayImage = cv2.cvtColor(imageDir[image], cv2.COLOR_BGR2GRAY)
        grayImages.append(grayImage)
    return grayImages


def rescale_image(imageDir, res_x=128, res_y=64):
    rscImages = []
    for image in range(0, len(imageDir)):
        rescale_dimensions = (res_y, res_x)
        rescaled_image = cv2.resize(imageDir[image], rescale_dimensions, interpolation=cv2.INTER_AREA)
        rescaled_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB)
        rscImages.append(rescaled_image)

    return rscImages


if __name__ == "__main__":
    pathLandscape = "P6ContentAwareEditing\\KristianNN\\train_landscape"
    pathPortrait = "P6ContentAwareEditing\\KristianNN\\train_portrait"
    pathMac = path=os.getcwd() + '/KristianNN/train_portrait'
    pathMac2 = os.getcwd() + '/KristianNN/train_landscape'

    targetdir = "images//kmeansHOG//"
    hsvImgesPortrait, portDir = makeImageFolder(pathMac)
    hsvImgesLandscape, landDir = makeImageFolder(pathMac2)
    hsvImgPort = hsvImgesPortrait
    hsvImgLand = hsvImgesLandscape
    rscImgLand = rescale_image(hsvImgLand, 256, 256)
    rscImgPort = rescale_image(hsvImgPort, 256, 256)

    hogFeaturesLand = HOG(rscImgLand)
    hogFeaturesPort = HOG(rscImgPort)
    HOGGERS = hogFeaturesLand + hogFeaturesPort
    fullDir = landDir + portDir
    allImages = hsvImgLand + hsvImgPort

    landRange = len(rscImgLand) - 1
    portRange = len(rscImgPort) + landRange

    print("------------------------------------")
    print("HOG features per image: ", len(hogFeaturesPort[0]))

    kmeans = KMeans(n_clusters=2, random_state=0, algorithm="elkan").fit(HOGGERS)
    landClusterCount = 0
    portClusterCount = 0
    print("------------------------------------")
    print("Kmeans results: ", kmeans.labels_)
    print("------------------------------------")
    print(f"length kmeans labels: {len(kmeans.labels_)}")
    print(f"length fulldir: {len(fullDir)}")
    print(f"landscape range: 0:{landRange}")
    print(f"portrait range: {landRange + 1}:{portRange}")
    print("------------------------------------")
    for index in range(0, len(kmeans.labels_[0:landRange])):
        if kmeans.labels_[index] == 0:
            landClusterCount += 1
    for index in range(landRange + 1, landRange + len(kmeans.labels_[landRange:portRange]) + 1):
        if kmeans.labels_[index] == 1:
            portClusterCount += 1

    print(f"Kmeans Landscape clustering accuracy: {(landClusterCount / len(hsvImgLand)) * 100}%")
    print(f"Kmeans Portrait clustering accuracy: {(portClusterCount / len(hsvImgPort)) * 100}%")
    print("------------------------------------")
    # for index in range(len(kmeans.labels_)):
    #     if kmeans.labels_[index] == 0:
    #         cluster0Img = allImages[index]
    #         fig, ax = plt.subplots()
    #         plt.imshow(cluster0Img)
    #         ax.set_title(f"cluster: {kmeans.labels_[index]}")
    #         plt.show()
    #
    #     elif kmeans.labels_[index] == 1:
    #         cluster1Img = allImages[index]
    #         fig, ax = plt.subplots()
    #         plt.imshow(cluster1Img)
    #         ax.set_title(f"cluster: {kmeans.labels_[index]}")
    #         plt.show()

    dbLandcount = 0
    dbPortcount = 0
    db = DBSCAN(eps=10, min_samples=4).fit(HOGGERS)
    for index in range(0, len(db.labels_[0:landRange])):
        if db.labels_[index] == 0:
            dbLandcount += 1
    for index in range(landRange + 1, landRange + len(db.labels_[landRange:portRange]) + 1):
        if db.labels_[index] == -1:
            dbPortcount += 1

    print(f"DBscan, Landscape clustering accuracy: {(dbLandcount / len(hsvImgLand)) * 100}%")
    print(f"DBscan, Portrait clustering accuracy: {(dbPortcount / len(hsvImgPort)) * 100}%")
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("=======================================================================")
    print("Labels on clusters \n", labels)
    print("-----------------------------------------------------------------------")
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("=======================================================================")

    groundTruthDB = np.zeros_like(labels)
    groundTruthDB[int(((len(groundTruthDB)/2)-1)):len(groundTruthDB)] = -1
    #print(groundTruthDB)
    evaluate = Evaluate(groundTruthDB, labels, -1, 0)
    evaluate.confusionMatrix()
    evaluate.precisionAndRecall()
    evaluate.predictionError()
    evaluate.sumAllDiff(groundTruthDB, labels)
