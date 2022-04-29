import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import numpy as np
import cv2
import os


def HOG(imageDir):
    featureVector = []
    for index in range(0, len(imageDir)):
        fd, hogImage = hog(imageDir[index], pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, multichannel=True, feature_vector=True)

        fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        axis1.axis('off')
        axis1.imshow(imageDir[index], cmap=plt.cm.gray)
        axis1.set_title('Input')

        # hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 10))
        featureVector.append(fd)
        axis2.axis('off')
        axis2.imshow(hogImage, cmap=plt.cm.gray)
        axis2.set_title('Histogram of Oriented Gradients')
        plt.show()

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

        temp = cv2.imread(str(path) + '\\' + str(pathDir[image]), cv2.IMREAD_COLOR)
        images.append(temp)

    return images


def makeImagesGrayscale(imageDir):
    grayImages = []

    for image in range(0, len(imageDir)):
        print("IMAGE: ", imageDir[image])
        grayImage = cv2.cvtColor(imageDir[image], cv2.COLOR_BGR2GRAY)
        grayImages.append(grayImage)
    return grayImages


def rescale_image(imageDir, res_x, res_y):
    rscImages = []
    for image in range(0, len(imageDir)):
        rescale_dimensions = (res_y, res_x)
        rescaled_image = cv2.resize(imageDir[image], rescale_dimensions, interpolation=cv2.INTER_AREA)
        rescaled_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB)
        rscImages.append(rescaled_image)

    return rscImages


if __name__ == "__main__":
    pathLandscape = "C:\\Users\\sebbe\\Desktop\\MED-local\\P6ContentAwareEditing\\KristianNN\\train_landscape"
    pathPortrait = "C:\\Users\\sebbe\\Desktop\\MED-local\\P6ContentAwareEditing\\KristianNN\\train_portrait"

    hsvImgesPortrait = makeImageFolder(pathPortrait)
    hsvImgesLandscape = makeImageFolder(pathLandscape)
    hsvImgPort = hsvImgesPortrait[0:3]
    hsvImgLand = hsvImgesLandscape[0:3]
    rscImg = rescale_image(hsvImgLand, 256, 256)
    hogFeatures = HOG(rscImg)
    print("HOG features: ", hogFeatures)
    print("------------------------------------")
    print("Features per image: ", len(hogFeatures[0]))
