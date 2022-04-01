import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def makeImageFolder():
    path = os.getcwd()
    path = path + '\\data'
    print('her er path: ', path)
    pathDir = os.listdir(path)
    print('her er directory: ', pathDir)
    images = []

    for image in range(0, len(pathDir)):
        # print("image: ", pathDir[image])
        # print("path: ", str(path) + '\\' + str(pathDir[image]))
        # print("next image: ", pathDir[image])
        # print("len", len(pathDir))

        temp = cv2.imread(str(path) + '\\' + str(pathDir[image]), cv2.IMREAD_COLOR)
        images.append(temp)

    return images


def makeImagesGrayscale(imageDir):
    grayImages = []
    for image in imageDir:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImages.append(grayImage)

    return grayImages


def doGausBlur(imageDir):
    gausImages = []
    for image in imageDir:
        gausImg = cv2.GaussianBlur(src=image, ksize=(3, 3), sigmaX=0, sigmaY=0)
        gausImages.append(gausImg)

    return gausImages


def fuckdigCannybitch(imageDir):
    cannyBitch = []
    for index in range(0, len(imageDir)):
        ree = rescale_image(imageDir[index], 64, 64)
        canny1B = cv2.Canny(ree, threshold1=100, threshold2=200)
        cv2.imshow(f'cannybitch', canny1B)
        cv2.waitKey(0)
        cannyBitch.append(canny1B)

    return cannyBitch


def doSobelHori(imageDir):
    horizontalSobel = []
    for index in range(0, len(imageDir)):
        reScaled = rescale_image(imageDir[index], 512, 512)
        sobely = cv2.Sobel(src=reScaled, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=5)  # Sobel edge detection y-axis

        horizontalSobel.append(sobely)

    return horizontalSobel


def doSobelVert(imageDir):
    verticalSobel = []
    for index in range(0, len(imageDir)):
        reScaled = rescale_image(imageDir[index], 512, 512)
        sobelx = cv2.Sobel(src=reScaled, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=5)
        verticalSobel.append(sobelx)

    return verticalSobel


def rescale_image(image, res_x, res_y):
    rescale_dimensions = (res_y, res_x)
    rescaled_image = cv2.resize(image, rescale_dimensions, interpolation=cv2.INTER_AREA)
    rescaled_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB)
    return rescaled_image


if __name__ == "__main__":
    imageData = makeImageFolder()
    grayScaleImages = makeImagesGrayscale(imageData)
    preProcessImg = doGausBlur(grayScaleImages)

    vertImg = doSobelVert(preProcessImg)
    horiImg = doSobelHori(preProcessImg)

    for image in vertImg:
        cv2.imshow(f'vertImg', image)
        cv2.waitKey(0)
        print(image)

    for image in horiImg:
        cv2.imshow(f'HoriImg', image)
        cv2.waitKey(0)
        print(image)

# # Sobel Edge Detection
# sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
#
# # Display Sobel Edge Detection Images
# cv2.imshow('Sobel X', sobelx)
# cv2.waitKey(0)
# cv2.imshow('Sobel Y', sobely)
# cv2.waitKey(0)
# cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# cv2.waitKey(0)
