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
    test = cv2.imread(str(path) + '\\' + str(pathDir[0]), cv2.IMREAD_COLOR)
    # cv2.imshow(f'test', test)
    # cv2.waitKey(0)
    for image in range(0, len(pathDir)):
        print("image: ", pathDir[image])

        temp = cv2.imread(str(path) + '\\' + str(pathDir[image]), cv2.IMREAD_COLOR)
        images.append(temp)
        cv2.imshow(f'test', temp)
        cv2.waitKey(0)

    return images

def makeImagesGrayscale(imageDir):
    grayImages = []
    for image in imageDir:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImages.append(grayImage)

    return grayImages

def doSobelHori(imageDir):
    horizontalSobel = []
    for index in range(0, len(imageDir)):
        sobely = cv2.Sobel(src=imageDir[index], ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) #Sobel edge detection y-axis
        horizontalSobel.append(sobely)

    return horizontalSobel


def doSobelVert(imageDir):
    verticalSobel = []
    for index in range(0, len(imageDir)):
        sobelx = cv2.Sobel(src=imageDir[index], ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        verticalSobel.append(sobelx)
    return verticalSobel

if __name__ == "__main__":
    imageData = makeImageFolder()
    grayScaleImages = makeImagesGrayscale(imageData)
    vertImg = doSobelVert(grayScaleImages)
    horiImg = doSobelHori(grayScaleImages)
    for image in vertImg:
        cv2.imshow(f'vertImg', image)
        cv2.waitKey(0)
        print(vertImg[0])

    for image in horiImg:
        cv2.imshow(f'HoriImg', image)
        cv2.waitKey(0)
        print(horiImg[0])

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


