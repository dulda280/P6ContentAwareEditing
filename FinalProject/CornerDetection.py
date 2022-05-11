import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from Downsampling import *


class CornerDetection:
    # import images

    def __init__(self, images):
        self.images = images

    def makeImagesGrayscale(self, imageDir):
        grayImages = []
        for image in imageDir:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayImages.append(grayImage)
        return grayImages

    def doGausBlur(self, imageDir):
        gausImages = []
        for image in imageDir:
            gausImg = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0, sigmaY=0)
            gausImages.append(gausImg)
        return gausImages

    def getAllCorners(self, imageDir):
        n_corners = []
        n_withinBox = []
        n_filterConers = []
        for index in range(0, len(imageDir)):
            filteredCorners = 0
            max_corners = int(imageDir[0].shape[0] * 1.5)
            corners = cv2.goodFeaturesToTrack(imageDir[index], max_corners, 0.03, 10)
            corners = np.int0(corners)
            # rint("Corners,,,,,,,,,,,,,,,,,,", corners)
            #canny = cv2.Canny(imageDir[index], threshold1=50, threshold2=250)
            x1, y1, x2, y2 = int(imageDir[index].shape[0] * 0.20), 0, int(imageDir[index].shape[0] * 0.80), int(imageDir[index].shape[1])
            rectIMG = cv2.rectangle(imageDir[index], (int(imageDir[index].shape[0] * 0.20), 0), (int(imageDir[index].shape[0] * 0.80), int(imageDir[index].shape[1])),
                                    255, 2)
            #cv2.imshow('rect', rectIMG)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(imageDir[index], (x, y), 3, 255, -1)
                if x1 < x < x2 and y1 < y < y2:
                    filteredCorners += 1

            n_filterConers.append(filteredCorners)
            n_withinBox.append([filteredCorners/len(corners)])
            n_corners.append([len(corners)])

        # print("n_corners", n_corners)
        return n_corners, n_withinBox

    def cornerDetection(self, imageDir):
        blurIMG = self.doGausBlur(imageDir)
        preProcIMG = self.makeImagesGrayscale(blurIMG)
        imageDir = preProcIMG
        cornerImages = []
        cornerCount = 0
        cornerCountArray = []
        for index in range(0, len(imageDir)):
            cornerCount = 0
            IMG = imageDir[index]
            corners = cv2.goodFeaturesToTrack(IMG, 500, 0.03, 10)
            corners = np.int0(corners)
            canny = cv2.Canny(IMG, threshold1=50, threshold2=250)
            for i in corners:
                x, y = i.ravel()

                tempX = int(i[0][0])
                tempY = int(i[0][1])
                # print(f"pixel value at index({int(tempX)}, {int(tempY)}): ", canny[tempX][tempY])
                # print("ffff", len(canny))
                for j in range(0, 4):
                    if x + j < len(canny) and y + j < len(canny) and x - j > 0 and y - j > 0:
                        if canny[x, y] > 0 or canny[x + j, y] > 0 or canny[x, y + j] > 0 or canny[x + j, y + j] > 0 or \
                                canny[
                                    x - j, y] > 0 or canny[x, y - j] > 0 or canny[x - j, y - j] > 0:
                            # print(f"pixel value at corner index({int(x)}, {int(y)}): {canny[int(tempX), int(tempX)]} ")
                            cv2.circle(canny, (x, y), 3, 255, -1)
                            cornerCount += 1
            cornerCountArray.append(cornerCount)

            cornerImages.append(canny)
        # print("Corners: ", len(cornerCountArray), cornerCountArray)
        return cornerCountArray, cornerImages

        # i, j = (canny > 200).nonzero()
        # vals = image[x, y]

    def main(self):
        blurImg = self.doGausBlur(self.images)
        preProcessImg = self.makeImagesGrayscale(blurImg)
        # cornerIMG = self.cornerDetection(preProcessImg)  # Nogle hjørner
        corners, cornerBoxed = self.getAllCorners(preProcessImg)  # Mega mange hjørner
        # print("Corners.......", corners)
        return corners, cornerBoxed

    # for image in cornerIMG:
    #    cv2.imshow("canny", image)
    #    cv2.waitKey(0)

    # newCanny = canny[1]
    # x, y = (newCanny < 200).nonzero()
    # vals = newCanny[x, y]
    # newCanny[x, y] = 0
    # cv2.imshow("newnew", newCanny)
    # cv2.waitKey(0)


def makeImageFolder():
    path = os.getcwd()
    path = path + '\\Input_Directory_Landscape'
    print('her er path: ', path)
    pathDir = os.listdir(path)
    print('her er directory: ', pathDir)
    imagess = []

    for image in range(0, len(pathDir)):
        # print("image: ", pathDir[image])
        # print("path: ", str(path) + '\\' + str(pathDir[image]))
        # print("next image: ", pathDir[image])
        # print("len", len(pathDir))
        print(str(path) + '\\' + str(pathDir[image]))
        temp = cv2.imread(str(path) + '\\' + str(pathDir[image]), cv2.IMREAD_COLOR)
        imagess.append(temp)

    return imagess


def rescale_image(imageDir, res_x, res_y):
    rscArray = []
    for image in imageDir:
        rescale_dimensions = (res_y, res_x)
        rescaled_image = cv2.resize(image, rescale_dimensions, interpolation=cv2.INTER_AREA)
        rescaled_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB)
        rscArray.append(rescaled_image)
    return rscArray


if __name__ == "__main__":
    images = makeImageFolder()
    scaleIMG = rescale_image(images[0:20], 256, 256)
    corners = CornerDetection(scaleIMG)
    preProcsIMG = corners.makeImagesGrayscale(corners.doGausBlur(scaleIMG))
    cornerCount, withinBoxCntPRCNT = corners.getAllCorners(preProcsIMG)

    print(f'Amount of corners: {cornerCount}')
    print("_____________________________")
    print(f'Corner% within Bbox: {withinBoxCntPRCNT}')
    # print("_____________________________")
    # print(f'Coners within Bbox: {withinBoxCnt}')

    for image in preProcsIMG:
        cv2.imshow("cornerIMG", image)
        cv2.waitKey(0)
