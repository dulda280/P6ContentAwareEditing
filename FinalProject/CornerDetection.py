import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from Downsampling import *


class CornerDetection:
    # import images
    img = Downsampling()
    images = img.rescale_images()

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

        for index in range(0, len(imageDir)):
            max_corners = imageDir[0].shape[0] * 1.5
            corners = cv2.goodFeaturesToTrack(imageDir[index], max_corners, 0.03, 10)
            corners = np.int0(corners)
            print("Corners,,,,,,,,,,,,,,,,,,", corners)
            canny = cv2.Canny(imageDir[index], threshold1=50, threshold2=250)

            for i in corners:
                x, y = i.ravel()
                cv2.circle(imageDir[index], (x, y), 3, 255, -1)
                cv2.circle(canny, (x, y), 3, 255, -1)

            n_corners.append(len(corners))

        print("n_corners", n_corners)
        return n_corners

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
                print(f"pixel value at index({int(tempX)}, {int(tempY)}): ", canny[tempX][tempY])
                print("ffff", len(canny))
                for j in range(0, 4):
                    if x + j < len(canny) and y + j < len(canny) and x - j > 0 and y - j > 0:
                        if canny[x, y] > 0 or canny[x + j, y] > 0 or canny[x, y + j] > 0 or canny[x + j, y + j] > 0 or \
                                canny[
                                    x - j, y] > 0 or canny[x, y - j] > 0 or canny[x - j, y - j] > 0:
                            print(f"pixel value at corner index({int(x)}, {int(y)}): {canny[int(tempX), int(tempX)]} ")
                            cv2.circle(canny, (x, y), 3, 255, -1)
                            cornerCount += 1
            cornerCountArray.append(cornerCount)

            cornerImages.append(canny)
        print("Corners: ", len(cornerCountArray), cornerCountArray)
        return cornerCountArray

        # i, j = (canny > 200).nonzero()
        # vals = image[x, y]

    # Finder antallet af kanter
    def main(self):
        blurImg = self.doGausBlur(self.images)
        preProcessImg = self.makeImagesGrayscale(blurImg)
        # cornerIMG = self.cornerDetection(preProcessImg)  # Nogle hjørner
        corners = self.getAllCorners(preProcessImg)  # Mega mange hjørner
        print("Corners.......", corners)
        return corners

    # for image in cornerIMG:
    #    cv2.imshow("canny", image)
    #    cv2.waitKey(0)

    # newCanny = canny[1]
    # x, y = (newCanny < 200).nonzero()
    # vals = newCanny[x, y]
    # newCanny[x, y] = 0
    # cv2.imshow("newnew", newCanny)
    # cv2.waitKey(0)
