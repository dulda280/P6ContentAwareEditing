import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

class CornerCannyEdge:

    def makeImageFolder(self):
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


    def makeImagesGrayscale(self, imageDir):
        grayImages = []
        for image in imageDir:
            grayImage = self.rescale_image(image, 512, 512)
            grayImage = cv2.cvtColor(grayImage, cv2.COLOR_BGR2GRAY)
            grayImages.append(grayImage)

        return grayImages


    def doGausBlur(self, imageDir):
        gausImages = []
        for image in imageDir:
            gausImg = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0, sigmaY=0)
            gausImages.append(gausImg)

        return gausImages


    def cannyEdge(self, imageDir):
        cannyBitch = []
        for index in range(0, len(imageDir)):
            ree = self.rescale_image(imageDir[index], 512, 512)
            canny1B = cv2.Canny(ree, threshold1=100, threshold2=200)
            # (r, g, b) = cv2.split(canny1B)
            cannyBitch.append(canny1B)

        return cannyBitch


    def doSobelHori(self, imageDir):
        horizontalSobel = []
        for index in range(0, len(imageDir)):
            reScaled = self.rescale_image(imageDir[index], 512, 512)
            sobely = cv2.Sobel(src=reScaled, ddepth=cv2.CV_16U, dx=0, dy=1, ksize=5)  # Sobel edge detection y-axis

            (B, G, R) = cv2.split(sobely)

            horizontalSobel.append(B)

        return horizontalSobel


    def doSobelVert(self, imageDir):
        verticalSobel = []
        for index in range(0, len(imageDir)):
            reScaled = self.rescale_image(imageDir[index], 512, 512)
            sobelx = cv2.Sobel(src=reScaled, ddepth=cv2.CV_16U, dx=1, dy=0, ksize=5)

            (B, G, R) = cv2.split(sobelx)
            kernel = np.ones((3, 3), np.uint8)
            image = cv2.erode(B, kernel)
            image = cv2.dilate(image, kernel)
            verticalSobel.append(image)

        return verticalSobel


    def cc_analysis(self, imageDir):
        ccImages = []
        for index in range(0, len(imageDir)):
            labelNums, labels = cv2.connectedComponents(imageDir[index], connectivity=8)
            labelHue = np.uint8(179 * labels / np.max(labels))
            print("labelNums: ", labelNums)
            print("labels: ", labels)
            blankCH = 255 * np.ones_like(labelHue)
            labeledIMG = cv2.merge([labelHue, blankCH, blankCH])

            # Converting cvt to BGR
            labeledIMG = cv2.cvtColor(labeledIMG, cv2.COLOR_HSV2BGR)

            # set bg label to black
            labeledIMG[labelHue == 0] = 0
            ccImages.append(labeledIMG)

        return ccImages


    def rescale_image(self, image, res_x, res_y):
        rescale_dimensions = (res_y, res_x)
        rescaled_image = cv2.resize(image, rescale_dimensions, interpolation=cv2.INTER_AREA)
        rescaled_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB)
        return rescaled_image


    def getAllCorners(self, imageDir):
        for index in range(0, len(imageDir)):
            corners = cv2.goodFeaturesToTrack(imageDir[index], 500, 0.03, 10)
            corners = np.int0(corners)
            canny = cv2.Canny(imageDir[index], threshold1=50, threshold2=250)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(imageDir[index], (x, y), 3, 255, -1)
                cv2.circle(canny, (x, y), 3, 255, -1)
            cv2.imshow("image", imageDir[index])
            cv2.waitKey(0)
            cv2.imshow("canny", canny)
            cv2.waitKey(0)


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
            print("corners: ", len(corners))
            print("cornerCorners: ", cornerCountArray[index])

        return cornerImages, cornerCountArray

        # i, j = (canny > 200).nonzero()
        # vals = image[x, y]



if __name__ == "__main__":
    cornerDetect = CornerCannyEdge()

    imageData = cornerDetect.makeImageFolder()
    blurImg = cornerDetect.doGausBlur(imageData)
    preProcessImg = cornerDetect.makeImagesGrayscale(blurImg)

    # mangler steps --> canny --> find BLOBs --> lave sobel på original billede og skær alt pånær BLOBs væk --> vertical og horizontal edges located

    # vertImg = doSobelVert(preProcessImg)
    # horiImg = doSobelHori(preProcessImg)
    canny = cornerDetect.cannyEdge(preProcessImg)
    ccCanny = cornerDetect.cc_analysis(canny)

    cornerIMG = cornerDetect.cornerDetection(imageData)
    cornerDetect.getAllCorners(preProcessImg)
    for image in canny:
        cv2.imshow("canny", image)
        cv2.waitKey(0)

    # newCanny = canny[1]
    # x, y = (newCanny < 200).nonzero()
    # vals = newCanny[x, y]
    # newCanny[x, y] = 0
    # cv2.imshow("newnew", newCanny)
    # cv2.waitKey(0)
    for image in ccCanny:
        cv2.imshow("cannycc", image)
        cv2.waitKey(0)

    for image in preProcessImg:
        cv2.imshow("canggnycc", image)
        cv2.waitKey(0)

    for image in cornerIMG[0]:
        cv2.imshow("cannffycc", image)
        cv2.waitKey(0)
