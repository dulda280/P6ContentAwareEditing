import cv2 as cv
import numpy as np


class AverageColors:

    debug = False

    # Calculate mean values for the R, G, and B colour channels
    def mean_values(self, img, square_size):
        hue = []
        sat = []
        val = []

        yStart = 0
        yStop = square_size

        averages = []
        hue_averages = []
        count = 0

        while yStop <= img.shape[0]:
            # reset x
            xStart = 0
            xStop = square_size

            while xStop <= img.shape[1]:

                for y in range(yStart, yStop):
                    for x in range(xStart, xStop):
                        hue.append(img[y, x][0])
                        sat.append(img[y, x][1])
                        val.append(img[y, x][2])

                # calculate HSV average
                average = [round(sum(hue) / len(hue), 0),
                           round(sum(sat) / len(sat), 0),
                           round(sum(val) / len(val), 0)]
                averages.append(average)
                hue_averages.append(round(sum(hue) / len(hue), 0))

                # empty arrays
                val = []
                sat = []
                hue = []
                count += 1

                # move to new area in x direction
                xStart += square_size
                xStop += square_size

            # move in y direction
            yStart += square_size
            yStop += square_size

        for y in range(0, img.shape[0] - 1):
            for x in range(0, img.shape[1] - 1):
                hue.append(img[y, x][0])
                sat.append(img[y, x][1])
                val.append(img[y, x][2])


        if self.debug == True:
            print(count)
            print(len(averages))
            print("All averages:", averages)
            #bgr_img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow("img", img)
            cv.waitKey(0)

        return averages, hue_averages

    def show_image(self, img, square_size, averages):

        yStart = 0
        yStop = square_size

        count = 0

        while yStop <= img.shape[0]:
            # reset x
            xStart = 0
            xStop = square_size

            while xStop <= img.shape[1]:

                for y in range(yStart, yStop):
                    for x in range(xStart, xStop):  # +1 ???
                        img[y, x] = averages[count]

                # to reach index 0-63
                if count < (len(averages)-1):  # len = 64
                    count += 1

                # move to new area in x direction
                xStart += square_size
                xStop += square_size

            # move in y direction
            yStart += square_size
            yStop += square_size

        if self.debug == True:
            print(count)
            #bgr_img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow("pixels", img)
            cv.waitKey(0)

    def main(self, directory):
        square_size = 8
        hueFeatureArr = []
        for img in directory:
            averages, hueAvg = self.mean_values(img, square_size)
            hueFeatureArr.append(hueAvg)
            #self.show_image(img, square_size, averages)

        return hueFeatureArr