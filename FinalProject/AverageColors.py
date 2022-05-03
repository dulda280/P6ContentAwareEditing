import cv2 as cv
import numpy as np


class AverageColors:

    debug = True

    # Calculate mean values for the R, G, and B colour channels
    def mean_values(self, img, square_size):
        red = []
        green = []
        blue = []

        yStart = 0
        yStop = square_size

        averages = []
        count = 0

        while yStop <= img.shape[0]:
            # reset x
            xStart = 0
            xStop = square_size

            while xStop <= img.shape[1]:

                for y in range(yStart, yStop):
                    for x in range(xStart, xStop):
                        red.append(img[y, x][2])
                        green.append(img[y, x][1])
                        blue.append(img[y, x][0])

                # calculate average
                average = [round(sum(blue) / len(blue), 0),
                           round(sum(green) / len(green), 0),
                           round(sum(red) / len(red), 0)]
                averages.append(average)

                # empty arrays
                red = []
                green = []
                blue = []
                count += 1

                # move to new area in x direction
                xStart += square_size
                xStop += square_size

            # move in y direction
            yStart += square_size
            yStop += square_size

        for y in range(0, img.shape[0] - 1):
            for x in range(0, img.shape[1] - 1):
                red.append(img[y, x][2])
                green.append(img[y, x][1])
                blue.append(img[y, x][0])

        if self.debug == True:
            print(count)
            print(len(averages))
            bgr_img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow("img", bgr_img)
            cv.waitKey(0)

        return averages

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
            bgr_img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow("pixels", bgr_img)
            cv.waitKey(0)

    def main(self, directory):
        square_size = 8
        for img in directory:
            averages = self.mean_values(img, square_size)
            self.show_image(img, square_size, averages)