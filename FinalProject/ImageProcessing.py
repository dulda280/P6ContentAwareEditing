import os
from os import listdir
import cv2 as cv
import numpy as np
import binascii
import struct
#from PIL import Image
import scipy
import scipy.misc
import scipy.cluster
np.seterr(divide='ignore', invalid='ignore')


class ImageProcessing:

    def count_bw_pixels(self, image, index):
        number_of_white_pix = np.sum(image == 255)
        number_of_black_pix = np.sum(image == 0)
        return f"Picture_{index} (  {number_of_white_pix} White pixels  ||  {number_of_black_pix} Black pixels  )"

    def connected_component_labelling(self, image):
        # Converting those pixels with values 1-127 to 0 and others to 1
        img = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1]

        # Applying cv2.connectedComponents()
        num_labels, labels = cv.connectedComponents(img)

        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0

        return cv.cvtColor(labeled_img, cv.COLOR_BGR2RGB), num_labels

    # def get_dominant_color(self, pil_img):
    #     img = pil_img.copy()
    #     img = Image.fromarray(np.uint8(img))
    #     img = img.convert("RGB")
    #     img = img.resize((1, 1), resample=0)
    #     dominant_color = img.getpixel((0, 0))
    #     return dominant_color
    def most_frequent_color(self, image):
        NUM_CLUSTERS = 5

        img = image.copy()
        ar = np.asarray(img)
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        #print('cluster centres:\n', codes)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences

        index_max = scipy.argmax(np.where(counts != 0))  # find most frequent
        peak = codes[index_max]
        colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
        #print('most frequent is %s (#%s)' % (peak, colour))
        return peak, colour

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.9) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv.LUT(image, table)

    def find_all_pixels_ignore_black(self, image):
        output_image = image
        output_array = []

        for y in range(0, output_image.shape[0]):
            for x in range(0, output_image.shape[1]):
                if output_image[y, x].all() == 0:
                    output_array.append(output_image[y, x])

        return np.asarray(output_array)
