import os
from os import listdir
import cv2 as cv
import numpy as np
import binascii
import struct
from PIL import Image
import scipy
import scipy.misc
import scipy.cluster


class ImageProcessing:
    folder_dir = "car_pictures"  # image path/directory
    original_img = []
    resized_img = []
    scale_percent = 20  # percent of original size

    def import_images(self):
        for img in os.listdir(self.folder_dir):
            img = cv.imread(os.path.join(self.folder_dir, img))
            if img is not None:
                self.original_img.append(img)

        return self.original_img

    def rescale_images(self, print_debug: bool):
        for img in self.original_img:
            if print_debug:
                print('Original Dimensions : ', img.shape)

            # calculate new dimensions
            width = int(img.shape[1] * self.scale_percent / 100)
            height = int(img.shape[0] * self.scale_percent / 100)
            dimensions = (400, 200)

            # resize image
            img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.reshape((img.shape[1] * img.shape[0], 3))
            print(img)
            self.resized_img.append(img)
            if print_debug:
                print('Resized Dimensions : ', img.shape)

            # show image
            # cv.imshow("Resized image", img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        return self.resized_img

    def rescale_image(self, image, res_x, res_y):
        rescale_dimensions = (res_y, res_x)
        rescaled_image = cv.resize(image, rescale_dimensions, interpolation=cv.INTER_AREA)
        rescaled_image = cv.cvtColor(rescaled_image, cv.COLOR_BGR2RGB)
        return rescaled_image

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

    def find_all_pixels_ignore_black(self, image):
        output_image = image
        output_array = []

        for y in range(0, output_image.shape[0]):
            for x in range(0, output_image.shape[1]):
                if output_image[y, x].all() == 0:
                    output_array.append(output_image[y, x])

        return np.asarray(output_array)
