import cv2
from skimage.exposure import is_low_contrast


class DetectLowContrast:
    # construct the argument parser and parse the arguments
    def is_low_contrast(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if is_low_contrast(gray, 0.35):
            return True
        else:
            return False
