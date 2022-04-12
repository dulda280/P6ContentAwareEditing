import os
import cv2 as cv
import queue


class EdgeDetection:
    image_directory = ""
    original_images_array = []

    def __init__(self):
        print(self)

    def import_images(self):
        for img in os.listdir(self.image_directory):
            img = cv.imread(os.path.join(self.image_directory, img))
            if img is not None:
                self.original_images_array.append(img)
        return self.original_images_array

    def find_connectivity(self, img, i, j):
        dx = [0, 0, 1, 1, 1, -1, -1, -1]
        dy = [1, -1, 0, 1, -1, 0, 1, -1]
        x = []
        y = []
        q = queue.Queue()
        if img[i][j] == 0:
            q.put((i, j))
        while q.empty() == False:
            u, v = q.get()
            x.append(u)
            y.append(v)
            for k in range(8):
                xx = u + dx[k]
                yy = v + dy[k]
                if img[xx][yy] == 0:
                    img[xx][yy] = 2
                    q.put((xx, yy))
        return x, y
