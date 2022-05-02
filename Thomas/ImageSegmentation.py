import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from msilib.schema import File
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = False 
model = VGG16(weights='imagenet', include_top=False)
from FileManager import *

fileManager = FileManager()

# Variables
imdir = "Thomas//images//image_directory"
targetdir = "Thomas//images//save_directory//"
number_clusters = 3

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, ('*.jpg' or '*.jpeg')))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data, use_multiprocessing=True))
    featurelist.append(features.flatten())

# Clustering
kmeans = KMeans(n_clusters=number_clusters, random_state=0, algorithm="elkan").fit(np.array(featurelist))

# Copy images renamed by cluster 
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, clusterGroup in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    #fileManager.save_image(targetdir, filelist[i], "cluster_group", str(i))
    if clusterGroup == 0:
        shutil.copy(filelist[i], targetdir + "//Cgroup_0//" + str(i) + "_" + "cgroup_" + str(clusterGroup) + "_"  + ".jpg")
    elif clusterGroup == 1:
        shutil.copy(filelist[i], targetdir + "//Cgroup_1//" + str(i) + "_" + "cgroup_" + str(clusterGroup) + "_"  + ".jpg")
    else:
        shutil.copy(filelist[i], targetdir + "//Cgroup_2//" + str(i) + "_" + "cgroup_" + str(clusterGroup) + "_"  + ".jpg")