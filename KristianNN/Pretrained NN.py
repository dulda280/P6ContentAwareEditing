# Import libraries

import os
import numpy
#from PIL import Image
import glob
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow import keras
from keras import layers
from keras import Model
from keras.applications.vgg16 import VGG16


# Path to trian and test pictures
path = "C:/Users/krell/OneDrive/Dokumenter/GitHub/P6ContentAwareEditing/KristianNN/train_landscape/*.jpg"
pathTest= "C:/Users/krell/OneDrive/Dokumenter/GitHub/P6ContentAwareEditing/KristianNN/test_landscape/*.jpg"
files = glob.glob(path)
files2 = glob.glob(pathTest)
# Size of img being fed to the network
img_size = 256
# Input and output arrays
y = []
x = []
# Array for test images
test = []
image_array = []
output_data = numpy.loadtxt("trainlandscape.csv")

print("Preparing train data...")
# saves images from file folder to an array
for i in files:
    image = cv2.imread(i)
    rescaled_image = cv2.resize(image, (img_size,img_size), interpolation=cv2.INTER_AREA)
    norm_image = cv2.normalize(rescaled_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    image_array.append(norm_image)
# same as above but the test images
for i in files2:
    image2 = cv2.imread(i)
    rescaled_image2 = cv2.resize(image2, (img_size,img_size), interpolation=cv2.INTER_AREA)
    norm_image2 = cv2.normalize(rescaled_image2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    test.append(norm_image2)

# Appends pictures into input array #useluss actually
for image in image_array:
    x.append(image)
# Saves the output data that is in train.csv into an array
for val in output_data:
    y.append(val)

# This first of all makes the array a numpy array, and it then reshapes the data, so that it can be fed to the network.
# For both test and train data.
x = np.array(x).reshape(-1,img_size,img_size,3)
test = np.array(test).reshape(-1,img_size,img_size,3)
y = np.array(y)


print(test[1][1][1])
model1 = VGG16(include_top=False, input_shape=(256, 256, 3))

flat1 = Flatten()(model1.layers[-1].output)
dense = Dense(64,kernel_initializer='normal', activation= 'relu')(flat1)
output = Dense(1,kernel_initializer='normal', activation= 'linear')(dense)

model1 = Model(inputs=model1.inputs, outputs=output)
# Define the loss and a
# ctivation function


opt = keras.optimizers.Adam(learning_rate=3e-5)
model1.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
# Fit the model with the desired batch size(runthroughs before the weights are updated), epochs(how many times the network goes through the layers)
# and validation split(split between train and test data).
print("Training network...")
model1.fit(x,y, batch_size=1, epochs= 15, validation_split = 0.2)
model1.summary()
# Use the trained network to make a prediction on some test images
prediction = model1.predict(test)
print("Predicted gamma values for test images:")
print(prediction)
gamma_images = []

# Below code is to apply the predicted gamma values to each picture and save them in a new folder.
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.9) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image,table)

test2 = []
print("Applying gamma values and saving pictures...")
for i in files2:
    image3 = cv2.imread(i)
    #rescaled_image3 = cv2.resize(image3, (1000,1000), interpolation=cv2.INTER_AREA)
    test2.append(image3)

for i in range(0,len(test2)):
    gammaImage = adjust_gamma(test2[i], gamma=prediction[i])
    gamma_images.append(gammaImage)
    cv2.imwrite('C:/Users/krell/OneDrive/Dokumenter/GitHub/P6ContentAwareEditing/KristianNN/redigeret/' + str(i) + '.jpg', gamma_images[i])
print("Saved Pictures in folder: RedigeretAfNetvaerk")