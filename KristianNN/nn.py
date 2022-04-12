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

path = "C:/Users/krell/PycharmProjects/P6ContentAwareEditing/KristianNN/train/*.jpg"
pathTest= "C:/Users/krell/PycharmProjects/P6ContentAwareEditing/KristianNN/test/*.jpg"
files = glob.glob(path)
files2 = glob.glob(pathTest)
img_size = 128
y = []
x = []
test = []
image_array = []
output_data = numpy.loadtxt("train.csv")


for i in files:
    image = cv2.imread(i)
    rescaled_image = cv2.resize(image, (img_size,img_size), interpolation=cv2.INTER_AREA)
    norm_image = cv2.normalize(rescaled_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    image_array.append(norm_image)

for i in files2:
    image2 = cv2.imread(i)
    rescaled_image2 = cv2.resize(image2, (img_size,img_size), interpolation=cv2.INTER_AREA)
    norm_image2 = cv2.normalize(rescaled_image2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    test.append(norm_image2)


for image in image_array:
    x.append(image)

for val in output_data:
    y.append(val)


x = np.array(x).reshape(-1,img_size,img_size,3)
test = np.array(test).reshape(-1,img_size,img_size,3)
y = np.array(y)


model = Sequential()
model.add(Conv2D(64, (3,3), kernel_initializer='normal', input_shape = x.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3),kernel_initializer='normal', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='normal',activation='relu'))


model.add(Dense(1, kernel_initializer='normal',activation='linear'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.fit(x,y, batch_size=1, epochs= 30, validation_split = 0.1)

prediction = model.predict(test)
print(prediction)
gamma_images = []
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.9) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image,table)

test2 = []
for i in files2:
    image3 = cv2.imread(i)
    rescaled_image3 = cv2.resize(image3, (1000,1000), interpolation=cv2.INTER_AREA)
    test2.append(rescaled_image3)

for i in range(0,len(test2)):
    gammaImage = adjust_gamma(test2[i], gamma=prediction[i])
    gamma_images.append(gammaImage)
    cv2.imwrite('C:/Users/krell/PycharmProjects/Convulutionalneurnalenrtnenrewo/redigeretnn/' + str(i) + '.jpg', gamma_images[i])
