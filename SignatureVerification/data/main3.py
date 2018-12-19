import os
import cv2
import numpy as np
from keras.models import load_model
import keras.layers as layers
import keras.models as models

files = []
path_to_dataset = os.getcwd()
files.extend(os.listdir(path_to_dataset + "/dataset5"))
path = 'dataset5/'
images = []
for i in files:
    img = cv2.imread(path+i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (300,150))
    retval, img = cv2.threshold(img, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    images.append(img)
    print("Encoded "+str(i)+" images.")

images = np.array(images).reshape(90,300,150,1)
images = images / 255.


test_files = []
path_to_dataset = os.getcwd()
test_files.extend(os.listdir(path_to_dataset + "/dataset6"))
path = 'dataset6/'
test_images = []
for i in test_files:
    img = cv2.imread(path+i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (300,150))
    retval, img = cv2.threshold(img, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    test_images.append(img)
    print("Encoded "+str(i)+" test images.")


test_images = np.array(test_images).reshape(390,300,150,1)
test_images = test_images / 255.



img_width = 300
img_height = 150
batch = 128

def modelCreator():
    modelA = models.Sequential()
    modelA.add(layers.Conv2D(16, (3, 3), input_shape=(img_width, img_height, 1)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2, 2)))
    modelA.add(layers.Conv2D(32, (3, 3)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2, 2)))
    modelA.add(layers.Conv2D(64, (3, 3)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2, 2)))
    modelA.add(layers.Conv2D(128, (3, 3)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2, 2)))
    modelA.add(layers.Conv2D(256, (3, 3)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2, 2)))
    modelA.add(layers.Flatten())
    modelA.add(layers.Dense(batch))
    modelA.add(layers.Reshape((1, batch)))
    return modelA
    

modelu = modelCreator()

modelu.load_weights('model_weights.h5')

vectorized_train = modelu.predict(images)

final_dists = []
finale = []

for i in range(0, len(test_images)):
    j = 0
    if(i%13==0):
        j = j + 3
        x = vectorized_train[j:3+j]
    vectorized_i = modelu.predict(test_images[i].reshape(1, 300, 150, 1))
    vectorized_i = np.concatenate([vectorized_i, vectorized_i, vectorized_i])
    v = np.sum(np.square(x - vectorized_i))
    final_dists.append(v)
    if(v < 1015):
        finale.append("YES")
    else:
        finale.append("NO")
    
import pandas as pd
test_files_pd = pd.DataFrame(test_files)
yes_no_pd = pd.DataFrame(finale)