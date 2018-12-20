import os 
import cv2
import keras.layers as layers
import keras.models as models
import numpy as np
import tensorflow as tf
from keras.regularizers import l2

img_width = 300
img_height = 150
batch = 128

def modelCreator():
    modelA = models.Sequential()
    modelA.add(layers.Conv2D(16, (3,3), input_shape = (img_width, img_height, 1)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2,2)))
    modelA.add(layers.Conv2D(32, (3,3)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2,2)))  
    modelA.add(layers.Conv2D(64, (3,3)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2,2)))
    modelA.add(layers.Conv2D(128, (3,3)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2,2)))
    modelA.add(layers.Conv2D(256, (3,3)))
    modelA.add(layers.BatchNormalization())
    modelA.add(layers.Activation("relu"))
    modelA.add(layers.MaxPooling2D((2,2)))     
    modelA.add(layers.Flatten())
    modelA.add(layers.Dense(batch))
    modelA.add(layers.Reshape((1, batch)))
    return modelA

modelu = modelCreator()


modelu.summary()
X1 = layers.Input((img_width,img_height,1))
X2 = layers.Input((img_width,img_height,1))
my1 = modelu(X1)
my2 = modelu(X2)
myo = layers.Subtract()([my1, my2])
m = layers.Dense(1, activation = "sigmoid")(myo)
model = models.Model(inputs = [X1, X2], outputs = m)


model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ['accuracy'])
model.summary()

path_to_dataset = os.getcwd() + "\\dataset"
#For training
anchor_main = []
real_forged_main = []
y_list = []
for x in range(1, 5):
    if(x != 2):
        forge = []
        forge.extend(os.listdir(path_to_dataset+str(x)+ "\\forge"))
        forge = np.array(forge)
        
        real = []
        real.extend(os.listdir(path_to_dataset+str(x)+"\\real"))
        real = np.array(real)
        
        forged_images = []
        t = 60
        z = 5
        if (x == 3):
            t = 150
        if (x == 5):
            t = 90
            z = 24
    
        path = 'dataset'+str(x)+'/forge/'
        for i in range(0,t):
            img = cv2.imread(path + str(forge[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_width,img_height))
            cv2.imwrite('save1.png', img)
            retval, img = cv2.threshold(img, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            cv2.imwrite('save2.png', img)
            forged_images.append(img)
            print("Encoded "+str(i)+" images.")
        
        forged_images = np.array(forged_images)
        
        real_images = []
        path = 'dataset'+str(x)+'/real/'
        for i in range(0,t):
            img = cv2.imread(path + str(real[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_width,img_height))
            retval, img = cv2.threshold(img, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            real_images.append(img)
            print("Encoded "+str(i)+" images.")
        
        real_images = np.array(real_images)
        
        anchor_images = []
        path = 'dataset'+str(x)+'/real/'
        for i in range(0,t):
            if i%z == 0:
                for j in range(0,z):
                    img = cv2.imread(path + str(real[i]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (img_width,img_height))
                    retval, img = cv2.threshold(img, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    anchor_images.append(img)
            print("Encoded "+str(i)+" images. Added 5 times")
        
        anchor_images = np.array(anchor_images)
        
        real_images = real_images / 255.
        forged_images = forged_images / 255.
        anchor_images = anchor_images / 255.
        real_images = real_images.reshape(real_images.shape[0], img_width, img_height, 1)
        forged_images = forged_images.reshape(forged_images.shape[0], img_width, img_height, 1)
        anchor_images = anchor_images.reshape(anchor_images.shape[0], img_width, img_height, 1)
        anchor_images = np.concatenate([anchor_images, anchor_images], axis = 0)
        real_forged_X = np.concatenate([real_images, forged_images], axis = 0)
        
        y = np.concatenate([np.ones((t, 1)), np.zeros((t,1))], axis = 0)
        y = y.reshape(len(y), 1, 1)
        y_list.append(y)
        anchor_main.append(anchor_images)
        real_forged_main.append(real_forged_X)
        
anchor_main = np.vstack(anchor_main)
real_forged_main = np.vstack(real_forged_main)
y = np.vstack(y_list)
model.fit([anchor_main,real_forged_main], y, epochs = 30, batch_size = 32, shuffle = True)

anchor_main = []
real_forged_main = []
#For Testing
for x in range(2, 3):
    forge = []
    forge.extend(os.listdir(path_to_dataset+str(x)+ "\\forge"))
    forge = np.array(forge)
    
    real = []
    real.extend(os.listdir(path_to_dataset+str(x)+"\\real"))
    real = np.array(real)
    
    forged_images = []
    k = 0
    t = 60
    bottom = 5
    if(x == 6):
        k = 0
        t = 288
        bottom = 24
    if(x == 5):
        k = 0
        t = 1032
    
    path = 'dataset'+str(x)+'/forge/'
    for i in range(k,t):
        img = cv2.imread(path + str(forge[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_width,img_height))
        retval, img = cv2.threshold(img, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        forged_images.append(img)
        print("Encoded "+str(i)+" images.")
    
    forged_images = np.array(forged_images)
    
    real_images = []
    path = 'dataset'+str(x)+'/real/'
    for i in range(k,t):
        img = cv2.imread(path + str(real[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_width,img_height))
        retval, img = cv2.threshold(img, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        real_images.append(img)
        print("Encoded "+str(i)+" images.")
    
    real_images = np.array(real_images)
    
    anchor_images = []
    path = 'dataset'+str(x)+'/real/'
    for i in range(k,t):
        if i%bottom == 0:
            for j in range(0,bottom):
                img = cv2.imread(path + str(real[i]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (img_width,img_height))
                retval, img = cv2.threshold(img, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                anchor_images.append(img)
        print("Encoded "+str(i)+" images. Added "+str(bottom)+" times")
    
    anchor_images = np.array(anchor_images)
    
    real_images = real_images / 255.
    forged_images = forged_images / 255.
    anchor_images = anchor_images / 255.
    real_images = real_images.reshape(real_images.shape[0], img_width, img_height, 1)
    forged_images = forged_images.reshape(forged_images.shape[0], img_width, img_height, 1)
    anchor_images = anchor_images.reshape(anchor_images.shape[0], img_width, img_height, 1)
    

##C1 image will be from our database
##C2 is current image which is to be checked for real or fake

#Finding appropriate threshold by searching


t = 3
x = 0
l = []
k1 = 60
k2 = 5
values = []

for t in np.arange(1, 3, 1):
    x = 0
    l = []
    for i in range(0, k1):
        if(i%k2 == 0):
            print(str(int(i/k2)) + "\n")
            y0 = modelu.predict(real_images[i].reshape(1,img_width,img_height,1))
        y1 = modelu.predict(forged_images[i].reshape(1,img_width,img_height,1))
        y_pred = np.sum(np.square((y0 - y1)))
        l.append(y_pred)
        if(y_pred>t):
            x += 1
        print(y_pred)
    print(x)
    
    z = 0
    for n in l:
        z += n
        
    a1 = z/len(l)
    
    ################################################
    values.append(x)
    x = 0
    l2 = []
    for i in range(0, k1):
        if(i%k2 == 0):
            print(str(int(i/k2)) + "\n")
            y0 = modelu.predict(real_images[i].reshape(1,img_width,img_height,1))
        y1 = modelu.predict(real_images[i].reshape(1,img_width,img_height,1))
        y_pred = np.sum(np.square((y0 - y1)))
        l2.append(y_pred)
        if(y_pred<t):
            x += 1
        print(y_pred)
    print(x)
    values[-1] = values[-1] + x
    
    print(values[-1])
    z = 0
    for t in l2:
        z += t
        
    a2 = z/len(l2)
    print(str((a1+a2)/2))


## To find threshold
values = []
for t in np.arange(650, 1000, 1):
    x = 0
    for i in l:
        if(i>t):
            x += 1
        print(i)
    print(x)
    
    z = 0
    for a4 in l:
        z += a4
        
    a1 = z/len(l)
    
    ################################################
    values.append(x)
    x = 0
    for i in l2:
        if(i<t):
            x += 1
        print(i)
    print(x)
    values[-1] = values[-1] + x
    
    print(values[-1])
    z = 0
    for a4 in l2:
        z += a4
        
    a2 = z/len(l2)
    print(str((a1+a2)/2))

modelu.save_weights("model_weights.h5")