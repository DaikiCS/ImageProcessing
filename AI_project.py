import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.callbacks import TensorBoard

import numpy as np
import time
import pickle

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATADIR = "/Users/daikiitoh/Desktop/Ai_project/flowers" # type in your directory where data is
CATAGORIES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
IMG_SIZE = 128

#pre_processing data
img_data = []

def create_img_data():
    for category in CATAGORIES:
        path = os.path.join(DATADIR, category)
        class_label = CATAGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                img_data.append([new_array, class_label])
            except Exception as e:
                pass

create_img_data()

#shuffle data

import random

random.shuffle(img_data)

# create numpy array of input data
X = []
y = []

for features, label in img_data:
    X.append(features)
    y.append(label)
  
y = np.array(y)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# save data
import pickle

pickle_out = open("data.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("label.pickle", "wb")
pickle.dump(y,pickle_out)
pickle_out.close()

NAME = "Flowers-CNN-32x3-dense32-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

data = pickle.load(open("data.pickle", "rb"))
labels = pickle.load(open("label.pickle", "rb"))

data = data/255.0

# split test data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, shuffle= False)

#create model

model = Sequential()
model.add(Conv2D(32, (4,4),input_shape=data.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (4,4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(.05))

model.add(Conv2D(64, (4,4), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (4,4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 64
epochs = 10
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks = [tensorboard])

#model.save("32x3-CNN-Flowers.model")