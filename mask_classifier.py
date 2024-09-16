# -*- coding: utf-8 -*-
"""Mask Classifier
"""

import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import cv2
import tensorflow_datasets as tfds

from google.colab import drive
drive.mount('/content/drive')

"""# Preprocessing

## Setting up Paths
"""



data_folder = "../Face Tracking with Masks/Data/experiements/data/"

os.listdir(data_folder)

with_mask_path = data_folder + "with_mask/"
without_mask_path = data_folder + "without_mask/"

with_files = [[with_mask_path + i, 1] for i in os.listdir(with_mask_path)]
without_files = [[without_mask_path + i, 0] for i in os.listdir(without_mask_path)]

import random

paths = with_files + without_files
random.shuffle(paths)

"""## Setting up a strategy to load and normalize image files"""

IMG_SIZE = (400, 350, 3)

def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

def load_img(path):
  img = cv2.imread(path[0]) # uses opencv to read the image in the file
  img_pil = Image.fromarray(img)
  reshaped = np.array(img_pil.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.ANTIALIAS))

  return scale(reshaped), path[1]

"""## Creating Data Generators"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import ReLU, MaxPooling2D, Flatten, Dense, Dropout

def batch(arr, n=32):
    l = len(arr)
    for ndx in range(0, l, n):
        yield arr[ndx:min(ndx + n, l)]

def load_batch(arr):
  x, y = [], []
  for i in arr:
    temp = load_img(i)
    x.append(temp[0])
    if temp[1] == 0:
      y.append([1, 0])
    else:
      y.append([0,1])

  return np.array(x), np.array(y)

split = int(0.9 * len(paths))
train_paths, test_paths = paths[:split], paths[split:]

EPOCHS = 3
BATCH_SIZE = 32

class ANN():
  def __init__(self, n, m):
    self.name = n
    self.model = m

  def train_model(self, paths, epochs=3, batch_size=32):
    for i in range(epochs):
      random.shuffle(paths)

      batches = batch(paths, batch_size)

      print("Epoch: " + str(i+1))
      print()
      for j, b in enumerate(batches):
        print("\nBatch: " + str(j+1))

        x, y = load_batch(b)

        self.model.fit(x, y, verbose=1)

        #no data leaks

        del x, y
      print()

  def eval_model(self, paths):
    x, y = load_batch(paths)
    preds = self.model.predict(x)
    for index, pred in enumerate(preds):
      for i, item in enumerate(pred):
        if item < .5:
          preds[index][i] = 0
        else:
          preds[index][i] = 1
    sum = 0
    for i in range(len(y)):
      if preds[i][0] == y[i][0] and preds[i][1] == y[i][1]:
        sum += 1
    return sum, len(y)

  def save_model(self, path):
    model.save("../content/drive/My Drive/Kaggle Team/2021/Face Tracking with Masks/our_models/" + this.name + ".h5")



"""# Define Models

### Xception
"""

input_t = tf.keras.Input(shape=(400, 350, 3))
xception = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_tensor=input_t,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)

for layer in xception.layers[:102]: layer.trainable = False


xcept = ANN("Transfer REs18", tf.keras.models.Sequential())

xcept.model.add(xception)
xcept.model.add(tf.keras.layers.Flatten())
xcept.model.add(tf.keras.layers.Dense(2, activation="softmax"))
xcept.model.summary()

for i, l in enumerate(xception.layers):
  print(str(i) + " " + str(l))

xcept.model.compile(loss='categorical_crossentropy', optimizer="Adam")

xcept.train_model(train_paths, epochs=1)

print(xcept.eval_model(test_paths))

"""### VGG16
"""

# https://github.com/qubvel/classification_models

input_t = tf.keras.Input(shape=(400, 350, 3))
v16 = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=input_t,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
)

for layer in v16.layers[:19]: layer.trainable = False



v16mod = ANN("vgg16", tf.keras.models.Sequential())

v16mod.model.add(v16)
v16mod.model.add(tf.keras.layers.Flatten())
v16mod.model.add(tf.keras.layers.Dense(2, activation="softmax"))

v16mod.model.summary()

v16mod.model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
hist=v16mod.train_model(train_paths, epochs=EPOCHS)

"""### DenseNet

https://keras.io/api/applications/densenet/
"""

input_tensor = tf.keras.Input(shape=(400, 350, 3))
denseNet = tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_tensor=input_tensor,
    input_shape=None,
    pooling=None,
    classes=2,
)

for layer in denseNet.layers[:251]:
  layer.trainable = False

dNet = ANN("DenseNet121",tf.keras.models.Sequential())

dNet.model.add(denseNet)
dNet.model.add(tf.keras.layers.Flatten())
dNet.model.add(tf.keras.layers.Dense(2, activation="softmax"))

dNet.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

dNet.train_model(train_paths, epochs=2)

for i, l in enumerate(denseNet.layers):
  print(str(i) + " " + str(l))



