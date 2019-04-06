## Import Libraries
# To Load Dataset
import pickle as pkl
import numpy as np

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# For imageplots and colormaps
import matplotlib.cm as cm
import matplotlib.pyplot as plt


## Unpack and prep Dataset
with open('train_image.pkl', 'rb') as ti:
    train_X = pkl.load(ti)

with open('train_label.pkl', 'rb') as tl:
    train_y = pkl.load(tl)

with open('test_image.pkl', 'rb') as test:
    test_X = pkl.load(test)

# Lists -> arrays
train_X = np.asarray(train_X)
train_y = np.asarray(train_y)
test_X = np.asarray(test_X)

# Reshape X to matrix of 28x28 each
# -1 arg means self-inference
train_X = train_X.reshape(-1, 784)
test_X = test_X.reshape(-1, 784)
# Normalization
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255

# One-hot labels
train_y = to_categorical(train_y)
num_classes = train_y.shape[1]
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size = 0.4)
print("========================================================================")
print("Training Data:" + str(train_X.shape))
print("Validation Data:" + str(valid_X.shape))
print("Training Labels:" + str(train_y.shape))
print("Validation Labels:" + str(valid_y.shape))
print("========================================================================")

## Train
batch_size = 1
epochs = 10
type = 'relu'
model = Sequential()
model.add(Dense(128, activation = type, input_shape = (784, )))
model.add(Dense(128, activation = type))
model.add(Dense(64, activation = type))
model.add(Dense(32, activation = type))
model.add(Dense(train_y.shape[1], activation = 'softmax'))
model.compile(optimizer = 'sgd',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
model.summary()
train = model.fit(train_X, train_y, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (valid_X, valid_y))
