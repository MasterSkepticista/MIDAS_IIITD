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

def extract():

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
    train_X = train_X.reshape(-1, 28, 28, 1)
    test_X = test_X.reshape(-1, 28, 28, 1)
    # Normalization
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255
    test_X = test_X / 255

    # One-hot labels
    train_y = to_categorical(train_y)
    num_classes = train_y.shape[1]
    return train_X, train_y, test_X, num_classes

def split(train_X, train_y):
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size = 0.3)
    return train_X, train_y, valid_X, valid_y

def data_summary(train_X, train_y, valid_X, valid_y):
    print("========================================================================")
    print("Training Data:" + str(train_X.shape))
    print("Validation Data:" + str(valid_X.shape))
    print("Training Labels:" + str(train_y.shape))
    print("Validation Labels:" + str(valid_y.shape))
    print("========================================================================")



def network(batch_size, epochs, train_X, train_y, valid_X, valid_y):
    ## Train
    kernel_size = (3, 3)
    model = Sequential()
    model.add(Conv2D(16, kernel_size, activation = 'linear', input_shape = (28,28,1), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D((2,2), padding = 'same'))
    model.add(Conv2D(32, kernel_size, activation = 'linear', padding = 'same'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D((2,2), padding = 'same'))
    model.add(Flatten())
    model.add(Dense(64, activation = 'linear'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.sgd(), metrics = ['accuracy'])
    model.summary()

    history = model.fit(train_X, train_y, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (valid_X, valid_y))
    test_y = model.predict_classes(test_X)

    return history, test_y

def plots(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


train_X, train_y, test_X, num_classes = extract()
train_X, train_y, valid_X, valid_y = split(train_X, train_y)
data_summary(train_X, train_y, valid_X, valid_y)
history, predictions = network(1, 3, train_X, train_y, valid_X, valid_y)
plots(history)
print(predictions)
