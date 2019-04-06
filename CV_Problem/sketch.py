## Import Libraries
# To Load Dataset
import pickle as pkl
import numpy as np

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


print(train_y)
plt.imshow(train_X[2].reshape(28, 28), cmap = cm.Greys_r)
plt.title(train_y[2])
plt.show()
