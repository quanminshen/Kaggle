import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging

from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# from nolearn.lasagne import BatchIterator
import tensorflow as tf

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'
FLOOKUP = 'data/IdLookupTable.csv'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] % (levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H: %M:%S',
                    filename='myapp.log',
                    filemode='w')


def load(test=False, cols=None):
    """
    Loads the dataset.
    Parameters
    :param test: optional, defaults to 'False'
                 Flag indicating if we need to load from 'FTEST'('True') or 'FTRAIN' ('False')
    :param cols: optional, defaults to 'None'
                 A list of columns you're interested in. If specified only reterns these columns.
    :return: A tuple of X and y, if 'test' was set to 'True' y contains 'None'. 
    """

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) # load pandas dataframe

    # The Image column has pixel values separated by space;
    # convert the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    # prints the number of values for each column
    print(df.count())
    # logging.info('The number of values for each column %d', df.count())
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[: -1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None
    return X, y


def plot_sample(x, y, axis):
    """
    Plots a single sample image with keypoints on top.
    :param x: Image data.
    :param y: Keypoints to plot.
    :param axis: Plot over which to draw the sample.
    :return: None
    """
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2]*48 + 48, y[1::2]*48 + 48, marker='x', s=10)


X, y = load()
img = X[11].reshape(96, 96)
plt.imshow(img, cmap='gray')
plt.show()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5)

# Predefined parameters
image_size = 96
num_keypoints = 30
batch_size = 36
num_epochs = 1001
learning_rate = 0.01
momentum = 0.9

model_name = "1fc_b" + str(batch_size) + "_e" + str(num_epochs - 1)
model_variable_scope = model_name
root_location = '/models/'
model_path = root_location + model_name + '/model.ckpt'
train_history_path = root_location = model_name + '/train_history'

os.makedirs(root_location + model_name + '/', exist_ok=True)









