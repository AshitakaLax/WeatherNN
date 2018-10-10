from __future__ import division, print_function, absolute_import


import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import tflearn
from tflearn.datasets import mnist
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pickle

X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y = shuffle(X, Y)
trainX = X[0:50000]
trainY = Y[0:50000]
validX = X[50000:]
validY = Y[50000:]

# max pooling compresses the data down from usefull for images

# 