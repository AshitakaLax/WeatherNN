
from __future__ import print_function
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import tflearn
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv

# The file containing the weather samples (including the column header)
WEATHER_SAMPLE_FILE = 'weather.csv'
#data, labels = load_csv(WEATHER_SAMPLE_FILE, target_column=12, categorical_labels=True, n_classes=2)
data, labels = load_csv(WEATHER_SAMPLE_FILE, target_column=11, has_header=True, n_classes=16)
def preprocess(data, columns_to_ignore):
	# Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[0, 1, 2]

# Preprocess data
data = preprocess(data, to_ignore)
net = tflearn.input_data(shape=[None, 8])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 1, activation="linear")
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, show_metric=True)

# Let's create some data for DiCaprio and Winslet
lowOutput =  [0, 9.92, 0.37, -0.01, 89.12, 4.72, 29.19, 29.98]
highOutput = [0, 10, 6.16, 1.26, 68.96, 0, 29.26, 30.05]

pred = model.predict([lowOutput, highOutput])

print("lowoutput estimate:", pred[0][1])
print("highOutput estimate:", pred[1][1])
