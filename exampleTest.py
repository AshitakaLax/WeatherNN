
from __future__ import print_function
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import tflearn
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv, to_categorical

# The file containing the weather samples (including the column header)
WEATHER_SAMPLE_FILE = 'weather.csv'
#data, labels = load_csv(WEATHER_SAMPLE_FILE, target_column=12, categorical_labels=True, n_classes=2)
data, labels = load_csv(WEATHER_SAMPLE_FILE, target_column=11, columns_to_ignore=[0,1,2])

TrainingSetFeatures = data[:6000]
TestSetFeatures = data[6000:]
TrainingSetLabels = labels[:6000]
TestSetLabels = labels[6000:]

def preprocessor(data):
	for i in range(len(data)):
		#grab the date element
		
		hours = self.Date.hour
		dayOfYear = self.Date.timetuple().tm_yday
		# convert the numbers into complex value https://www.tensorflow.org/api_docs/python/tf/complex
		hourVectorReal = math.cos(2*math.pi * (hours/24))
		hourVectorImg = math.sin(2*math.pi * (hours/24))
		
		dayVectorReal = math.cos(2*math.pi * (dayOfYear/365))
		dayVectorImg = math.sin(2*math.pi * (dayOfYear/365))

def categorizeLabels(labels):
	for i in range(len(labels)):
		evSample = float(labels[i])
		if evSample > 4000:
			labels[i] = 4
		elif evSample > 3000:
			labels[i] = 3
		elif evSample > 2000:
			labels[i] = 2
		elif evSample > 1000:
			labels[i] = 1
		else:
			labels[i] = 0

categorizeLabels(TrainingSetLabels)
TrainingSetLabels = to_categorical(TrainingSetLabels, 5)
categorizeLabels(TestSetLabels)
TestSetLabels = to_categorical(TestSetLabels, 5)
#labels = np.shape(labels, (-1, 2))
data = np.array(data, dtype=np.float32)

#create a test set from the number of samples and traning set

print(data.shape)
net = tflearn.input_data(shape=[None, 8])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 5, activation="softplus")
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.1)
# categorized the data into bins for and that should be the number of 0.88888
#EBM_Audio_Classification DeepLearning
# Define model
model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True, run_id='WEATHER_1')

# Let's create some data for DiCaprio and Winslet
lowOutput =  [0, 9.92, 0.37, -0.01, 89.12, 4.72, 29.19, 29.98]
highOutput = [0, 10, 6.16, 1.26, 68.96, 0, 29.26, 30.05]

pred = model.predict([lowOutput, highOutput])

print("lowoutput estimate:", pred[0][1])
print("highOutput estimate:", pred[1][1])
