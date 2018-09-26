# CLASS: CS5890
# AUTHOR: Levi Balling
# DATE: Sep 22, 2018
# SUMMARY: Data Parser to scan through the various raw data

# module for parsing CSV data sets
from __future__ import print_function
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import math
from IPython import display
from sklearn import metrics
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# The file containing the weather samples (including the column header)
WEATHER_SAMPLE_FILE = 'weather.csv'

WeatherList = pd.read_csv(WEATHER_SAMPLE_FILE, sep=',')
print(WeatherList.head())
print(WeatherList.describe())

weather_feature = WeatherList[["visibility"]]
weather_feature_columns = [tf.feature_column.numeric_column("visibility")]
weather_targets = WeatherList["solar_energy"]
weather_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
weather_optimizer = tf.contrib.estimator.clip_gradients_by_norm(weather_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
weather_linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=weather_feature_columns,
    optimizer=weather_optimizer
)

def weather_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

linear_regressor = tf.estimator.LinearRegressor(
	weather_feature_columns,
	optimizer=weather_optimizer
)

linear_regressor.train(
    input_fn = lambda:weather_input_fn(weather_feature, weather_targets),
    steps=100
)

# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: weather_input_fn(weather_feature, weather_targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, weather_targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

min_weather_value = WeatherList["visibility"].min()
max_weather_value = WeatherList["visibility"].max()
min_max_difference = max_weather_value - min_weather_value

print("Min. visibility Value: %0.3f" % min_weather_value)
print("Max. visibility Value: %0.3f" % max_weather_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)