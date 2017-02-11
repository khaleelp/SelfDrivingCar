#Model file

#import libraries
import csv
import pickle
import numpy as np

import os
import json

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Flatten, ELU
from sklearn.preprocessing import LabelBinarizer

from sklearn.utils import shuffle

ch, row, col = 3, 160, 320  # camera format


with open('data/driving_data.p', mode='rb') as f:
	driving_data = pickle.load(f)

images = driving_data['images']
labels = driving_data['labels']

print(images[0].shape)

# shuffle a dataset
images, labels = shuffle(images, labels)

# split train & valid data
X_train, X_validation, y_train, y_validation = train_test_split(images, labels, test_size=0.1, random_state=42)

#Comma.ai model
#Source: https://github.com/commaai/research/blob/master/train_steering_model.py
def commaai_model(time_len=1):

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model


#Generator, and training
def generator(X_train, y_train, batch_size):
    batch_train = np.zeros((batch_size, row, col, 3))
    batch_angle = np.zeros(batch_size)
    while True:
        for i in range(batch_size):
            batch_train[i] = X_train[i]
            batch_angle[i] = y_train[i]
        yield batch_train, batch_angle



batch_size = 256
EPOCHS = 10

model = commaai_model()

model.summary()

# Compile model using Adam optimizer
# and loss computed by mean squared error
model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

samples_generator = generator(X_train, y_train, batch_size)

# Model training
# TODO: Add validation data
history = model.fit_generator(
    samples_generator,
    samples_per_epoch=128*188,
    nb_epoch=EPOCHS,
    verbose=1
)

#score = model.evaluate(X_test, y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
print("Model fit")

#Save the model
model_json = 'model.json'
model_weights = 'model.h5'

json_string = model.to_json()
try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass

with open(model_json, 'w') as jfile:
    json.dump(json_string, jfile)
model.save_weights(model_weights)
print("Model Saved!")




