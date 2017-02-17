#imports
import csv
import cv2
import sys
import pickle
import random
import numpy as np

import os
import json

import matplotlib.pyplot as plt

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Flatten, ELU

from keras.regularizers import l2

from sklearn.utils import shuffle

#ch, row, col = 3, 160, 320
channels, rows, columns = 3, 66, 200  # custom camera format used in Nvidia model

#Fetch data from pickle file
with open('data/driving_data.p', mode='rb') as f:
    driving_data = pickle.load(f)

center_images = driving_data['center_images']
left_images = driving_data['left_images']
right_images = driving_data['right_images']
labels = driving_data['labels']

#shuffle a dataset
#images, labels = shuffle(center_images, labels)

# split train & valid data
X_train, X_validation, y_train, y_validation = train_test_split(center_images, labels, test_size=0.1, random_state=42)

print(len(X_train))
print(len(X_validation))

#custom rows and columns to fit Nvidia model
custom_rows, custom_cols = 66, 200


def flip_image(image, angle):
    if random.randint(0, 1):
        return cv2.flip(image, 1), -angle
    else:
        return image, angle


def rotate_image(image, angle):
    rows, cols, channel = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.uniform(-5, 5), 1)
    return cv2.warpAffine(image, M, (cols, rows)), angle


def brightness_image(image, steering):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.3, 1.2)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), steering


def cut_image(img):
    rows, cols, channel = img.shape
    top = int(.4 * rows)
    botton = int(.85 * rows)
    border = int(.05 * cols)
    return img[top:botton, border:cols - border, :]


# Crop image as required
# http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
def crop_image(image, y1, y2, x1, x2):
    return image[y1:y2, x1:x2]


# translate image and compensate for the translation on the steering angle
def translate_image(image, steering, horz_range=30, vert_range=5):
    rows, cols, chs = image.shape
    tx = np.random.randint(-horz_range, horz_range + 1)
    ty = np.random.randint(-vert_range, vert_range + 1)
    steering = steering + tx * 0.004  # multiply by steering angle units per pixel
    tr_M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, tr_M, (cols, rows), borderMode=1)
    return image, steering


def augment_image(image, angle):
    # Rotate image
    # image, angle = rotate_image(image,angle)

    # Crop
    image = crop_image(image, 20, 140, 50, 270)

    # Translate
    image, steering = translate_image(np.copy(image), angle)

    # Flip
    image, angle = flip_image(np.copy(image), angle)

    # Brightness
    # image, angle = brightness_image(np.copy(image),angle)

    # Data normalization
    #image = (image - 128.0) / 128.0

    return image, angle


def get_image(center_images, left_images, right_images, labels, index, image_offset=0.20):

    camera = np.random.choice(['center', 'left', 'right'])

    if camera == 'center':
        image, steering = plt.imread("data/"+center_images[index]), float(labels[index])
    elif camera == 'left':
        image, steering = plt.imread("data/"+left_images[index]), float(labels[index]) + image_offset
    elif camera == 'right':
        image, steering = plt.imread("data/"+right_images[index]), float(labels[index]) - image_offset

    # Augment image
    image, angle = augment_image(image, steering)
    # Resize image
    image = cv2.resize(image, (custom_cols, custom_rows))
    image = np.reshape(image, (1, custom_rows, custom_cols, channels))
    return image, steering


# Custom generator for model training
def my_generator(center_images, left_images, right_images, labels, batch_size):
    batch_train = np.zeros((batch_size, rows, columns, 3))
    batch_angle = np.zeros(batch_size)
    while True:
        for index in range(batch_size):
            image, angle = get_image(center_images, left_images, right_images, labels, index)
            batch_train[index], batch_angle[index] = image, angle
        yield batch_train, batch_angle


def commaai_model(time_len=1):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(channels, rows, columns),
                     output_shape=(channels, rows, columns)))
    model.add(Convolution2D(16, 8, 8, input_shape=(3, 160, 320), subsample=(4, 4), border_mode="same"))
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


def nvidia_model(time_len=1):
    INIT = 'glorot_uniform'  # 'he_normal', glorot_uniform
    keep_prob = 0.2
    reg_val = 0.01

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(rows, columns, channels),
                     output_shape=(rows, columns, channels)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT, W_regularizer=l2(reg_val)))

    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(ELU())

    model.add(Dense(1))

    # model.compile(optimizer="adam", loss="mse") # , metrics=['accuracy']

    return model


# Train the model
batch_size = 64
EPOCHS = 20

model = nvidia_model()
#model.summary()

# Compile model using Adam optimizer and loss computed by mean squared error
model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'],
              validation_data=(X_validation, y_validation),
              verbose=1)

my_generator = my_generator(center_images, left_images, right_images, labels, batch_size)
history = model.fit_generator(
    my_generator,
    samples_per_epoch=batch_size*(int((len(center_images)/batch_size))-1), # of training samples
    nb_epoch=EPOCHS,
    verbose=1
)
print("model training complete")

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
model.save(model_weights)

print("model saved")