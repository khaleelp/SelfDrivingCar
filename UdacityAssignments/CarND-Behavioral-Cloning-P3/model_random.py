import os
import cv2
import pandas
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
import cv2

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from keras.layers.pooling import MaxPooling2D
import numpy as np
from random import random



########### Utility functions ############

def sampling2(data):
    st = data[:,3].astype(np.float32)
    bins, num = 100, 100
    _, b = np.histogram(st, bins)
    dist = np.digitize(st, b)
    return np.concatenate([data[dist==rng][:num] for rng in range(bins)])

def read_image(f):
    image = cv2.cvtColor(cv2.imread(os.path.join(dir, f)), cv2.COLOR_BGR2YUV)
    return image

#### Read out images
def read_images(balanced_feats, balanced_labels):
    image_features = np.array([read_image(f) for f in balanced_feats])
    image_labels = balanced_labels
    samples = np.random.choice(image_features.shape[0], 3)
    return image_features, image_labels


##### Flip out images
def flip_images(image_features, image_labels):
    flip_limit = 0.0
    flip = np.logical_or(image_labels < -flip_limit, image_labels > flip_limit)
    flipped_features = np.append(image_features, [np.fliplr(f) for f in image_features[flip]], axis=0)
    flipped_labels = np.append(image_labels, -image_labels[flip], axis=0)

    return flipped_features, flipped_labels


#### Translate images
def translate(img, angle):
    TRANS_X_RANGE = 100  # Number of translation pixels up to in the X direction for augmented data (-RANGE/2, RANGE/2)
    TRANS_Y_RANGE = 40  # Number of translation pixels up to in the Y direction for augmented data (-RANGE/2, RANGE/2)
    TRANS_ANGLE = .3  # Maximum angle change when translating in the X direction
    x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
    new_angle = angle + ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE
    y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)

    # Form the translation matrix
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

    # Translate the image
    im = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    return im, new_angle

def translate_images(flipped_features, flipped_labels):
    trans_limit = 0.0
    trans = np.logical_or(flipped_labels < -trans_limit, flipped_labels > trans_limit)

    translated_labels = []
    translated_features = []
    for f, a in zip(flipped_features[trans], flipped_labels[trans]):
        tf, ta = translate(f, a)
        translated_features.append(tf)
        translated_labels.append(ta)

    translated_labels = np.array(translated_labels)
    translated_features = np.array(translated_features).reshape(-1, 160,320,3)

    return translated_features, translated_labels

def prune_images(features, labels, n=1000):
    samples = np.random.choice(features.shape[0], n, replace=False)
    pruned_features, pruned_labels = features[samples], labels[samples]
    _ = plt.hist(pruned_labels, bins = 100)
    return pruned_features, pruned_labels

#############################################################



dir = "./data"
with open(os.path.join(dir, 'driving_log.csv'), 'r') as f:
    data = pandas.read_csv(f, header=0, skipinitialspace=True).values
    print(data.shape)




sampled_data = sampling2(data)
print(sampled_data.shape)

c, l, r, st, _, _, _ = np.split(sampled_data, 7, axis=1)
adj = 0.25
balanced_feats = np.append(c, l)
balanced_feats = np.append(balanced_feats, r)
balanced_labels = np.append(st, st + adj)
balanced_labels = np.append(balanced_labels, st - adj)
print(balanced_feats.shape, balanced_labels.shape)


image_features, image_labels = read_images(balanced_feats, balanced_labels)
flipped_features, flipped_labels = flip_images(image_features, image_labels)
translated_features, translated_labels = translate_images(flipped_features, flipped_labels)
features = np.append(flipped_features, translated_features, axis=0)
labels = np.append(flipped_labels, translated_labels, axis=0)
X_train, y_train = shuffle(features, labels)
#pruned_features, pruned_labels = prune_images(features, labels, 1000)
#X_train, y_train = shuffle(pruned_features, pruned_labels)


def my_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=X_train.shape[1:]))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', name='c1'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', name='c2'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', name='c3'))
    model.add(Dropout(0.2))    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu', name='c4'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu', name='c5'))
    model.add(Dropout(0.2))    

    model.add(Flatten())
    model.add(Dense(100, name='d1', activation='relu'))
    model.add(Dense(50, name='d2', activation='relu'))
    model.add(Dense(10, name='d3', activation='relu'))
    model.add(Dense(1, name='out'))
    
    model.summary()
    
    return model

def my_training(X_train, y_train, filename):
    model = my_model()
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss='mse')

    history = model.fit(X_train, y_train, batch_size=32, nb_epoch=5, verbose=2, validation_split=0.05)
    y_pred = model.predict(X_train)
    model.save(filename)

    plt.plot(y_train, y_pred, 'ro')
    plt.show()
    print(np.min(y_pred, axis=0), np.max(y_pred, axis=0))

def generator(features, labels, batch_size=32):
    num_samples = len(features)
    while 1: # Loop forever so the generator never terminates
        features, labels = shuffle(features, labels)
        for offset in range(0, num_samples, batch_size):
            batch_features = features[offset:offset+batch_size]
            batch_labels = labels[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def my_generator_training(X_train, y_train):
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    model = my_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
        validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 3)



my_training(X_train, y_train, "model.h5")
