import json
import os
import pickle
import numpy as np
import matplotlib.image as mpimg
import cv2
from sklearn.utils import shuffle

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l2


"""
UTILITY METHODS TO PRE-PROCESS IMAGES
"""
#Cut and re-size image
def img_pre_process(image):
    roi = image[60:140, :, :] #Cut top and bottom of image
    image = cv2.resize(roi, (64,64), interpolation=cv2.INTER_AREA) #reducing image size so that model runs faster
    return image

#Change image brigtness
#Credit: https://github.com/mohankarthik/CarND-BehavioralCloning-P3/blob/master/model.py
def img_change_brightness(img):
    # Convert the image to HSV
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute a random brightness value and apply to the image
    brightness = 0.25 + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness

    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

#translate image and compensate for the translation on the steering angle
def translate_image(image, steering, horz_range=30, vert_range=5):
    rows, cols, chs = image.shape
    tx = np.random.randint(-horz_range, horz_range+1)
    ty = np.random.randint(-vert_range, vert_range+1)
    steering = steering + tx * 0.004 # multiply by steering angle units per pixel
    tr_M = np.float32([[1,0,tx], [0,1,ty]])
    image = cv2.warpAffine(image, tr_M, (cols,rows), borderMode=1)
    return image, steering


FOLDER_PATH = 'data'
def read_image(image_path):
    full_image_path = os.path.join(FOLDER_PATH, image_path.strip())
    image = mpimg.imread(full_image_path)
    return img_pre_process(image)


#Below method will be used as batch generator by model. Below method also augments data by,
# flipping: to avoid bias to left\right turns
# brightness: augmentaion to generalize on second track. not mandatory for first track
# translate: to simulate the car being at the edges of the road and hill-slopes.
# translate is done both on normal and flipped images
def generate_steering_angle(data, batch_size=128):
    X = []
    Y = []
    while True:
        data = shuffle(data)
        for line in data:
            image = read_image(line['center'])
            angle = line['angle']
            #Original image
            image_brightened = img_change_brightness(image)
            X.append(image_brightened)
            Y.append(angle)

            #Flip original image
            flipped_image = cv2.flip(image, 1)
            flipped_image_brightened = img_change_brightness(flipped_image)
            X.append(flipped_image_brightened)
            Y.append(-angle)

            #Translate original image to simulate car at the edge
            translated_image, translated_angle = translate_image(image_brightened, angle)
            X.append(translated_image)
            Y.append(translated_angle)

            #Translate flipped image to simulate car at the edge
            translated_flipped_image, translated_flipped_angle = translate_image(flipped_image_brightened, angle)
            X.append(translated_flipped_image)
            Y.append(translated_flipped_angle)

            if len(X) >= batch_size:
                X, Y = shuffle(X, Y)
                yield np.array(X), np.array(Y)  # image, streeing angle
                X = []
                Y = []


#Generate validation data
def generate_validation(data):
    X = []
    Y = []
    while True:
        data = shuffle(data)
        for line in data:
            angle = line['angle']
            image = read_image(line['center'])

            X.append(image)
            Y.append(angle)
            yield np.array(X), np.array(Y)  # (image, steering angle)


#Using pre-trained VGG16 as base model and adding regression layer to predict steering angle
def create_model():
    # LOAD pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=[64, 64, 3])

    x = base_model.output
    x = Flatten()(x)

    # Regression layer
    x = Dense(1000, activation='relu', name='fc1', W_regularizer=l2(0.0001))(x)
    # x = Dropout(0.5)(x)
    x = Dense(250, activation='relu', name='fc2', W_regularizer=l2(0.0001))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)

    model = Model(input=base_model.input, output=predictions)
    # train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


#MAIN
if __name__ == '__main__':

    # STEP1: Create model
    model = create_model()

    # STEP2: Load training and validation data
    training_pickle = 'data/train.p'
    with open(training_pickle, 'rb') as handle:
        driving_info = pickle.load(handle)

    validation_pickle = 'data/validation.p'
    with open(validation_pickle, 'rb') as handle:
        validation_info = pickle.load(handle)

    print("train size", len(driving_info))
    print("validation size", len(validation_info))

    # STEP3: Train the model
    model.fit_generator(
        generate_steering_angle(driving_info, batch_size=32),
        samples_per_epoch=len(driving_info) * 4,
        nb_epoch=1,
        validation_data=generate_validation(validation_info),
        nb_val_samples=len(validation_info) / 7)

    # STEP4: Save the model
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

    print("model saved.")