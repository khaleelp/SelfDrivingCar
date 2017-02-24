import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import random
import cv2

from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height - 15), (0, height / 2 - 10),
                          (width, height / 2 - 10), (width, height - 15)]],
                        dtype=np.int32)
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    # filling pixels inside the polygon defined by \"vertices\" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero\n",
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def rezise(img):
    return cv2.resize(img, (75, 48))

def preprocess_image(img):
    img = region_of_interest(img)
    img = rezise(img)
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        old_steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        old_throttle = float(data["throttle"]) or 1.2
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image_array = preprocess_image(image_array)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        #throttle = 0.6 if speed < 21 else 0.2
        #throttle = 1.8/(1+2*(abs(old_steering_angle - steering_angle)))
        #throttle = 0.2/(1+(abs(old_steering_angle - steering_angle)/50))
        throttle = 0.2
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)