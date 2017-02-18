#This file reads captured images and write it to driving_data.p file

import csv 
import sys
import pickle
import cv2
import numpy as np
import random

import matplotlib.pyplot as plt

csv_path = "data/driving_log.csv"
#csv_path = "/Users/khaleel.pasha/Desktop/driving_log.csv"
center_images, left_images, right_images, steering_angles = [],[],[],[]
car_images, steering_angles_corrected = [], []

channels, row, col = 3, 160, 320  # camera format
#custom rows and columns
custom_rows, custom_cols = 66, 200

with open(csv_path) as csv_file:
    reader = csv.DictReader(csv_file)
    for current_row in reader:
        # read in images and steering angle
        steering = float(current_row['steering'].strip())
        image_center = current_row['center'].strip()
        image_left = current_row['left'].strip()
        image_right = current_row['right'].strip()

        # add images and angles to data set
        center_images.append(image_center)
        left_images.append(image_left)
        right_images.append(image_right)
        steering_angles.append(steering)

        #TODO: remove later
        if steering != 0.0:
            center_images.append(image_center)
            left_images.append(image_left)
            right_images.append(image_right)
            steering_angles.append(steering)
            center_images.append(image_center)
            left_images.append(image_left)
            right_images.append(image_right)
            steering_angles.append(steering)


def balance_data(center_images, left_images, right_images, steering_angles):
    zero_nonzero_ratio = 0.7
    data_length = len(center_images)
    zero_angles_count = []
    for index, label in enumerate(steering_angles):
        if label == 0.0:
            zero_angles_count.append(index)
    non_zero_angles_count = data_length-len(zero_angles_count)
    print("data_length :", data_length)
    print("zero angles :", len(zero_angles_count))
    print("non_zero_angles :", non_zero_angles_count)

    new_zero_angles_count = int(non_zero_angles_count * zero_nonzero_ratio)
    angles_to_remove = int(data_length - new_zero_angles_count - non_zero_angles_count)

    # Randomly remove excess zero angle images
    remove_indices = np.random.choice(zero_angles_count, angles_to_remove, replace=False)

    print("angles to remove:", angles_to_remove)

    center_images = np.delete(center_images, remove_indices, axis=0)
    left_images = np.delete(left_images, remove_indices, axis=0)
    right_images = np.delete(right_images, remove_indices, axis=0)
    steering_angles = np.delete(steering_angles,remove_indices, axis=0)

    print("data size left:", len(center_images))
    print("data size center:", len(center_images))
    print("data size left:", len(left_images))
    print("data size right:", len(right_images))
    print("data size steering angles:", len(steering_angles))

    for index, label in enumerate(steering_angles):
        correction = 0.2  # this is a parameter to tune
        steering_center = steering_angles[index]
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        #Append car images
        car_images.append(center_images[index])
        car_images.append(left_images[index])
        car_images.append(right_images[index])

        #Append car steering angles
        steering_angles_corrected.append(steering_center)
        steering_angles_corrected.append(steering_left)
        steering_angles_corrected.append(steering_right)

    driving_data = dict()
    driving_data['car_images'] = car_images
    driving_data['labels'] = steering_angles_corrected
    print("total rows :", len(car_images))


    return driving_data


def balance_data1(center_images, left_images, right_images, steering_angles):
    zero_nonzero_ratio = 1.0
    data_length = len(center_images)
    zero_angles_count = []
    for index, label in enumerate(steering_angles):
        if label == 0.0:
            zero_angles_count.append(index)
    non_zero_angles_count = data_length-len(zero_angles_count)
    print("data_length :", data_length)
    print("zero angles :", len(zero_angles_count))
    print("non_zero_angles :", non_zero_angles_count)

    #new_zero_angles_count = int(non_zero_angles_count * zero_nonzero_ratio)
    #angles_to_remove = data_length - new_zero_angles_count - non_zero_angles_count
    angles_to_remove = len(zero_angles_count) * 0.2

    # Randomly remove excess zero angle images
    remove_indices = np.random.choice(zero_angles_count, angles_to_remove, replace=False)

    print("angles to remove:", angles_to_remove)

    center_images = np.delete(center_images, remove_indices, axis=0)
    left_images = np.delete(left_images, remove_indices, axis=0)
    right_images = np.delete(right_images, remove_indices, axis=0)
    steering_angles = np.delete(steering_angles,remove_indices, axis=0)

    print("data size center:", len(center_images))
    print("data size left:", len(left_images))
    print("data size right:", len(right_images))
    print("data size steering angles:", len(steering_angles))

    driving_data = dict()
    driving_data['center_images'] = center_images
    driving_data['left_images'] = left_images
    driving_data['right_images'] = right_images
    driving_data['labels'] = steering_angles
    return driving_data

def balance_data2(center_images, left_images, right_images, steering_angles):
    driving_data = dict()
    driving_data['center_images'] = center_images
    driving_data['left_images'] = left_images
    driving_data['right_images'] = right_images
    driving_data['labels'] = steering_angles
    print("total rows :", len(center_images))
    return driving_data

driving_data = balance_data1(center_images, left_images, right_images, steering_angles)

with open('data/driving_data.p', mode='wb') as f:
    pickle.dump(driving_data, f)



