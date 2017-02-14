#This file reads captured images and write it to camera_data.pickle file

import csv 
import sys
import pickle
import cv2
import numpy as np
import random

import matplotlib.pyplot as plt

#csv_path = "data/driving_log.csv"
csv_path = "/Users/khaleel.pasha/Desktop/driving_log.csv"
left_images, center_images, right_images, steering_angles = [],[],[],[]

channels, row, col = 3, 160, 320  # camera format
#custom rows and columns
custom_rows, custom_cols = 66, 200

data = []
with open(csv_path) as csv_file:
    reader = csv.DictReader(csv_file)
    for current_row in reader:
        center_images.append(current_row['center'].strip())
        left_images.append(current_row['left'].strip())
        right_images.append(current_row['right'].strip())
        steering_angles.append(float(current_row['steering'].strip()))


def balance_data(center_images, left_images, right_images, steering_angles):
    data_length = len(center_images)
    zero_angles_count = []
    for index, label in enumerate(steering_angles):
        if label == 0.0:
            zero_angles_count.append(index)
    non_zero_angles_count = data_length-len(zero_angles_count)
    print("data_length :", data_length)
    print("zero angles :", len(zero_angles_count))
    print("non_zero_angles :", non_zero_angles_count)

    new_zero_angles_count = non_zero_angles_count
    angles_to_remove = data_length - new_zero_angles_count - non_zero_angles_count

    # Randomly remove excess zero angle images
    remove_indices = np.random.choice(zero_angles_count, angles_to_remove, replace=False)

    print("angles to remove:", angles_to_remove)

    #TODO: issue with removing data from center images. fix this later
    driving_data = dict()
    driving_data['center_images'] = np.delete(center_images, remove_indices, axis=0)
    driving_data['left_images'] = np.delete(left_images, remove_indices, axis=0)
    driving_data['right_images'] = np.delete(right_images, remove_indices, axis=0)
    driving_data['labels'] = np.delete(steering_angles,remove_indices, axis=0)

    print("data size left:", len(driving_data))
    print("data size center:", len(driving_data['center_images']))
    print("data size left:", len(driving_data['left_images']))
    print("data size right:", len(driving_data['right_images']))
    print("data size steering angles:", len(driving_data['labels']))

    return driving_data

def balance_data1(center_images, left_images, right_images, steering_angles):
    driving_data = dict()
    driving_data['center_images'] = center_images
    driving_data['left_images'] = left_images
    driving_data['right_images'] = right_images
    driving_data['labels'] = steering_angles
    print("total rows :", len(center_images))
    return driving_data


driving_data = balance_data(center_images, left_images, right_images, steering_angles)

with open('data/driving_data.p', mode='wb') as f:
    pickle.dump(driving_data, f)



