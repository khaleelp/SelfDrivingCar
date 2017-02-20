#This file reads captured images and write it to driving_data.p file

import csv 
import sys
import pickle
import cv2
import numpy as np
import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

csv_path = "data/driving_log.csv"
#csv_path = "/Users/khaleel.pasha/Desktop/driving_log.csv"
center_images, left_images, right_images, steering_angles = [],[],[],[]
car_images, steering_angles_corrected = [], []
training_images, training_angles, validation_images, validation_angles = [], [], [], []

channels, row, col = 3, 160, 320  # camera format
#custom rows and columns
custom_rows, custom_cols = 66, 200
angle_correction = 0.20

#https://medium.com/@acflippo/cloning-driving-behavior-by-augmenting-steering-angles-5faf7ea8a125#.2rfyq8aah
def perturb_angle(angle):
    new_angle = angle* (1 + np.random.uniform(-1, 1)/30.0)
    return min(1, new_angle) if new_angle > 0 else max(-1, new_angle)

with open(csv_path) as csv_file:
    reader = csv.DictReader(csv_file)
    for current_row in reader:
        # read in images and steering angle
        steering = float(current_row['steering'].strip())
        image_center = current_row['center'].strip()
        image_left = current_row['left'].strip()
        image_right = current_row['right'].strip()

        if abs(steering) > 0.20:
            angle_correction = 0.15
        elif abs(steering) > 0.40:
            angle_correction = 0.15
        elif abs(steering) > 0.70:
            angle_correction = 0.15

        if abs(steering) != 0.0:
            # add images and angles to data set
            car_images.append(image_center)
            steering_angles.append(steering)
            # angle correction
            car_images.append(image_left)
            steering_angles.append(steering + angle_correction)
            # angle correction
            car_images.append(image_right)
            steering_angles.append(steering - angle_correction)
        else:
            prob = np.random.uniform()
            if prob <= 0.05:
                # add images and angles to data set
                car_images.append(image_center)
                steering_angles.append(steering)
                # angle correction
                car_images.append(image_left)
                steering_angles.append(steering + angle_correction)
                # angle correction
                car_images.append(image_right)
                steering_angles.append(steering - angle_correction)


    print("Data Size:", len(car_images))
    print("Max angle:", max(steering_angles))
    print("Min angle:", min(steering_angles))

    #Add more data
    steering_angles_temp = steering_angles[:]
    for index, steering_angle in enumerate(steering_angles_temp):
        if steering_angle>0.7:
            for i in range(0, 10):
                new_angle = perturb_angle(steering_angle)
                car_images.append(car_images[index])
                steering_angles.append(new_angle)
                # print("Old angle - new angle:", steering_angle, new_angle)
                # print("len steering temp:", len(steering_angles_temp))


    print("Data Size:", len(car_images))
    print("Max angle:", max(steering_angles))
    print("Min angle:", min(steering_angles))

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
        correction = 0.25  # this is a parameter to tune
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
    angles_to_remove = int(len(zero_angles_count) * 0.9)

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
    #print("total rows :", len(center_images))
    return driving_data

def sample_data_from_bin(current_bin, images_in_bin, angles_in_bin):
    sample_per_angle = 4

    print("Current bin:", current_bin)
    print("Total_images:", len(images_in_bin))
    #print("Angles in bin:", len(angles_in_bin))
    #print("bin min:", min(angles_in_bin))
    #print("bin max:", max(angles_in_bin))

    if len(images_in_bin) > sample_per_angle:
        images_in_bin, angles_in_bin = shuffle(images_in_bin, angles_in_bin)
        training_images.extend(images_in_bin[0:sample_per_angle])
        training_angles.extend(angles_in_bin[0:sample_per_angle])
        validation_images.extend(images_in_bin[sample_per_angle:2*sample_per_angle])
        validation_angles.extend(angles_in_bin[sample_per_angle:2*sample_per_angle])

def balance_data3(car_images, steering_angles):
    driving_data = dict()
    training_images, training_angles = [], []
    validation_images, validation_angles = [], []

    modified_data = []
    sample_bin_start = 0.0
    sample_bin_end = 1.0
    sample_bin_step = 0.01
    current_bin = sample_bin_start

    angles_in_bin = []
    images_in_bin = []
    #Iterate through all angles from 0.0 to 1.0
    while current_bin < sample_bin_end:
        #Go through all steering angles and find angles only from current bin
        for index, steering_angle in enumerate(steering_angles):
            #If angle fallls in current bin, add it to angles in bin
            if (current_bin < abs(steering_angle)) and (abs(steering_angle)<=current_bin+sample_bin_step) :
                angles_in_bin.append(steering_angles[index])
                images_in_bin.append(car_images[index])
        #sample the bin
        sample_data_from_bin(current_bin, images_in_bin, angles_in_bin)
        #clear angles in current bin
        angles_in_bin.clear()
        images_in_bin.clear()
        #move on to next bin
        current_bin = current_bin+sample_bin_step

'''
#balance data
#balance_data3(car_images, steering_angles)

driving_data = dict()
driving_data["training_images"] = training_images
driving_data["training_angles"] = training_angles
driving_data["validation_images"] = validation_images
driving_data["validation_angles"] = validation_angles
'''

driving_data = dict()
driving_data["training_images"] = car_images
driving_data["training_angles"] = steering_angles
driving_data["validation_images"] = car_images[0:50]
driving_data["validation_angles"] = steering_angles[0:50]

print("training images count:", len(driving_data["training_images"]))
print("training angles count:", len(driving_data["training_angles"]))
print("validation images count:", len(validation_images))
print("validation angles count:", len(validation_angles))


with open('data/driving_data.p', mode='wb') as f:
    pickle.dump(driving_data, f)



