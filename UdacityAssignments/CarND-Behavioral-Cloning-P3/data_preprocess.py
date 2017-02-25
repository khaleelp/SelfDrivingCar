import pickle
import numpy as np
from sklearn.utils import shuffle

import copy

from pandas.io.parsers import read_csv


"""
create new data by duplicating existing images by perturbing the angles a bit.
"""
#Credit
#https://medium.com/@acflippo/cloning-driving-behavior-by-augmenting-steering-angles-5faf7ea8a125#.41vw9spjy
def perturb_angle(angle):
    new_angle = angle* (1 + np.random.uniform(-1, 1)/30.0)
    return min(1, new_angle) if new_angle > 0 else max(-1, new_angle)

# STEP1: LOAD and MODIFY data
drive_info = read_csv("data/driving_log.csv", header=0, usecols=[0, 1, 2, 3]).as_matrix();

modified_data = []
for row in drive_info:
    center = copy.deepcopy(row)
    center[1] = row[3]  # original_angle
    modified_data.append(center)

    left = copy.deepcopy(row)
    left[0] = left[1]  # left_image
    left[1] = left[3] + 0.25
    modified_data.append(left)

    right = copy.deepcopy(row)
    right[0] = right[2]  # right image
    right[1] = right[3] - 0.25
    modified_data.append(right)

# STEP2: Add more samples for steering angle more than 0.5 or less then -0.5
insufficient_data = [data for data in modified_data if abs(float(data[3])) > 0.5]
for line in insufficient_data:
    for i in range(0, 14):
        new_line = copy.deepcopy(line)
        new_line[1] = perturb_angle(line[3])
        modified_data.append(new_line)
driving_data = np.array(modified_data)
print('modified data', len(modified_data))

#Setup training and validation data
training_data = []
validation_data = []

step = 0.01
per_step = 19
current_bin = 0.0

"""
Utility method to extract data from bin
"""
def extract_data_from_bin(current_bin_data):
    validation_bin = []

    if len(current_bin_data) > per_step:
        current_bin_data = shuffle(current_bin_data)
        validation_bin = current_bin_data[per_step:2 * per_step]
        for line in validation_bin:
            validation_data.append(line)
        current_bin_data = current_bin_data[0:per_step]
    for line in current_bin_data:
        training_data.append(line)

# STEP3: Divide data on 100 bins and populate values to each bin
sample_bin_start = 0.0
sample_bin_end = 1.0
sample_bin_step = 0.01
current_bin_data = []

while current_bin < sample_bin_end:
    current_bin_data = [data for data in modified_data if
                        (current_bin < abs(data[1])) and (abs(data[1]) <= current_bin + step)]
    extract_data_from_bin(current_bin_data)
    print("bin:len", current_bin, " : ", len(current_bin_data))
    current_bin = current_bin + step

zero_bin = [data for data in modified_data if (data[1] == 0.0)]
extract_data_from_bin(zero_bin)

training_data = np.array(training_data)
validation_data = np.array(validation_data)
print("trained data", len(training_data))
print("validation data", len(validation_data))


#STEP4: Save training and validation data
pickle_data = []
validation_pickle = []

for row in training_data:
    pickle_data.append({'center': row[0], 'angle': row[1]})

for row in validation_data:
    validation_pickle.append({'center': row[0], 'angle': row[1]})

training_file = 'train.p'
with open("data/" + training_file, 'wb') as handle:
    pickle.dump(np.array(pickle_data), handle, protocol=pickle.HIGHEST_PROTOCOL)

validation_file = 'validation.p'
with open("data/" + validation_file, 'wb') as handle:
    pickle.dump(np.array(validation_pickle), handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Training and validation angles saved to pickle file")
