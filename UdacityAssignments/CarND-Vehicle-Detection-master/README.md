# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


software pipeline to identify vehicles in a video from a front-facing camera on a car.

Videos and images used in this project are all from Udacity.

The raw data can be found from https://github.com/udacity/CarND-Vehicle-Detection

Steps for vehicle detection are as follow:

    1. Feature extraction and data split. Following are the steps:
       a. Car and non-car image data is downloaded from GTI vehicle image database and KITTI vision benchmark suite
       b. Extract features and combine them
       c. Split data in training and test
    2. Build classifier to predict images based on features, using LinearSVM. 
    3. Build pipeline to detect cars. Pipeline has following steps:
       a. Find all rectangles which has cars in image
       b. Draw heatmap with identified boxes
       c. Remove false positives by applying thresolds
       d. Label the heatmap image to identify number of cars in frame
       e. Draw identified cars areas on original image
    4. Run pipeline on video
       Run the above pipleine for every frame of video.

You can see all details in vehicle_detection.ipynb
Final output of test video is submitted as P5.mp4

Images after each step are plotted inline.