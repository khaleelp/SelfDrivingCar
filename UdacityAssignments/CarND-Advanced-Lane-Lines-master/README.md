## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  


The Project
---

Below is the list of techniques used to detect lanes.


    1. Camera calibration.
    2. Distortion correction
    3. Edge detection by applying sobel, magnitude and direction thresolds
    4. Pipeline returns black and white images with edges detected.
    5. Perpective Transform: Warp image
    6. Detect lines . Detect left, right lanes . Do sanity on images and use pervious values if check fails
    Draw polygon

You can see all details in [![advanced_lane_lines_detection.ipynb] ()] file

Images after each step are plotted.

