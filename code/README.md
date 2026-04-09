# ECE 276A Project 1: Orientation Tracking

## Files Included
* `project_1.py`: The main source code for orientation tracking and panorama stitching.
* `README.md`: This file.

## Prerequisites
The code requires Python 3 and the following libraries:
* numpy
* matplotlib
* torch (PyTorch)

## Path
Make sure change the "data_root" in `project_1.py` according to your own pc.

The code assumes the data is stored in a data folder in the same directory as the script. Please ensure your directory looks like this:

/Project_1_Folder
    |-- project_1.py
    |-- data/
        |-- trainset/
            |-- imu/
            |-- cam/
            |-- vicon/
        |-- testset/
            |-- imu/
            |-- cam/