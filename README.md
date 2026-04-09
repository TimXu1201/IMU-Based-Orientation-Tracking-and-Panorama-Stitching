# IMU-Based Orientation Tracking and Panorama Stitching

This repository contains my work on **IMU-based orientation estimation** and **panorama generation from synchronized camera frames**.

The project combines sensor calibration, quaternion-based motion propagation, optimization-based refinement, and image reprojection into a single pipeline for inertial perception and visual scene reconstruction.

## Project Highlights

- IMU bias calibration for accelerometer and gyroscope measurements
- quaternion-based orientation propagation from angular velocity
- projected gradient descent refinement using accelerometer observations
- Euler-angle visualization and comparison against Vicon ground truth
- panorama stitching from camera frames using the estimated orientation trajectory

## Repository Structure

- `code/project_1.py`
  Main implementation for orientation tracking and panorama generation.
- `docs/`
  Supporting sensor-reference material.
- `orientation_*.png`
  Selected orientation-tracking result figures.
- `panorama_*.png`
  Selected panorama outputs and comparisons.
- `276A_project1.pdf`
  Project report.

## Environment

Typical dependencies:

- Python 3.10+
- numpy
- matplotlib
- torch

## Notes

- The original script uses a local Windows path for the course dataset. Update the `data_root` path in `code/project_1.py` before rerunning.
- Raw IMU, camera, and Vicon data are not included in the public repository.
- Selected output figures are kept to make the repository easier to browse as a portfolio project.
