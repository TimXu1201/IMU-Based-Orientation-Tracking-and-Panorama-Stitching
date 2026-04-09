# IMU-Based Orientation Tracking and Panorama Stitching

This repository presents an end-to-end workflow for **IMU-based orientation estimation** and **panorama generation from synchronized camera frames**.

The implementation combines sensor calibration, quaternion-based motion propagation, optimization-based refinement, and image reprojection into a single visual-inertial pipeline.

## Project Highlights

- IMU bias calibration for accelerometer and gyroscope measurements
- quaternion-based orientation propagation from angular velocity
- projected gradient descent refinement using accelerometer observations
- Euler-angle visualization and comparison against reference motion
- panorama stitching from camera frames using the estimated orientation trajectory

## Repository Structure

- `code/project_1.py`
  Main implementation for orientation tracking and panorama generation.
- `docs/`
  Supporting technical references for the sensor setup.
- `orientation_*.png`
  Selected orientation-tracking result figures.
- `panorama_*.png`
  Selected panorama outputs and comparisons.
- `report.pdf`
  Project Description.

## Environment

Typical dependencies:

- Python 3.10+
- numpy
- matplotlib
- torch

## Notes

- `code/project_1.py` still uses a local Windows dataset path and may need a quick path update before rerunning.
- Raw IMU, camera, and reference-motion datasets are not included in the public repository.
- Representative output figures are kept to make the repository easier to browse.
