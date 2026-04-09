# Orientation Tracking Code

This folder contains the implementation used for IMU-based orientation tracking and panorama stitching.

## Main File

- `project_1.py`

## Dependencies

- numpy
- matplotlib
- torch

## Data Layout

Update the `data_root` variable in `project_1.py` to match your local dataset location.

Expected layout:

```text
Project_1_Folder/
|-- project_1.py
`-- data/
    |-- trainset/
    |   |-- imu/
    |   |-- cam/
    |   `-- vicon/
    `-- testset/
        |-- imu/
        `-- cam/
```
