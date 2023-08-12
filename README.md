# Lower-body Kinematics Estimation Using Inertial Measurement Units
## Team 1

## Overview
All codes implemented for this project can be found in the folder ```code``` while results can be found in the folder ```results```.

All data can downloaded from the folders ```data_imu``` (containing IMU data) and ```data_mocap``` (containing marker-based motion capture data).

## Implementation
All codes were implemented in Python 3.8 version. Below is the list of libraries needed to succesfully run the code:
- ```numpy```
- ```math```
- ```pandas```
- ```matplotlib```
- ```opensim``` (see https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting to learn how to install the ```opensim``` library for constrained IMU inverse kinematics)
- ```plotly```
- ```AHRS``` (see https://ahrs.readthedocs.io/en/latest/index.html to learn how to install the ```AHRS``` library for using sensor fusion algorithms)

## Guideline
1. Use ```imu_unc_ik.py``` to obtain joint angles without using OpenSense (i.e., unconstrained IMU).
2. Use ```opensense_ik.py``` to obtain joint angles using OpenSense (i.e., constrained IMU).
3. Use ```xsens_to_opensim_all.py``` to convert orientaion estimated from sensor fusion algorithms (including Xsens filter) to the OpenSim format (i.e., .sto files).
3. Use ```comparison_sync.py``` to synchronize motion capture and IMU data.
4. Use ```comparison_unconstrained.py``` to compare unconstrained IMU-based kinematics (without OpenSense).
5. Use ```compare_opensim_opensense.py``` to compare constrained IMU-based kinematics (with OpenSense).

Unconstrained and constrained IMU inverse kinematics results of (1) and (2) will be stored in the folder ```results```. (2) uses the .sto files generated by (3) and you can customize which .sto file is used for calibration and which is used for the actual calculations in (2). When running (4), (5), or (6), results from the folder ```results``` will be used to perform comparison between different methods.
