import plotly.express as px
import numpy as np
import pandas as pd

import os
from os import walk

import opensim as osim
from math import pi

# files = []
headerPath = 'results\\aim2and3\\constrained\\sto\\'
cal_thigh = 'r1l1'
run_thigh = 'r1l1'
calPaths = [headerPath + cal_thigh + '\\s1_ronin', headerPath + cal_thigh + '\\s2_vaishak', headerPath + cal_thigh + '\\s3_vu', headerPath + cal_thigh + '\\s4_alex']
runPaths = [headerPath + run_thigh + '\\s1_ronin', headerPath + run_thigh + '\\s2_vaishak', headerPath + run_thigh + '\\s3_vu', headerPath + run_thigh + '\\s4_alex']
# filter_types = ['Madgwick', 'Xsens', 'Mahony', 'Complementary', 'EKF']
filter_types = ['Xsens', 'Mahony']
# filter_types = ['Complementary', 'EKF']

task = 'combine_10min'
# paths = ['opensense_sto\\s3_vu', 'opensense_sto\\s4_alex']
savePath = 'aim2and3\\constrained\\IKResults'

modelFileName = 'Rajagopal_2015.osim';          # The path to an input model
sensor_to_opensim_rotations = osim.Vec3(-pi/2, 0, 0);# The rotation of IMU data to the OpenSim world frame
baseIMUName = 'torso_imu';                     # The base IMU is the IMU on the base body of the model that dictates the heading (forward) direction of the model.
baseIMUHeading = 'z';                           # The Coordinate Axis of the base IMU that points in the heading direction. 
visulizeCalibration = False;                     # Boolean to Visualize the Output model
visualizeTracking = False;  # Boolean to Visualize the tracking simulation
startTime = 0;          # Start time (in seconds) of the tracking simulation. 
endTime = 99999;              # End time (in seconds) of the tracking simulation.


# for path in paths:
#     for (dirpath, dirnames, filenames) in walk(os.curdir + '\\' + path):
#         files.extend([path + '\\' + filename for filename in filenames])
#         break

for filter in filter_types:
    for i in range(len(calPaths)):
        calThigh, subject = calPaths[i].split('\\')[-2:]
        runThigh = runPaths[i].split('\\')[-2]
        calibrateOrientationsFileName = calPaths[i] + '\\' + filter + '_' + subject + '_' + task + '.sto'
        runOrientationsFileName = runPaths[i] + '\\' + filter + '_' + subject + '_' + task + '.sto'
        print("Calibrating for " + calibrateOrientationsFileName)
        # Set variables to use (OpenSense_CalibrateModel.py)


        # Instantiate an IMUPlacer object
        imuPlacer = osim.IMUPlacer();

        # Set properties for the IMUPlacer
        imuPlacer.set_model_file(modelFileName);
        imuPlacer.set_orientation_file_for_calibration(calibrateOrientationsFileName);
        imuPlacer.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations);
        imuPlacer.set_base_imu_label(baseIMUName);
        imuPlacer.set_base_heading_axis(baseIMUHeading);

        # Run the IMUPlacer
        imuPlacer.run(visulizeCalibration);

        # Get the model with the calibrated IMU
        model = imuPlacer.getCalibratedModel();

        # Print the calibrated model to file.
        model.printToXML('calibrated_' + modelFileName)


        print("Tracking Orientation for " + runOrientationsFileName)

        # Orientation Tracking (OpenSense_OrientationTracking.py)
        calibratedModelFileName = 'calibrated_Rajagopal_2015.osim';  # The path to an input model

        resultsDirectory = savePath + '\\cal' + cal_thigh + 'run' + run_thigh + '\\' + subject;

        if not os.path.exists(savePath + '\\cal' + cal_thigh + 'run' + run_thigh):
            os.makedirs(savePath + '\\cal' + cal_thigh + 'run' + run_thigh)
        if not os.path.exists(resultsDirectory):
            os.makedirs(resultsDirectory)

        print('Saving to: ' + resultsDirectory)

        # Instantiate an InverseKinematicsTool
        imuIK = osim.IMUInverseKinematicsTool();
        
        # Set tool properties
        imuIK.set_model_file(calibratedModelFileName);
        imuIK.set_orientations_file(runOrientationsFileName);
        imuIK.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations)
        imuIK.set_results_directory(resultsDirectory)

        # Set time range in seconds
        imuIK.set_time_range(0, startTime); 
        imuIK.set_time_range(1, endTime);   

        # Run IK
        imuIK.run(visualizeTracking);