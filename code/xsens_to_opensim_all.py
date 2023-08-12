# name: xsens_to_opensim.py
# description: convert quaternion data from Xsens to OpenSim (.sto) format
# author: Vu Phan, Alex Kyu
# date: 2023/02/20, last edited 2023/03/15


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 

from ahrs.filters import Mahony, Madgwick, EKF, Complementary


# Select filter
# filter_type   = 'Xsens'
# filters = ['Xsens', 'Mahony', 'Madgwick', 'Complementary', 'EKF']
filters = ['EKF']
f = 40.0 # Hz
dt = 1/f
calibration_time = {'s1_ronin': int(f*20), 's2_vaishak': int(f*50), 's3_vu': int(f*22), 's4_alex': int(f*22)}

# Constants
# path = 'data\\s1_ronin\\combine_10min\\'
# path = 'data\\s4_vaishak\\combine_10min\\'
# path = 'data\\s4_vu\\combine_10min\\'
# path = 'data\\s4_alex\\combine_10min\\'

paths = ['data\\s1_ronin\\combine_10min\\', 'data\\s2_vaishak\\combine_10min\\', 'data\\s3_vu\\combine_10min\\', 'data\\s4_alex\\combine_10min\\']
# paths = ['data\\s3_vu\\combine_10min\\', 'data\\s4_alex\\combine_10min\\']

lthigh = ['_00B4D7FD', '_00B4D7D5', '_00B4D7CF']
rthigh = ['_00B4D6D1', '_00B4D7D1', '_00B4D7D6']
# thigh_combinations = [[0, 0], [0, 1], [0, 2],
#                       [1, 0], [1, 1], [1, 2],
#                       [2, 0], [2, 1], [2, 2]]
# thigh_combinations = [[0, 0], [0, 1], [1, 1], [2, 2]]
thigh_combinations = [[0, 0]]

# thigh_combinations = [[1, 1], [2, 2]]


for thigh_combo in thigh_combinations:
    for filter_type in filters:
        for path in paths:
            print("Processing " + str(path) + ", thigh_combo " + str(thigh_combo) + ", filter " + str(filter_type))
            subject, task_type = path.split("\\")[1:3]
            trial_id      = 'MT_01200E04_001-000'
            imu_placement = {'torso_imu': '_00B4D7D4', 'pelvis_imu': '_00B4D7D3', 
                            'calcn_r_imu': '_00B4D7FE', 'tibia_r_imu': '_00B4D7FB', 'femur_r_imu': rthigh[thigh_combo[0]],
                            'calcn_l_imu': '_00B4D7FF', 'tibia_l_imu': '_00B4D7CE', 'femur_l_imu': lthigh[thigh_combo[1]]}
            imu_dt = {'torso_imu': None, 'pelvis_imu': None, 
                            'calcn_r_imu': None, 'tibia_r_imu': None, 'femur_r_imu': None,
                            'calcn_l_imu': None, 'tibia_l_imu': None, 'femur_l_imu': None}
                
            num_imu = len(imu_placement)

            # Create fake headers
            time = 0
            format_dt = 'DataRate=40.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.4-2022-07-23-0e9fedc\nendheader\n'
            format_dt = format_dt + 'time'

            # Extract data
            num_sample = 999999
            for location in imu_placement.keys():
                format_dt = format_dt + '\t' 
                format_dt = format_dt + location


                fn = path + trial_id + imu_placement[location] + '.txt'
                with open(fn, 'r') as f:
                    txt = f.readlines()
                    header = txt[4].split('\t')

                temp_dt = np.genfromtxt(fn, delimiter='\t', skip_header=5)
                temp_dt = temp_dt[:, 2::]
                header = header[2::]
                header[-1] = header[-1][0:-1]
                imu_dt[location] = pd.DataFrame(temp_dt, columns = header)

                
                if num_sample > imu_dt[location].shape[0]:
                    num_sample = imu_dt[location].shape[0]

            format_dt = format_dt + '\n'

            # Get orientation (in terms of quaternion)
            init_sample = calibration_time[subject]
            # num_sample = 3000

            if filter_type != 'Xsens':
                imu_ahrs = {'torso_imu': None, 'pelvis_imu': None, 
                            'calcn_r_imu': None, 'tibia_r_imu': None, 'femur_r_imu': None,
                            'calcn_l_imu': None, 'tibia_l_imu': None, 'femur_l_imu': None}
                
                for location in imu_placement.keys():
                    gyr_data = imu_dt[location][['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()[init_sample:num_sample, :]
                    acc_data = imu_dt[location][['Acc_X','Acc_Y','Acc_Z']].to_numpy()[init_sample:num_sample, :]
                    mag_data = imu_dt[location][['Mag_X','Mag_Y','Mag_Z']].to_numpy()[init_sample:num_sample, :]
                    
                    # if filter_type == 'Mahony':
                    #     imu_ahrs[location] = Mahony(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0)
                    # elif filter_type == 'Madgwick':
                    #     imu_ahrs[location] = Madgwick(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0)
                    # elif filter_type == 'EKF':
                    #     imu_ahrs[location] = EKF(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, frame='ENU')
                    # elif filter_type == 'Complementary':
                    #     imu_ahrs[location] = Complementary(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0)

                    if filter_type == 'Mahony':
                        imu_ahrs[location] = Mahony(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, k_P = 0.1, k_I = 3)
                    elif filter_type == 'Madgwick':
                        imu_ahrs[location] = Madgwick(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, gain = 0.02)
                    elif filter_type == 'Complementary':
                        imu_ahrs[location] = Complementary(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, gain = 0.04)
                    elif filter_type == 'EKF':
                        imu_ahrs[location] = EKF(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, frame='ENU', noises = np.array([5**2, 1**2, 4**2]))

            for i in range(init_sample,num_sample):
                format_dt = format_dt + str(i*dt)

                for location in imu_placement.keys():
                    format_dt = format_dt + '\t'
                    
                    if filter_type == 'Xsens': # get quaternion directly from Xsens data
                        format_dt = format_dt + str(imu_dt[location]['Quat_q0'][i]) + ',' + \
                                                str(imu_dt[location]['Quat_q1'][i]) + ',' + \
                                                str(imu_dt[location]['Quat_q2'][i]) + ',' + \
                                                str(imu_dt[location]['Quat_q3'][i])
                    else: # compute quaternion from open-source sensor fusion algorithms
                        format_dt = format_dt + str(imu_ahrs[location].Q[i-init_sample, 0]) + ',' + \
                                                str(imu_ahrs[location].Q[i-init_sample, 1]) + ',' + \
                                                str(imu_ahrs[location].Q[i-init_sample, 2]) + ',' + \
                                                str(imu_ahrs[location].Q[i-init_sample, 3])

                format_dt = format_dt + '\n'

            # print(format_dt)

            # Export .sto file (i.e., OpenSim format)
            if not os.path.exists('results\\aim2and3\\constrained\\sto\\' + 'r' + str(thigh_combo[0] + 1) + 'l' + str(thigh_combo[1] + 1) + '\\' + subject):
                os.makedirs('results\\aim2and3\\constrained\\sto\\' + 'r' + str(thigh_combo[0] + 1) + 'l' + str(thigh_combo[1] + 1) + '\\' + subject)
            with open('results\\aim2and3\\constrained\\sto\\' + 'r' + str(thigh_combo[0] + 1) + 'l' + str(thigh_combo[1] + 1) + '\\' + subject + '\\' + filter_type + '_' + subject + '_' + task_type + '.sto', 'w') as file:
                file.write(format_dt)
