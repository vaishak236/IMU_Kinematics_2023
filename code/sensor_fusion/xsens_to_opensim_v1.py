# name: xsens_to_opensim.py
# description: convert quaternion data from Xsens to OpenSim (.sto) format
# author: Vu Phan, Alex Kyu
# date: 2023/02/20, last edited 2023/03/15


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from ahrs.filters import Mahony, Madgwick, EKF


# Select filter
filter_type   = 'EKF'

# Constants
path = 'IMUData\\'
#trial_id      = 'Lunge_right-000'
trial_id      = 'MT_01200E04_000-000'
imu_placement = {'torso_imu': '_00B4D7D4', 'pelvis_imu': '_00B4D7D3', 
				'calcn_r_imu': '_00B4D7FE', 'tibia_r_imu': '_00B4D7FB', 'femur_r_imu': '_00B4D6D1',
				'calcn_l_imu': '_00B4D7FF', 'tibia_l_imu': '_00B4D7CE', 'femur_l_imu': '_00B4D7FD'}
imu_dt		  = {'torso_imu': None, 'pelvis_imu': None, 
				'calcn_r_imu': None, 'tibia_r_imu': None, 'femur_r_imu': None,
				'calcn_l_imu': None, 'tibia_l_imu': None, 'femur_l_imu': None}
      

f = 40.0 # Hz
dt = 1/f
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
    
    
    # Smoothing window here
#     imu_dt[location] = imu_dt[location].rolling(5, win_type='triang').mean()
#     imu_dt[location].dropna(inplace=True)

    
    if num_sample > imu_dt[location].shape[0]:
        num_sample = imu_dt[location].shape[0]

format_dt = format_dt + '\n'
xgyro_df = []

# Get orientation (in terms of quaternion)
init_sample = 840
num_sample = 6000 # overwrite to take only part of data
        
if filter_type != 'Xsens':
    imu_ahrs = {'torso_imu': None, 'pelvis_imu': None, 
                'hand_r_imu': None, 'radius_r_imu': None, 'humerus_r_imu': None,
                'calcn_r_imu': None, 'tibia_r_imu': None, 'femur_r_imu': None,
                'hand_l_imu': None, 'radius_l_imu': None, 'humerus_l_imu': None,
                'calcn_l_imu': None, 'tibia_l_imu': None, 'femur_l_imu': None}
    
    for location in imu_placement.keys():
        gyr_data = imu_dt[location][['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()[init_sample:, :]
        acc_data = imu_dt[location][['Acc_X','Acc_Y','Acc_Z']].to_numpy()[init_sample:, :]
        mag_data = imu_dt[location][['Mag_X','Mag_Y','Mag_Z']].to_numpy()[init_sample:, :]
        
        if filter_type == 'Mahony':
            imu_ahrs[location] = Mahony(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0)
        elif filter_type == 'Madgwick':
            imu_ahrs[location] = Madgwick(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0)
        elif filter_type == 'EKF':
            imu_ahrs[location] = EKF(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, frame='ENU')
        

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
with open('test.sto', 'w') as f:
    f.write(format_dt)
