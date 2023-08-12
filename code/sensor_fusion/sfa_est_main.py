# name: sfa_est_main.py
# description: Using sensor fusion methods to estimate joint angles
# author: Vu Phan
# date: 2023/03/20

import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 

from ahrs.filters import Complementary, Mahony, Madgwick, EKF
from tqdm import tqdm


# Quaternion handling
# Get conjugate of a quaternion
# (Referencs: https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/functions/index.htm)
def quaternion_conjugate(q):

	q_conj = np.zeros((q.shape))
	# print(q_conj) # FOR DEBUGGING

	q_conj[0] = q[0]
	q_conj[1] = -q[1]
	q_conj[2] = -q[2]
	q_conj[3] = -q[3]
	# print(q_conj) # FOR DEBUGGING

	return q_conj

# Multiply two quaternions
# (References: https://www.mathworks.com/help/aerotbx/ug/quatmultiply.html)
def quaternion_multiply(q, r):

	n = np.zeros((q.shape))

	n[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
	n[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
	n[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
	n[3] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]

	return n

def get_angle_from_quaternion(q):
	# (References: Fan B. et al, IEEE Sensors Journal, 2022)
	angle_x = math.atan2(-2*q[2]*q[3] + 2*q[0]*q[1] , q[3]**2 - q[2]**2 - q[1]**2 + q[0]**2)
	angle_x = np.rad2deg(angle_x)

	angle_y = math.asin(2*q[1]*q[3] + 2*q[0]*q[2])
	angle_y = np.rad2deg(angle_y)

	angle_z = math.atan2(-2*q[1]*q[2] + 2*q[0]*q[3], q[1]**2 + q[0]**2 - q[3]**2 - q[2]**2)
	angle_z = np.rad2deg(angle_z)

	# angle_x = math.atan2(2*q[2]*q[3] + 2*q[0]*q[1] , q[3]**2 - q[2]**2 - q[1]**2 + q[0]**2)
	# angle_x = np.rad2deg(angle_x)

	# angle_y = math.asin(2*q[1]*q[3] - 2*q[0]*q[2])
	# angle_y = np.rad2deg(angle_y)

	# angle_z = math.atan2(2*q[1]*q[2] + 2*q[0]*q[3], q[1]**2 + q[0]**2 - q[3]**2 - q[2]**2)
	# angle_z = np.rad2deg(angle_z)

	return angle_x, angle_y, angle_z

# Select filter
filter_type   = 'Mahony'

# Constants
path = 'test_imu_data\\'
#trial_id      = 'Lunge_right-000'
trial_id      = 'MT_01200E04_001-000'
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

print(num_sample)

# Get orientation (in terms of quaternion)
init_sample = 840
num_sample = 25500 # overwrite to take only part of data
        
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
        elif filter_type == 'Complementary':
        	imu_ahrs[location] = Complementary(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0)
        

for i in tqdm(range(init_sample,num_sample)):
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

# CALIBRATION
# Find transformation to bone 
q_st_f = quaternion_multiply(quaternion_conjugate(imu_ahrs['femur_l_imu'].Q[0]), imu_ahrs['pelvis_imu'].Q[0])
q_ss_t = quaternion_multiply(quaternion_conjugate(imu_ahrs['tibia_l_imu'].Q[0]), imu_ahrs['pelvis_imu'].Q[0])


knee_angle = {'x': np.zeros(num_sample), 'y': np.zeros(num_sample), 'z': np.zeros(num_sample)}
for i in tqdm(range(init_sample,num_sample)):
	q_f = quaternion_multiply(imu_ahrs['femur_l_imu'].Q[i], q_st_f)
	q_t = quaternion_multiply(imu_ahrs['tibia_l_imu'].Q[i], q_ss_t)
	q_f_t = quaternion_multiply(quaternion_conjugate(q_f), q_t)

	# q_f_t = quaternion_multiply(quaternion_conjugate(imu_ahrs['femur_l_imu'].Q[i]), imu_ahrs['tibia_l_imu'].Q[i]) # no calib

	angle_x, angle_y, angle_z = get_angle_from_quaternion(q_f_t)
	knee_angle['x'][i] = angle_x
	knee_angle['y'][i] = angle_y
	knee_angle['z'][i] = angle_z

# plt.plot(knee_angle['x'], label = 'x')
# plt.plot(knee_angle['y'], label = 'y')
# plt.plot(knee_angle['z'], label = 'z')
# plt.legend()
# plt.show()

with open('mahony_left_knee.npy', 'wb') as f:
	np.save(f, knee_angle['y'])

