import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px
import quaternion
import math
import os

from ahrs.filters import Mahony, Madgwick, EKF, Complementary

# madgwick, mahony, Xsens, complementary, and EKF for all subjects for all joint angles
# 5 filters x 4 subjects x 6 joint angles (2 hip, 2 knee, 2 ankle) = 120 sets of data

# Calibration times:
# - s1_ronin = 20 seconds
# - s2_vaishak = 50 seconds
# - s3_vu = 22 seconds
# - s4_alex = 22 seconds

# Constants
trial_id      = 'MT_01200E04_001-000'
lthigh = ['_00B4D7FD', '_00B4D7D5', '_00B4D7CF']
rthigh = ['_00B4D6D1', '_00B4D7D1', '_00B4D7D6']

# thigh_combinations = [[0, 0, 0, 1], [1, 1, 1, 1], [2, 2, 2, 2]]
thigh_combinations = [[0, 0, 0, 0]]
# above indexes are [calibration right, calibration left, run right, run left]

f = 40.0 # Hz
dt = 1/f

reference_imu_location = 'pelvis_imu'
thigh_left_imu = 'femur_l_imu'
thigh_right_imu = 'femur_r_imu'
shank_left_imu = 'tibia_l_imu'
shank_right_imu = 'tibia_r_imu'
foot_left_imu = 'calcn_l_imu'
foot_right_imu = 'calcn_r_imu'

paths = ['data\\s1_ronin\\combine_10min\\', 'data\\s2_vaishak\\combine_10min\\',
        'data\\s3_vu\\combine_10min\\', 'data\\s4_alex\\combine_10min\\']
# paths = ['data\\s3_vu\\combine_10min\\']
# paths = ['data\\s4_alex\\combine_10min\\']
calibration_time = {'s1_ronin': int(f*20), 's2_vaishak': int(f*50), 's3_vu': int(f*22), 's4_alex': int(f*22)}
filters = ['Xsens', 'Mahony', 'Madgwick', 'Complementary', 'EKF']
# filters = ['Mahony', 'Complementary', 'EKF']


def get_angle_from_quaternion(q):
    # print(quaternion.as_float_array(q))
	# (References: Fan B. et al, IEEE Sensors Journal, 2022)
    angle_x = math.atan2(-2*q[2]*q[3] + 2*q[0]*q[1] , q[3]**2 - q[2]**2 - q[1]**2 + q[0]**2)
    angle_x = np.rad2deg(angle_x)

    angle_y = math.asin(2*q[1]*q[3] + 2*q[0]*q[2])
    angle_y = np.rad2deg(angle_y)

    angle_z = math.atan2(-2*q[1]*q[2] + 2*q[0]*q[3], q[1]**2 + q[0]**2 - q[3]**2 - q[2]**2)
    angle_z = np.rad2deg(angle_z)

    return angle_x, angle_y, angle_z

def get_quaternions(path, filter_type, init_sample, imu_placement, final_sample=None):

    imu_dt = {'torso_imu': None, 'pelvis_imu': None, 
                    'calcn_r_imu': None, 'tibia_r_imu': None, 'femur_r_imu': None,
                    'calcn_l_imu': None, 'tibia_l_imu': None, 'femur_l_imu': None}

    # Extract data
    num_sample = 999999
    for location in imu_placement.keys():
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
            # elif filter_type == 'Complementary':
            #     imu_ahrs[location] = Complementary(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0)
            # elif filter_type == 'EKF':
            #     imu_ahrs[location] = EKF(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, frame='ENU')

            if filter_type == 'Mahony':
                imu_ahrs[location] = Mahony(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, k_P = 0.1, k_I = 3)
            elif filter_type == 'Madgwick':
                imu_ahrs[location] = Madgwick(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, gain = 0.02)
            elif filter_type == 'Complementary':
                imu_ahrs[location] = Complementary(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, gain = 0.04)
            elif filter_type == 'EKF':
                imu_ahrs[location] = EKF(gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = 40.0, frame='ENU', noises = np.array([5**2, 1**2, 4**2]))


    if filter_type == 'Xsens':
        q_pelvis = quaternion.as_quat_array(imu_dt[reference_imu_location][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']][init_sample:num_sample].to_numpy())
        q_thigh_left = quaternion.as_quat_array(imu_dt[thigh_left_imu][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']][init_sample:num_sample].to_numpy())
        q_thigh_right = quaternion.as_quat_array(imu_dt[thigh_right_imu][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']][init_sample:num_sample].to_numpy())
        q_shank_left = quaternion.as_quat_array(imu_dt[shank_left_imu][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']][init_sample:num_sample].to_numpy())
        q_shank_right = quaternion.as_quat_array(imu_dt[shank_right_imu][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']][init_sample:num_sample].to_numpy())
        q_foot_left = quaternion.as_quat_array(imu_dt[foot_left_imu][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']][init_sample:num_sample].to_numpy())
        q_foot_right = quaternion.as_quat_array(imu_dt[foot_right_imu][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']][init_sample:num_sample].to_numpy())
    else:
        q_pelvis = quaternion.as_quat_array(imu_ahrs[reference_imu_location].Q)
        q_thigh_left = quaternion.as_quat_array(imu_ahrs[thigh_left_imu].Q)
        q_thigh_right = quaternion.as_quat_array(imu_ahrs[thigh_right_imu].Q)
        q_shank_left = quaternion.as_quat_array(imu_ahrs[shank_left_imu].Q)
        q_shank_right = quaternion.as_quat_array(imu_ahrs[shank_right_imu].Q)
        q_foot_left = quaternion.as_quat_array(imu_ahrs[foot_left_imu].Q)
        q_foot_right = quaternion.as_quat_array(imu_ahrs[foot_right_imu].Q)
    
    acc_x_pelvis = imu_dt[reference_imu_location][['Acc_X']][init_sample:num_sample].to_numpy()
    
    return q_pelvis, q_thigh_left, q_thigh_right, q_shank_left, q_shank_right, q_foot_left, q_foot_right, acc_x_pelvis


def calc_joint_angles(q_pelvis, q_thigh_left, q_thigh_right, q_shank_left, q_shank_right, q_foot_left, q_foot_right,
                      cal_q_pelvis, cal_q_thigh_left, cal_q_thigh_right, cal_q_shank_left, cal_q_shank_right,
                      cal_q_foot_left, cal_q_foot_right):
    q_cal_pelvis = np.conjugate(cal_q_pelvis[0])*cal_q_pelvis[0] # this is just the identity quaternion (ronin)
    q_cal_thigh_left = np.conjugate(cal_q_thigh_left[0])*cal_q_pelvis[0]
    q_cal_thigh_right = np.conjugate(cal_q_thigh_right[0])*cal_q_pelvis[0]
    q_cal_shank_left = np.conjugate(cal_q_shank_left[0])*cal_q_pelvis[0]
    q_cal_shank_right = np.conjugate(cal_q_shank_right[0])*cal_q_pelvis[0]
    q_cal_foot_left = np.conjugate(cal_q_foot_left[0])*cal_q_pelvis[0]
    q_cal_foot_right = np.conjugate(cal_q_foot_right[0])*cal_q_pelvis[0]

    def get_joint_angle(q0_cal, q0_arr, q1_cal, q1_arr):
        joint_angle_arr = []
        for i in range(len(q0_arr)):
            joint_angle_arr.append(np.array(get_angle_from_quaternion(quaternion.as_float_array(np.conjugate(q0_arr[i]*q0_cal)*q1_arr[i]*q1_cal)))) # (ronin)
        return np.array(joint_angle_arr)
    q_hip_left_angle = get_joint_angle(q_cal_pelvis, q_pelvis, q_cal_thigh_left, q_thigh_left)
    q_hip_right_angle = get_joint_angle(q_cal_pelvis, q_pelvis, q_cal_thigh_right, q_thigh_right)
    q_knee_left_angle = get_joint_angle(q_cal_thigh_left, q_thigh_left, q_cal_shank_left, q_shank_left)
    q_knee_right_angle = get_joint_angle(q_cal_thigh_right, q_thigh_right, q_cal_shank_right, q_shank_right)
    q_ankle_left_angle = get_joint_angle(q_cal_shank_left, q_shank_left, q_cal_foot_left, q_foot_left)
    q_ankle_right_angle = get_joint_angle(q_cal_shank_right, q_shank_right, q_cal_foot_right, q_foot_right)
 
    return q_hip_left_angle, q_hip_right_angle, q_knee_left_angle, q_knee_right_angle, q_ankle_left_angle, q_ankle_right_angle


for thigh_combination in thigh_combinations:

    cal_imu_placement = {'torso_imu': '_00B4D7D4', 'pelvis_imu': '_00B4D7D3', 
                        'calcn_r_imu': '_00B4D7FE', 'tibia_r_imu': '_00B4D7FB', 'femur_r_imu': rthigh[thigh_combination[0]],
                        'calcn_l_imu': '_00B4D7FF', 'tibia_l_imu': '_00B4D7CE', 'femur_l_imu': lthigh[thigh_combination[1]]}
    imu_placement = {'torso_imu': '_00B4D7D4', 'pelvis_imu': '_00B4D7D3', 
                    'calcn_r_imu': '_00B4D7FE', 'tibia_r_imu': '_00B4D7FB', 'femur_r_imu': rthigh[thigh_combination[2]],
                    'calcn_l_imu': '_00B4D7FF', 'tibia_l_imu': '_00B4D7CE', 'femur_l_imu': lthigh[thigh_combination[3]]}

    save_path = 'results\\aim2and3\\uncon\\imu_uncon_ik_calr' + str(thigh_combination[0]+1) + 'l' + str(thigh_combination[1]+1) + '_runr' + str(thigh_combination[2]+1) + 'l' + str(thigh_combination[3]+1) + '_npy'
    for path in paths:
        acc_x_pelvis = None
        subject, task_type = path.split("\\")[1:3]
        if not os.path.exists(save_path + '\\' + subject):
            os.makedirs(save_path + '\\' + subject)
        for filter_type in filters:
            print("Processing " + path + " using filter_type " + filter_type)
            q_pelvis, q_thigh_left, q_thigh_right, q_shank_left, q_shank_right, q_foot_left, q_foot_right, acc_x_pelvis = get_quaternions(path, filter_type, calibration_time[subject], imu_placement)
            cal_q_pelvis, cal_q_thigh_left, cal_q_thigh_right, cal_q_shank_left, cal_q_shank_right, cal_q_foot_left, cal_q_foot_right, _ = get_quaternions(path, filter_type, calibration_time[subject], cal_imu_placement)

            q_hip_left_angle, q_hip_right_angle, q_knee_left_angle, q_knee_right_angle, q_ankle_left_angle, q_ankle_right_angle = calc_joint_angles(q_pelvis, q_thigh_left, q_thigh_right, q_shank_left, q_shank_right, q_foot_left, q_foot_right,
                                                                                                                                                    cal_q_pelvis, cal_q_thigh_left, cal_q_thigh_right, cal_q_shank_left, cal_q_shank_right, cal_q_foot_left, cal_q_foot_right)
            joint_angle_dict = {'hip_left': q_hip_left_angle, 'hip_right': q_hip_right_angle,
                        'knee_left': q_knee_left_angle, 'knee_right': q_knee_right_angle,
                        'ankle_left': q_ankle_left_angle, 'ankle_right': q_ankle_right_angle}

            for joint_name, joint_angles in joint_angle_dict.items():
                with open(save_path + '\\' + subject + '\\' + filter_type + '_' + subject + '_' + task_type + '_' + joint_name + '.npy', 'wb') as f:
                    np.save(f, joint_angles)
        
        if acc_x_pelvis is not None: # just to make sure that its not None
            with open(save_path + '\\' + subject + '\\' + subject + '_' + task_type + '_pelvis_acc_x.npy', 'wb') as f:
                np.save(f, acc_x_pelvis)
