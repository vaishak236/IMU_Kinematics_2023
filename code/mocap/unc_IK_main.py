# name: unc_IK_main.py
# description: Perform unconstrained IK for the mocap data
# author: Vu Phan
# date: 2023/03/19


import pandas as pd 
import numpy as np 
import math
import matplotlib.pyplot as plt 

from unc_IK_init import *
from unc_IK_utils import *

from scipy import signal


# --- Setup --- #
path 			= '../../data_mocap/'
selected_side 	= side_id['left']
sbj				= 4

# --- Calculate offset --- #
print('*** Get offset')
static_fn		= 'OpenSim_Sub0' + str(sbj) + '_static_markers.trc'
offset_dt, _ 	= get_optitrack_mocap_data_2(path + static_fn) # read and pre-process Optitrack mocap data
offset_dt 		= offset_dt.astype('float64')

hip_offset, knee_offset, ankle_offset 	= get_angle_from_anatomical_markers(offset_dt, selected_side)

print('- Completed! \n')

# --- Obtain the mocap data --- #
print('*** Start reading data')
dt_fn			= 'OpenSim_Sub0' + str(sbj) + '_CombinedTasks_markers.trc'
mocap_dt, time 	= get_optitrack_mocap_data_2(path + dt_fn) # read and pre-process Optitrack mocap data
mocap_dt 		= mocap_dt.astype('float64')

print('- Completed! \n')

print('*** Start processing data')
mocap_dt 		= lowpass_filter(mocap_dt, mocap_const['sampling_rate'], mocap_const['filter_cutoff'], mocap_const['filter_order']) # filter the data
mocap_dt 	 	= downsample(mocap_dt, mocap_const['sampling_rate'], imu_const['sampling_rate'])

print('- Completed! \n')

# --- Perform IK --- #
print('*** Start IK')
hip_angle, knee_angle, ankle_angle 	= get_angle_from_anatomical_markers(mocap_dt, selected_side)

print('- Completed! \n')

print('*** Compensate offset')
hip_angle['x']		= hip_angle['x'] - hip_offset['x'].mean()
hip_angle['y']		= hip_angle['y'] - hip_offset['y'].mean()
hip_angle['z']		= hip_angle['z'] - hip_offset['z'].mean()

knee_angle['z']		= knee_angle['z'] - knee_offset['z'].mean()

ankle_angle['z']	= ankle_angle['z'] - ankle_offset['z'].mean()

print('- Completed \n')

# --- Visualization --- #
# plt.figure(0)
# plt.plot(hip_angle['z'], label = 'x')
# plt.ylim([0, 100])
# plt.title('Hip flexion')
# plt.figure(1)
# plt.plot(knee_angle['z'], label = 'y')
# plt.ylim([0, 100])
# plt.title('Knee flexion')
# plt.figure(2)
# plt.plot(ankle_angle['z'], label = 'z')
# plt.ylim([0, 100])
# plt.title('Ankle flexion')

# # print(hip_angle['z'].mean())
# # print(knee_angle['z'].mean())
# # print(ankle_angle['z'].mean())

# plt.legend()
# plt.show()


print('*** Export results')
fn = selected_side + '_' + dt_fn [0:-4] + 'mocap_hip_angle.npy'
with open(fn, 'wb') as f:
	np.save(f, [hip_angle])

fn = selected_side + '_' + dt_fn [0:-4] + 'mocap_knee_angle.npy'
with open(fn, 'wb') as f:
	np.save(f, [knee_angle])

fn = selected_side + '_' + dt_fn [0:-4] + 'mocap_ankle_angle.npy'
with open(fn, 'wb') as f:
	np.save(f, [ankle_angle])


print('- Completed \n')