# name: mocap_pelvis_acc.py
# description: export pelvis acceleration on the mocap data for sync
# author: Vu Phan
# date: 2023/03/23


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

# --- Obtain pelvis acceleration --- #
print('*** Start calculating pelvis acceleration')
pelvis_yacc	= 1*mocap_dt['RPS1 Y']
pelvis_yacc = np.diff(pelvis_yacc)/(1/40) # velocity
pelvis_yacc = np.diff(pelvis_yacc)/(1/40) # acceleration

print('- Completed! \n')

# --- Store data to .npy --- #
print('*** Export results')
fn = dt_fn [0:-4] + '_mocap_pelvis_acc.npy'
with open(fn, 'wb') as f:
	np.save(f, [pelvis_yacc])

print('- Completed! \n')

