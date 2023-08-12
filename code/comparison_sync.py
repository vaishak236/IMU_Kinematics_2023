# name: comparison_utils.py
# description: Useful functions for comparison
# author: Vu Phan
# date: 2023/03/023

import numpy as np 
import math
import matplotlib.pyplot as plt


sbj = 4

if sbj == 2:
	sbj_folder = 's2_vaishak'
elif sbj == 3:
	sbj_folder = 's3_vu'
elif sbj == 4:
	sbj_folder = 's4_alex'
else:
	pass

mocap_path	= '../results/uncon/mocap_uncon_ik_npy/'
imu_path	= '../results/uncon/imu_uncon_ik_npy_default/' + sbj_folder + '/'

mocap_sync_fn	= 'OpenSim_Sub0' + str(sbj) + '_CombinedTasks_markers_mocap_pelvis_acc.npy'
imu_sync_fn		= sbj_folder + '_combine_10min_pelvis_acc_x.npy'


def get_rmse(mocap, imu):
	MSE = np.square(np.subtract(mocap, imu)).mean()  
	RMSE = math.sqrt(MSE)

	return RMSE


with open(mocap_path + mocap_sync_fn, 'rb') as f:
	mocap_sync = np.load(f)
mocap_sync = mocap_sync.T
# mocap_sync = mocap_sync[700:-1] # pre-cut the mocap to near 3 jumps

with open(imu_path + imu_sync_fn, 'rb') as f:
	imu_sync = np.load(f)
imu_sync = imu_sync - 9.81
# imu_sync = imu_sync[700:-1] # pre-cut the imu to near 3 jumps



window = 2000
first_start = 'mocap'

shifting_id = 0
prev_err = 999
error = []

for i in range(2000):
	if first_start == 'mocap':
		curr_err = get_rmse(mocap_sync[i:(window + i)], imu_sync[0:window])
	else:
		curr_err = get_rmse(mocap_sync[0:window], imu_sync[i:(window + i)])

	error.append(curr_err)

	if curr_err < prev_err:
		shifting_id = i 
		prev_err = curr_err
	else:
		pass

print(first_start)
print(shifting_id)

# shifting_id = 0

plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(9,7), dpi=100, gridspec_kw={'hspace': 0.3})

if first_start == 'mocap':
	ax[0].plot(mocap_sync[shifting_id:-1], linewidth = 2.5, c = 'b', alpha = 0.4, label = 'Mocap')
	ax[0].plot(imu_sync, linewidth = 2.5, c = 'r', alpha = 0.4, label = 'IMU')
else:
	ax[0].plot(mocap_sync, linewidth = 2.5, c = 'b', alpha = 0.4, label = 'Mocap')
	ax[0].plot(imu_sync[shifting_id:-1], linewidth = 2.5, c = 'r', alpha = 0.4, label = 'IMU')
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Vertical acc. of pelvis $(m/s^2)$')
# ax[0].set_title('Synchronization using pelvis vertical acc.')
ax[0].set_xlim([0, 550])
ax[0].set_ylim([-25, 40])
ax[0].spines['left'].set_position(('outward', 8))
ax[0].spines['bottom'].set_position(('outward', 5))
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].legend(loc = 'upper left', frameon = False, prop={'size': 13})


ax[1].plot(error, linewidth = 2.5, c = 'gray', alpha = 0.4, label = 'Mocap vs. IMU')
ax[1].scatter(shifting_id, error[shifting_id], s = 50, c = 'r', alpha = 0.4, label = 'Best match')
ax[1].set_xlabel('Sample')
ax[1].set_ylabel('RMSE $(m/s^2)$')
ax[1].set_xlim([0, 1500])
ax[1].set_ylim([0, 10])
ax[1].spines['left'].set_position(('outward', 8))
ax[1].spines['bottom'].set_position(('outward', 5))
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].legend(loc = 'lower left', frameon = False, prop={'size': 13})

plt.show()




