# name: compare_opensim_opensense.py
# description: Compare the IK results between OpenSim (marker) with OpenSense (IMU)
# author: Vu Phan
# date: 2023/04/05



import numpy as np 
import math
import pandas as pd
import matplotlib.pyplot as plt

from mocap.unc_IK_utils import downsample


sync = 819 

freq = 40 # Hz

angle_id = 'hip_rotation_l'

# # Duration of sit-to-stand trial 1
# sts_b1 = 26 # (s)
# sts_e1 = 54

# Duration for walking trial 1
sts_b1 = 61 # (s)
sts_e1 = 72

mocap_path	= '../results/constrained/mocap/IKResults/s4_alex/'
# mocap_fn	= 'OpenSim_Sub04_CombinedTasks_Constrained_IK_trial1_task1.mot'
mocap_fn	= 'OpenSim_Sub04_CombinedTasks_Constrained_IK_trial1_task3.mot'

imu_path	= '../results/constrained/imu/IKResults/s4_alex/'
xsens_fn 	= 'ik_Xsens_s4_alex_combine_10min.mot'
ekf_fn 		= 'ik_EKF_s4_alex_combine_10min.mot'
madgwick_fn = 'ik_Madgwick_s4_alex_combine_10min.mot'
mahony_fn 	= 'ik_Mahony_s4_alex_combine_10min.mot'
comp_fn 	= 'ik_Complementary_s4_alex_combine_10min.mot'

def get_rmse(mocap, imu):
	MSE = np.square(np.subtract(mocap, imu)).mean()  
	RMSE = math.sqrt(MSE)

	return RMSE


with open(mocap_path + mocap_fn, 'r') as f:
	txt = f.readlines()
	header = txt[10].split('\t')

angles = np.genfromtxt(mocap_path + mocap_fn, delimiter='\t', skip_header=11)
dt = pd.DataFrame(angles, columns = header)

dt = downsample(dt, 100, 40)
mocap = (1)*dt[angle_id]
mocap = np.array(mocap) + 5 # offset for knee and ankle
# mocap = np.array(mocap) # offset


# Xsens built-in filter
with open(imu_path + xsens_fn, 'r') as f:
	txt = f.readlines()
	header = txt[6].split('\t')

# print(header)
angles = np.genfromtxt(imu_path + xsens_fn, delimiter='\t', skip_header=7)
dt = pd.DataFrame(angles, columns = header)

imu_xsens = 1*dt[angle_id]
imu_xsens = imu_xsens[sts_b1*freq-sync:sts_e1*freq-sync]
imu_xsens = np.array(imu_xsens)

# EKF
with open(imu_path + xsens_fn, 'r') as f:
	txt = f.readlines()
	header = txt[6].split('\t')

angles = np.genfromtxt(imu_path + ekf_fn, delimiter='\t', skip_header=7)
dt = pd.DataFrame(angles, columns = header)

imu_ekf = 1*dt[angle_id]
imu_ekf = imu_ekf[sts_b1*freq-sync:sts_e1*freq-sync]
imu_ekf = np.array(imu_ekf)

# Madgwick
with open(imu_path + madgwick_fn, 'r') as f:
	txt = f.readlines()
	header = txt[6].split('\t')

angles = np.genfromtxt(imu_path + madgwick_fn, delimiter='\t', skip_header=7)
dt = pd.DataFrame(angles, columns = header)

imu_madgwick = 1*dt[angle_id]
imu_madgwick = imu_madgwick[sts_b1*freq-sync:sts_e1*freq-sync]
imu_madgwick = np.array(imu_madgwick)

# Mahony
with open(imu_path + mahony_fn, 'r') as f:
	txt = f.readlines()
	header = txt[6].split('\t')

angles = np.genfromtxt(imu_path + mahony_fn, delimiter='\t', skip_header=7)
dt = pd.DataFrame(angles, columns = header)

imu_manohy = 1*dt[angle_id]
imu_manohy = imu_manohy[sts_b1*freq-sync:sts_e1*freq-sync]
imu_manohy = np.array(imu_manohy)

# Complementary
with open(imu_path + comp_fn, 'r') as f:
	txt = f.readlines()
	header = txt[6].split('\t')

angles = np.genfromtxt(imu_path + comp_fn, delimiter='\t', skip_header=7)
dt = pd.DataFrame(angles, columns = header)

imu_comp = 1*dt[angle_id]
imu_comp = imu_comp[sts_b1*freq-sync:sts_e1*freq-sync]
imu_comp = np.array(imu_comp)


print(get_rmse(mocap, imu_xsens))
print(get_rmse(mocap, imu_madgwick))
print(get_rmse(mocap, imu_manohy))
print(get_rmse(mocap, imu_comp))
print(get_rmse(mocap, imu_ekf))



# plt.plot(mocap)
# plt.plot(imu_xsens)
# plt.show()



# segment_id = [127, 229, 336, 455, 572, 674, 792, 904, 1005] # for sit-to-stand trial 1

segment_id = [107, 153, 199, 246, 290, 335, 381] # for walking trial 1

s_mocap = []
s_xsens = []
s_madgwick = []
s_mahony = []
s_comp = []
s_ekf = []
for i in range(len(segment_id) - 1):
	N = segment_id[i+1] - segment_id[i]
	s_mocap.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), mocap[segment_id[i]:segment_id[i+1]]))
	s_xsens.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_xsens[segment_id[i]:segment_id[i+1]]))
	s_madgwick.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_madgwick[segment_id[i]:segment_id[i+1]]))
	s_mahony.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_manohy[segment_id[i]:segment_id[i+1]]))
	s_comp.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_comp[segment_id[i]:segment_id[i+1]]))
	s_ekf.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_ekf[segment_id[i]:segment_id[i+1]]))


plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize = (8,6))
ax.plot(100*np.linspace(0, 1, 100), np.mean(s_mocap, axis = 0), label = 'Mocap')
ax.plot(100*np.linspace(0, 1, 100), np.mean(s_xsens, axis = 0), label = 'Xsens')
ax.plot(100*np.linspace(0, 1, 100), np.mean(s_madgwick, axis = 0), label = 'Madgwick')
ax.plot(100*np.linspace(0, 1, 100), np.mean(s_mahony, axis = 0), label = 'Mahony')
ax.plot(100*np.linspace(0, 1, 100), np.mean(s_comp, axis = 0), label = 'Complimentary')
ax.plot(100*np.linspace(0, 1, 100), np.mean(s_ekf, axis = 0), label = 'EKF')

ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_mocap, axis = 0)-2*np.std(s_mocap, axis = 0)), (np.mean(s_mocap, axis = 0)+2*np.std(s_mocap, axis = 0)), alpha=.1)
ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_xsens, axis = 0)-2*np.std(s_xsens, axis = 0)), (np.mean(s_xsens, axis = 0)+2*np.std(s_xsens, axis = 0)), alpha=.1)
ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_madgwick, axis = 0)-2*np.std(s_madgwick, axis = 0)), (np.mean(s_madgwick, axis = 0)+2*np.std(s_madgwick, axis = 0)), alpha=.1)
ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_mahony, axis = 0)-2*np.std(s_mahony, axis = 0)), (np.mean(s_mahony, axis = 0)+2*np.std(s_mahony, axis = 0)), alpha=.1)
ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_comp, axis = 0)-2*np.std(s_comp, axis = 0)), (np.mean(s_comp, axis = 0)+2*np.std(s_comp, axis = 0)), alpha=.1)
ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_ekf, axis = 0)-2*np.std(s_ekf, axis = 0)), (np.mean(s_ekf, axis = 0)+2*np.std(s_ekf, axis = 0)), alpha=.1)

# ax.set_ylim([-40, 140])
ax.set_ylim([-60, 40])
ax.set_xlim([0, 100])
ax.set_xlabel('Repetition duration')
ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax.set_ylabel('Angle $(^o)$')

ax.spines['left'].set_position(('outward', 8))
ax.spines['bottom'].set_position(('outward', 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc = 'upper left', frameon = False, prop={'size': 13}, ncol = 3)
plt.show()








