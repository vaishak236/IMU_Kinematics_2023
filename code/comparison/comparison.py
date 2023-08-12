# name: test_compare.py
# description: Compare angles from different methods
# author: Vu Phan
# date: 2023/03/20


import numpy as np 
import math
import matplotlib.pyplot as plt


mocap_path	= '../../results/uncon/mocap_uncon_ik_npy/'
mocap_fn	= 'L_OpenSim_Sub03_CombinedTasks_markersmocap_knee_angle.npy'

imu_path	= '../../results/uncon/imu_uncon_ik_npy/s3_vu/'
comp_fn 	= 'Complementary_s3_vu_combine_10min_knee_left.npy'
madgwick_fn = 'Madgwick_s3_vu_combine_10min_knee_left.npy'
mahony_fn 	= 'Mahony_s3_vu_combine_10min_knee_left.npy'
ekf_fn 		= 'EKF_s3_vu_combine_10min_knee_left.npy'
xsens_fn	= 'Xsens_s3_vu_combine_10min_knee_left.npy'

imu_angle_id = 1

shifting_id = 763

def get_rmse(mocap, imu):
	MSE = np.square(np.subtract(mocap, imu)).mean()  
	RMSE = math.sqrt(MSE)

	return RMSE

with open(mocap_path + mocap_fn, 'rb') as f:
	mocap = np.load(f, allow_pickle = True)

mocap = mocap[0]['z']
mocap = mocap[shifting_id:-1]

# mocap = mocap[1285:2700] # sit-to-stand 1
# mocap = mocap[10885:12435] # sit-to-stand 2
mocap = mocap[23241:24679] # sit-to-stand 3
# mocap = mocap[12783:13083] # walking 2


with open(imu_path + madgwick_fn, 'rb') as f:
	imu_madgwick = np.load(f)
imu_madgwick = imu_madgwick[:, imu_angle_id]
# imu_madgwick = imu_madgwick[1285:2700] # sit-to-stand 1
# imu_madgwick = imu_madgwick[10885:12435] # sit-to-stand 2
imu_madgwick = imu_madgwick[23241:24679] # sit-to-stand 3
# imu_madgwick = imu_madgwick[12783:13083] # walking 2

with open(imu_path + mahony_fn, 'rb') as f:
	imu_manohy = np.load(f)
imu_manohy = imu_manohy[:, imu_angle_id]
# imu_manohy = imu_manohy[1285:2700] # sit-to-stand 1
# imu_manohy = imu_manohy[10885:12435] # sit-to-stand 2
imu_manohy = imu_manohy[23241:24679] # sit-to-stand 3
# imu_manohy = imu_manohy[12783:13083] # walking 2

with open(imu_path + comp_fn, 'rb') as f:
	imu_comp = np.load(f)
imu_comp = imu_comp[:, imu_angle_id]
# imu_comp = imu_comp[1285:2700] # sit-to-stand 1
# imu_comp = imu_comp[10885:12435] # sit-to-stand 2
imu_comp = imu_comp[23241:24679] # sit-to-stand 3
# imu_comp = imu_comp[12783:13083] # walking 2

with open(imu_path + ekf_fn, 'rb') as f:
	imu_ekf = np.load(f)
imu_ekf = imu_ekf[:, imu_angle_id]
# imu_ekf = imu_ekf[1285:2700] # sit-to-stand 1
# imu_ekf = imu_ekf[10885:12435] # sit-to-stand 2
imu_ekf = imu_ekf[23241:24679] # sit-to-stand 3
# imu_ekf = imu_ekf[12783:13083] # walking 2

with open(imu_path + xsens_fn, 'rb') as f:
	imu_xsens = np.load(f)
imu_xsens = imu_xsens[:, imu_angle_id]
# imu_xsens = imu_xsens[1285:2700] # sit-to-stand 1
# imu_xsens = imu_xsens[10885:12435] # sit-to-stand 2
imu_xsens = imu_xsens[23241:24679] # sit-to-stand 3
# imu_xsens = imu_xsens[12783:13083] # walking 2

print(get_rmse(mocap, imu_madgwick))
print(get_rmse(mocap, imu_manohy))
print(get_rmse(mocap, imu_comp))
print(get_rmse(mocap, imu_ekf))
print(get_rmse(mocap, imu_xsens))


plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(9,6), dpi=100, gridspec_kw={'hspace': 0.3})
ax.plot(mocap, linewidth = 2.5, alpha = 0.4, label = 'Mocap')
ax.plot(imu_madgwick, linewidth = 2.5, alpha = 0.4, label = 'Madgwick')
ax.plot(imu_manohy, linewidth = 2.5, alpha = 0.4, label = 'Mahony')
ax.plot(imu_comp, linewidth = 2.5, alpha = 0.4, label = 'Complimentary')
ax.plot(imu_ekf, linewidth = 2.5, alpha = 0.4, label = 'EKF')
ax.plot(imu_xsens, linewidth = 2.5, alpha = 0.4, label = 'Xsens')
ax.set_ylim([-5, 100])
ax.set_xlim([0, len(imu_xsens)])
ax.set_xlabel('Sample')
ax.set_ylabel('Angle $(^o)$')

ax.spines['left'].set_position(('outward', 8))
ax.spines['bottom'].set_position(('outward', 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc = 'upper left', frameon = False, prop={'size': 13}, ncol = 3)
plt.show()

