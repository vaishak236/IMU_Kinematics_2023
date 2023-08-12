# name: test_compare.py
# description: Compare angles from different methods
# author: Vu Phan
# date: 2023/03/20


import numpy as np 
import math
import matplotlib.pyplot as plt

def get_rmse(mocap, imu):
	MSE = np.square(np.subtract(mocap, imu)).mean()  
	RMSE = math.sqrt(MSE)

	return RMSE

with open('L_OpenSim_Sub03_CombinedTasks_markersmocap_knee_angle.npy', 'rb') as f:
	mocap = np.load(f, allow_pickle = True)

mocap = mocap[0]['z']
# mocap = mocap[2010:3480] # sit-to-stand 1
# mocap = mocap[11610:13210] # sit-to-stand 2
# mocap = mocap[23967:25465] # sit-to-stand 3
mocap = mocap[13510:13810] # walking 2

with open('madgwick_left_knee.npy', 'rb') as f:
	imu_madgwick = np.load(f)
# imu_madgwick = imu_madgwick[1285:2755] # sit-to-stand 1
# imu_madgwick = imu_madgwick[10885:12485] # sit-to-stand 2
# imu_madgwick = imu_madgwick[23241:24739] # sit-to-stand 3
imu_madgwick = imu_madgwick[12783:13083] # walking 2

with open('mahony_left_knee.npy', 'rb') as f:
	imu_manohy = np.load(f)
# imu_manohy = imu_manohy[1285:2755] # sit-to-stand 1
# imu_manohy = imu_manohy[10885:12485] # sit-to-stand 2
# imu_manohy = imu_manohy[23241:24739] # sit-to-stand 3
imu_manohy = imu_manohy[12783:13083] # walking 2

with open('complementary_left_knee.npy', 'rb') as f:
	imu_comp = np.load(f)
# imu_comp = imu_comp[1285:2755] # sit-to-stand 1
# imu_comp = imu_comp[10885:12485] # sit-to-stand 2
# imu_comp = imu_comp[23241:24739] # sit-to-stand 3
imu_comp = imu_comp[12783:13083] # walking 2

with open('ekf_left_knee.npy', 'rb') as f:
	imu_ekf = np.load(f)
# imu_ekf = imu_ekf[1285:2755] # sit-to-stand 1
# imu_ekf = imu_ekf[10885:12485] # sit-to-stand 2
# imu_ekf = imu_ekf[23241:24739] # sit-to-stand 3
imu_ekf = imu_ekf[12783:13083] # walking 2


print(get_rmse(mocap, imu_madgwick))
print(get_rmse(mocap, imu_manohy))
print(get_rmse(mocap, imu_comp))
print(get_rmse(mocap, imu_ekf))

plt.plot(mocap, label = 'mocap')
plt.plot(imu_madgwick, label = 'madgwick')
plt.plot(imu_manohy, label = 'mahony')
plt.plot(imu_comp, label = 'complimentary')
plt.plot(imu_ekf, label = 'EKF')
plt.ylim([-5, 135])
plt.xlim([0, 1498])
plt.legend()
plt.show()

