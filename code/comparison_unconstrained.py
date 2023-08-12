# name: test_compare.py
# description: Compare angles from different methods
# author: Vu Phan
# date: 2023/03/20


import numpy as np 
import math
import matplotlib.pyplot as plt
import pandas as pd

from mocap.unc_IK_utils import downsample

sbj = 1
side = 'right'
joint = 'hip'

if side == 'left':
	if joint == 'knee':
		angle_id = 'knee_angle_l'
	elif joint =='hip':
		angle_id = 'hip_rotation_l'
	elif joint == 'ankle':
		angle_id = 'ankle_angle_l'
else:
	if joint == 'knee':
		angle_id = 'knee_angle_r'
	elif joint =='hip':
		angle_id = 'hip_rotation_r'
	elif joint == 'ankle':
		angle_id = 'ankle_angle_r'


if sbj == 1:
	sbj_folder = 's1_ronin'
elif sbj == 2:
	sbj_folder = 's2_vaishak'
elif sbj == 3:
	sbj_folder = 's3_vu'
elif sbj == 4:
	sbj_folder = 's4_alex'
else:
	pass

mocap_path	= '../results/uncon/mocap_uncon_ik_npy/'
mocap_fn	= side[0].upper() + '_OpenSim_Sub0' + str(sbj) + '_CombinedTasks_markersmocap_' + joint + '_angle.npy'

# mocap_path	= '../results/constrained/mocap/IKResults/s4_alex/'
# # mocap_fn	= 'OpenSim_Sub04_CombinedTasks_Constrained_IK_trial1_task1.mot'
# mocap_fn	= 'OpenSim_Sub04_CombinedTasks_Constrained_IK_trial1_task3.mot'


# imu_path	= '../results/aim2and3/uncon/' # for aim 2
# imu_path	= '../results/aim2and3_default/uncon/' # for aim 2 untuned results
imu_path = '../results/uncon/' # for untuned normal results

imu_folder = 'imu_uncon_ik_npy_default/' # untuned unconstrained normal
# imu_folder = 'imu_uncon_ik_calr1l1_runr1l1_npy/' # misplacement check - tuned unconstrained normal
# imu_folder = 'imu_uncon_ik_calr1l1_runr1l2_npy/' # misplacement check - tuned unconstrained misplacement
# imu_folder = 'imu_uncon_ik_calr2l2_runr2l2_npy/' # noise - skin
# imu_folder = 'imu_uncon_ik_calr3l3_runr3l3_npy/' # noise - cotton pad

imu_path = imu_path + imu_folder + sbj_folder + '/'

comp_fn 	= 'Complementary_' + sbj_folder + '_combine_10min_' + joint + '_' + side + '.npy'
madgwick_fn = 'Madgwick_' + sbj_folder + '_combine_10min_' + joint + '_' + side + '.npy'
mahony_fn 	= 'Mahony_' + sbj_folder + '_combine_10min_' + joint + '_' + side + '.npy'
ekf_fn 		= 'EKF_' + sbj_folder + '_combine_10min_' + joint + '_' + side + '.npy'
xsens_fn	= 'Xsens_' + sbj_folder + '_combine_10min_' + joint + '_' + side + '.npy'

imu_angle_id = 1

if sbj == 1:
	shifting_id = 776
if sbj == 2:
	shifting_id = 1971
elif sbj == 3:
	shifting_id = 763
elif sbj == 4:
	shifting_id = 819
else:
	pass 

def get_rmse(mocap, imu):
	MSE = np.square(np.subtract(mocap, imu)).mean()  
	RMSE = math.sqrt(MSE)

	return RMSE

# ==============================================================
with open(mocap_path + mocap_fn, 'rb') as f: # unconstrained mocap
	mocap = np.load(f, allow_pickle = True)
mocap = mocap[0]['z']
mocap = mocap[shifting_id:-1]
if sbj != 2:
	mocap = mocap - mocap[0]# SUBTRACT OFFSET BASED ON STATIC

# # ==============================================================
# with open(mocap_path + mocap_fn, 'r') as f: # constrained mocap
# 	txt = f.readlines()
# 	header = txt[10].split('\t')

# angles = np.genfromtxt(mocap_path + mocap_fn, delimiter='\t', skip_header=11)
# dt = pd.DataFrame(angles, columns = header)

# dt = downsample(dt, 100, 40)
# mocap = (1)*dt[angle_id]
# mocap = np.array(mocap) + 5 # offset for knee and ankle
# # mocap = np.array(mocap) # offset
# # ==============================================================


if sbj == 1:
	# mocap = mocap[235:1670] # sit-to-stand 1
	mocap = mocap[300:1400] # sit-to-stand 1 (ronin)
	# mocap = mocap[10500:11520] # sit-to-stand 2
	# mocap = mocap[22400:23420] # sit-to-stand 3
	# mocap = mocap[2214:2366] # walking 1
if sbj == 2:
	mocap = mocap[440:1430] # sit-to-stand 1
elif sbj == 3:
	# mocap = mocap[1285:2700] # sit-to-stand 1
	mocap = mocap[10885:12435] # sit-to-stand 2
	# mocap = mocap[23241:24679] # sit-to-stand 3
	# mocap = mocap[12783:13083] # walking 2
elif sbj == 4:
	# mocap = mocap[234:1340] # sit-to-stand 1
	mocap = mocap[1843:1978] # walking 1

with open(imu_path + madgwick_fn, 'rb') as f:
	imu_madgwick = np.load(f)
imu_madgwick = imu_madgwick[:, imu_angle_id]
if (side == 'left') and (joint != 'knee'):
	imu_madgwick = (-1)*imu_madgwick


if sbj == 1:
	# imu_madgwick = imu_madgwick[235:1670] # sit-to-stand 1
	imu_madgwick = imu_madgwick[300:1400] # sit-to-stand 1 (ronin)
	# imu_madgwick = imu_madgwick[10500:11520] # sit-to-stand 2
	# imu_madgwick = imu_madgwick[22400:23420] # sit-to-stand 3
	# imu_madgwick = imu_madgwick[2214:2366] # walking 1
elif sbj == 2:
	imu_madgwick = imu_madgwick[440:1430] # sit-to-stand 1
elif sbj == 3:
	# imu_madgwick = imu_madgwick[1285:2700] # sit-to-stand 1
	imu_madgwick = imu_madgwick[10885:12435] # sit-to-stand 2
	# imu_madgwick = imu_madgwick[23241:24679] # sit-to-stand 3
	# imu_madgwick = imu_madgwick[12783:13083] # walking 2
elif sbj == 4:
	# imu_madgwick = imu_madgwick[234:1340] # sit-to-stand 1
	imu_madgwick = imu_madgwick[1843:1978] # walking 1

with open(imu_path + mahony_fn, 'rb') as f:
	imu_manohy = np.load(f)
imu_manohy = imu_manohy[:, imu_angle_id]
if (side == 'left') and (joint != 'knee'):
	imu_manohy = (-1)*imu_manohy


if sbj == 1:
	# imu_manohy = imu_manohy[235:1670] # sit-to-stand 1
	imu_manohy = imu_manohy[300:1400] # sit-to-stand 1 (ronin)
	# imu_manohy = imu_manohy[10500:11520] # sit-to-stand 2
	# imu_manohy = imu_manohy[22400:23420] # sit-to-stand 3
	# imu_manohy = imu_manohy[2214:2366] # walking 1
elif sbj == 2:
	imu_manohy = imu_manohy[440:1430] # sit-to-stand 1
elif sbj == 3:
	# imu_manohy = imu_manohy[1285:2700] # sit-to-stand 1
	imu_manohy = imu_manohy[10885:12435] # sit-to-stand 2
	# imu_manohy = imu_manohy[23241:24679] # sit-to-stand 3
	# imu_manohy = imu_manohy[12783:13083] # walking 2
elif sbj == 4:
	# imu_manohy = imu_manohy[234:1340] # sit-to-stand 1
	imu_manohy = imu_manohy[1843:1978] # walking 1

# with open(imu_path + comp_fn, 'rb') as f:
# 	imu_comp = np.load(f)
# imu_comp = imu_comp[:, imu_angle_id]
# # imu_comp = imu_comp[1285:2700] # sit-to-stand 1
# # imu_comp = imu_comp[10885:12435] # sit-to-stand 2
# imu_comp = imu_comp[23241:24679] # sit-to-stand 3
# # imu_comp = imu_comp[12783:13083] # walking 2

with open(imu_path + ekf_fn, 'rb') as f:
	imu_ekf = np.load(f)
imu_ekf = imu_ekf[:, imu_angle_id]
if (side == 'left') and (joint != 'knee'):
	imu_ekf = (-1)*imu_ekf

if sbj == 1:
	# imu_ekf = imu_ekf[235:1670] # sit-to-stand 1
	imu_ekf = imu_ekf[300:1400] # sit-to-stand 1 (ronin)
	# imu_ekf = imu_ekf[10500:11520] # sit-to-stand 2
	# imu_ekf = imu_ekf[22400:23420] # sit-to-stand 3
	# imu_ekf = imu_ekf[2214:2366] # walking 1
elif sbj == 2:
	imu_ekf = imu_ekf[440:1430] # sit-to-stand 1
elif sbj == 3:
	# imu_ekf = imu_ekf[1285:2700] # sit-to-stand 1
	imu_ekf = imu_ekf[10885:12435] # sit-to-stand 2
	# imu_ekf = imu_ekf[23241:24679] # sit-to-stand 3
	# imu_ekf = imu_ekf[12783:13083] # walking 2
elif sbj == 4:
	# imu_ekf = imu_ekf[234:1340] # sit-to-stand 1
	imu_ekf = imu_ekf[1843:1978] # walking 1

with open(imu_path + xsens_fn, 'rb') as f:
	imu_xsens = np.load(f)
imu_xsens = imu_xsens[:, imu_angle_id]
if (side == 'left') and (joint != 'knee'):
	imu_xsens = (-1)*imu_xsens

if sbj == 1:
	# imu_xsens = imu_xsens[235:1670] # sit-to-stand 1
	imu_xsens = imu_xsens[300:1400] # sit-to-stand 1 (ronin)
	# imu_xsens = imu_xsens[10500:11520] # sit-to-stand 2
	# imu_xsens = imu_xsens[22400:23420] # sit-to-stand 3
	# imu_xsens = imu_xsens[2214:2366] # walking 1
elif sbj == 2:
	imu_xsens = imu_xsens[440:1430] # sit-to-stand 1
elif sbj == 3:
	# imu_xsens = imu_xsens[1285:2700] # sit-to-stand 1
	imu_xsens = imu_xsens[10885:12435] # sit-to-stand 2
	# imu_xsens = imu_xsens[23241:24679] # sit-to-stand 3
	# imu_xsens = imu_xsens[12783:13083] # walking 2
elif sbj == 4:
	# imu_xsens = imu_xsens[234:1340] # sit-to-stand 1
	imu_xsens = imu_xsens[1843:1978] # walking 1

print('Madgwick = ' + str(get_rmse(mocap, imu_madgwick)))
print('Mahony = ' + str(get_rmse(mocap, imu_manohy)))
# print(get_rmse(mocap, imu_comp))
print('EKF = ' + str(get_rmse(mocap, imu_ekf)))
print('Xsens = ' + str(get_rmse(mocap, imu_xsens)))


plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(9,6), dpi=100, gridspec_kw={'hspace': 0.3})
ax.plot(mocap, linewidth = 2.5, alpha = 0.4, label = 'Mocap')
ax.plot(imu_madgwick, linewidth = 2.5, alpha = 0.4, label = 'Madgwick')
ax.plot(imu_manohy, linewidth = 2.5, alpha = 0.4, label = 'Mahony')
# ax.plot(imu_comp, linewidth = 2.5, alpha = 0.4, label = 'Complimentary')
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

