# name: unc_IK_utils.py
# description: Customized functions for processing mocap data
# author: Vu Phan
# date: 2023/03/19

import pandas as pd 
import numpy as np 
import math

from tqdm import tqdm
from scipy import signal
from scipy.spatial.transform import Rotation as R


##########################################
#                                        #
#             Pre-processing             #
#                                        #
##########################################

def get_optitrack_mocap_data(path):
	''' 
	Read and format mocap data collected by OptiTrack system
	Input:
	- path: Directory containing mocap data (which is a .csv file)
	Output:
	- dt: Mocap data
	'''
	
	dt = pd.read_csv(path, skiprows = 3)
	dt = dt.iloc[:, 1:] 
	dt = dt.iloc[2:, :] 

	names_pos 	= list(dt.columns)
	names_pos	= [name.split(':')[1][0:4] for name in names_pos[1:]]
	names_pos	= [''] + names_pos
	names_axis	= dt.iloc[0, :]
	names 	 	= []
	for i in range(len(names_pos)):
		names.append(names_pos[i] + ' ' + names_axis[i])
	dt 			= dt.iloc[1:, :]
	dt.columns 	= names

	dt = dt.reset_index() # reset index after removing columns 
	dt = dt.iloc[:, 1:] 

	dt = dt.astype('float64') # format elements to be float type

	return dt

def get_optitrack_mocap_data_2(path):
	''' 
	Read and format mocap data collected by OptiTrack system
	Input:
	- path: Directory containing mocap data (which is a .trc file)
	Output:
	- dt: Mocap data
	'''
	
	with open(path, 'r') as f:
		txt 	= f.readlines()
		header	= txt[3].split('\t')

	header 		= header[2::]
	d_header 	= []
	for i in range(len(header)):
		d_header.append(header[i].strip('\n') + ' X')
		d_header.append(header[i].strip('\n') + ' Y')
		d_header.append(header[i].strip('\n') + ' Z')

	value	= np.genfromtxt(path, delimiter='\t', skip_header=5)
	time	= 1*value[:, 1]
	value	= value[:, 2::]

	dt 		= pd.DataFrame(value, columns = d_header)
	time 	= pd.DataFrame(time, columns = ['Time'])

	return dt, time

def crop_data(dt):
	'''
	Crop to the valid period of data
	Input:
	- dt: Data in pd.DataFrame
	Output:
	- cropped_dt: Cropped data in pd.DataFrame
	'''

	pass # tbd

def lowpass_filter(dt, dt_freq, cutoff_freq, filter_order):
	'''
	Lowpass filter the given data
	Input:
	- dt: Data that need to be filtered
	- dt_freq: Original sampling rate of the given data
	- cutoff_freq: Cut-off frequency of the filter
	- filter_order: Order of the filter
	Output:
	- filtered_dt: Filtered data
	'''

	Wn 						= cutoff_freq*2/dt_freq
	b, a 					= signal.butter(filter_order, Wn, btype = 'low')
	filtered_dt 			= signal.filtfilt(b, a, dt, axis = 0) # not accounted the 1st column since it's "time"
	filtered_dt 			= pd.DataFrame(filtered_dt, columns = dt.columns)
	filtered_dt.iloc[:, 0] 	= dt.iloc[:, 0]

	return filtered_dt

def downsample(dt, dt_freq, targ_freq):
	''' 
	Downsample data
	Input:
	- dt: Input data (n x m), where n is no. of samples, and m is no. of columns/features
	- dt_freq: Origninal sampling rate
	- targ_freq: Target sampling rate
	Output:
	- ds_dt: Downsampled data
	'''

	num_curr_samples = dt.shape[0]
	num_targ_samples = int((num_curr_samples*targ_freq)/dt_freq)
	ds_dt = signal.resample(dt, num_targ_samples)

	ds_dt = pd.DataFrame(ds_dt, columns = dt.columns)

	return ds_dt


##########################################
#                                        #
#             Transformation             #
#                                        #
##########################################

def get_pelvis_transformation(dt, side, i):
	''' 
	Obtain transformation from lab to pelvis
	Input:
	- dt: Mocap data
	- side: 'R' for right or 'L' for left side
	- i: index of the current sample
	Output:
	- lab_to_pelvis: Transformation from lab coordinate system to pelvis coordinate system
	'''

	rasi 	= np.array([dt['RASI X'][i], dt['RASI Y'][i], dt['RASI Z'][i]])
	lasi 	= np.array([dt['LASI X'][i], dt['LASI Y'][i], dt['LASI Z'][i]])
	rpsi 	= np.array([dt['RPS2 X'][i], dt['RPS2 Y'][i], dt['RPS2 Z'][i]])	
	lpsi 	= np.array([dt['LPS2 X'][i], dt['LPS2 Y'][i], dt['LPS2 Z'][i]])
	psis 	= (rpsi + lpsi)/2.0
	asis 	= (rasi + lasi)/2.0
	gtr 	= np.array([dt[side + 'GTR X'][i], dt[side + 'GTR Y'][i], dt[side + 'GTR Z'][i]])
	lep 	= np.array([dt[side + 'LEP X'][i], dt[side + 'LEP Y'][i], dt[side + 'LEP Z'][i]])
	lml 	= np.array([dt[side + 'LML X'][i], dt[side + 'LML Y'][i], dt[side + 'LML Z'][i]])

	pelvis_depth	= np.linalg.norm(asis - psis)
	pelvis_width	= np.linalg.norm(rasi - lasi)
	leg_length 		= np.linalg.norm(gtr - lep) + np.linalg.norm(lep - lml)

	vz 			= rasi - lasi
	temp_vec 	= np.cross(vz, rasi - psis)
	vx 			= np.cross(temp_vec, vz)
	vy			= np.cross(vz, vx)
	# rps1 		= np.array([dt['APEX X'][i], dt['APEX Y'][i], dt['APEX Z'][i]])
	# vy 			= rps1 - rpsi
	# vx			= np.cross(vy, vz)

	fx 		= vx/np.linalg.norm(vx)
	fy 		= vy/np.linalg.norm(vy)
	fz		= vz/np.linalg.norm(vz)
	if side == 'R':
		hip_origin 	= asis + (-0.24*pelvis_depth - 9.9/1000)*fx + \
					(-0.16*pelvis_width - 0.04*leg_length - 7.1/1000)*fy + \
					(0.28*pelvis_depth + 0.16*pelvis_width + 7.9/1000)*fz
	else:
		hip_origin 	= asis + (-0.24*pelvis_depth - 9.9/1000)*fx + \
					(-0.16*pelvis_width - 0.04*leg_length - 7.1/1000)*fy - \
					(0.28*pelvis_depth + 0.16*pelvis_width + 7.9/1000)*fz

	fx 				= np.append(fx, 0)
	fy 				= np.append(fy, 0)
	fz 				= np.append(fz, 0)	
	# hip_origin		= 1*asis
	hip_origin		= np.append(hip_origin, 1)
	lab_to_pelvis	= np.transpose([fx, fy, fz, hip_origin])

	return lab_to_pelvis

def get_femur_transformation(dt, side, i):
	'''
	Obtain transformation from lab to femur
	Input:
	- dt: Mocap data
	- side: 'R' for right or 'L' for left side
	- i: index of the current sample
	Output:
	- lab_to_femur: Transformation from lab coordinate system to femur coordinate system
	'''

	lhip 	= np.array([dt[side + 'GTR X'][i], dt[side + 'GTR Y'][i], dt[side + 'GTR Z'][i]])
	lknee_l = np.array([dt[side + 'LEP X'][i], dt[side + 'LEP Y'][i], dt[side + 'LEP Z'][i]])
	lknee_m = np.array([dt[side + 'MEP X'][i], dt[side + 'MEP Y'][i], dt[side + 'MEP Z'][i]])
	lknee_o = (lknee_l + lknee_m)/2.0

	vy		= lhip - lknee_o
	# vy		= lhip - lknee_l
	temp_v1 = lhip - lknee_l
	temp_v2 = lknee_m - lknee_l
	vztemp 	= np.cross(temp_v1, temp_v2)

	vz 		= np.cross(vztemp, vy)
	vx		= np.cross(vy, vz)
	fx 		= vx/np.linalg.norm(vx)
	fy 		= vy/np.linalg.norm(vy)
	fz		= vz/np.linalg.norm(vz)

	fx 				= np.append(fx, 0)
	fy 				= np.append(fy, 0)
	fz 				= np.append(fz, 0)
	knee_origin		= np.append(lknee_o, 1)
	lab_to_femur	= np.transpose([fx, fy, fz, knee_origin])

	return lab_to_femur

def get_tibia_transformation(dt, side, i):
	'''
	Obtain transformation from lab to tibia
	Input:
	- dt: Mocap data
	- side: 'R' for right or 'L' for left side
	- i: Index of the current sample
	Output:
	- lab_to_tibia: Transformation from lab coordinate system to tibia coordinate system
	'''

	lknee_l = np.array([dt[side + 'LEP X'][i], dt[side + 'LEP Y'][i], dt[side + 'LEP Z'][i]])
	lknee_m = np.array([dt[side + 'MEP X'][i], dt[side + 'MEP Y'][i], dt[side + 'MEP Z'][i]])
	lknee_o = (lknee_l + lknee_m)/2.0
	lankle_l = np.array([dt[side + 'LML X'][i], dt[side + 'LML Y'][i], dt[side + 'LML Z'][i]])
	lankle_m = np.array([dt[side + 'MML X'][i], dt[side + 'MML Y'][i], dt[side + 'MML Z'][i]])
	lankle_o = (lankle_l + lankle_m)/2.0

	vy			= lknee_o - lankle_o
	vztempknee	= lknee_m - lknee_l 
	vx 			= np.cross(vy, vztempknee)
	vz 			= np.cross(vx, vy)
	fx			= vx/np.linalg.norm(vx)
	fy 			= vy/np.linalg.norm(vy)
	fz			= vz/np.linalg.norm(vz)

	fx 				= np.append(fx, 0)
	fy 				= np.append(fy, 0)
	fz 				= np.append(fz, 0)
	knee_origin		= np.append(lknee_o, 1)
	lab_to_tibia	= np.transpose([fx, fy, fz, knee_origin])

	return lab_to_tibia

def get_foot_transformation(dt, side, i):
	''' 
	Obtain transformation from lab to foot
	Input:
	- dt: Mocap data
	- side: 'R' for right or 'L' for left side
	- i: index of the current sample
	Output:
	- lab_to_foot: Transformation from lab coordinate system to foot coordinate system
	'''

	mt1 	= np.array([dt[side + '1MT X'][i], dt[side + '1MT Y'][i], dt[side + '1MT Z'][i]])
	mt5 	= np.array([dt[side + '5MT X'][i], dt[side + '5MT Y'][i], dt[side + '5MT Z'][i]])
	cal 	= np.array([dt[side + 'CAL X'][i], dt[side + 'CAL Y'][i], dt[side + 'CAL Z'][i]])
	mml 	= np.array([dt[side + 'MML X'][i], dt[side + 'MML Y'][i], dt[side + 'MML Z'][i]])
	lml 	= np.array([dt[side + 'LML X'][i], dt[side + 'LML Y'][i], dt[side + 'LML Z'][i]])

	temp_vec1	= mt1 - cal
	temp_vec2 	= mt5 - cal 

	if side == 'R':
		# temp_vec 	= lml - mml 
		vy			= np.cross(temp_vec2, temp_vec1)
	else:
		# temp_vec 	= mml - lml 
		vy 			= np.cross(temp_vec1, temp_vec2)
	temp_vec 		= mml - lml 
	vx 				= np.cross(vy, temp_vec)
	vz 				= np.cross(vx, vy)
	fx				= vx/np.linalg.norm(vx)
	fy 				= vy/np.linalg.norm(vy)
	fz				= vz/np.linalg.norm(vz)

	fx 				= np.append(fx, 0)
	fy 				= np.append(fy, 0)
	fz 				= np.append(fz, 0)	
	ankle_origin	= (mml + lml)/2.0
	ankle_origin   	= np.append(ankle_origin, 1)
	lab_to_foot	= np.transpose([fx, fy, fz, ankle_origin])

	return lab_to_foot


##########################################
#                                        #
#                   IK                   #
#                                        #
##########################################

# --- Directly from the anatomical landmarks --- #
def get_angle_from_anatomical_markers(dt, side):
	'''
	Compute joint angles from the anatomical markers
	Input:
	- dt: Motion capture data
	- side: 'R' for right or 'L' for left side
	Output:
	- hip_angle, knee_angle, ankle_angle: dictionary of x-, y-, and z- angles
	'''

	num_samples 	= dt.shape[0]
	hip_angle		= {'x': np.zeros(num_samples), 'y': np.zeros(num_samples), 'z': np.zeros(num_samples)}
	knee_angle 		= {'x': np.zeros(num_samples), 'y': np.zeros(num_samples), 'z': np.zeros(num_samples)}
	ankle_angle		= {'x': np.zeros(num_samples), 'y': np.zeros(num_samples), 'z': np.zeros(num_samples)}

	# print('*** Start calculating knee angle: \n')

	for i in tqdm(range(num_samples)):
		lab_to_pelvis	= get_pelvis_transformation(dt, side, i)
		lab_to_femur	= get_femur_transformation(dt, side, i)
		lab_to_tibia 	= get_tibia_transformation(dt, side, i)
		lab_to_foot		= get_foot_transformation(dt, side, i)

		pelvis_to_femur 			= np.matmul(np.linalg.inv(lab_to_pelvis), lab_to_femur)
		angle_x, angle_y, angle_z 	= transformation_to_angle(pelvis_to_femur)
		hip_angle['x'][i] = angle_x
		hip_angle['y'][i] = angle_y
		hip_angle['z'][i] = angle_z

		femur_to_tibia 				= np.matmul(np.linalg.inv(lab_to_femur), lab_to_tibia)
		angle_x, angle_y, angle_z 	= transformation_to_angle(femur_to_tibia)
		knee_angle['x'][i] = angle_x
		knee_angle['y'][i] = angle_y
		if side == 'R':
			knee_angle['z'][i] = angle_z
		else:
			knee_angle['z'][i] = -angle_z

		tibia_to_foot 				= np.matmul(np.linalg.inv(lab_to_tibia), lab_to_foot)
		angle_x, angle_y, angle_z 	= transformation_to_angle(tibia_to_foot)
		ankle_angle['x'][i] = angle_x
		ankle_angle['y'][i] = angle_y
		ankle_angle['z'][i] = angle_z

	return hip_angle, knee_angle, ankle_angle


# --- Through tracking clusters --- #
# TBD


##########################################
#                                        #
#            Post-processing             #
#                                        #
##########################################

def transformation_to_angle(transformation):
	''' 
	Obtain angle from the transformation 
	Input: 
	- transformation: Transformation matrix (4x4) including rotation and translation
	Output:
	- angle_x, angle_y, angle_z: Euler angle in the x, y, and z axis
	'''

	# R		= transformation[0:3, 0:3]
	# angle_x	= math.atan2(R[2][1], R[2][2])
	# angle_y = math.atan2(-R[2][0], np.sqrt((R[0][0])**2 + (R[1][0])**2))
	# angle_z = math.atan2(R[1][0], R[0][0])

	# angle_x = np.rad2deg(angle_x)
	# angle_y = np.rad2deg(angle_y)
	# angle_z = np.rad2deg(angle_z)

	r 		= R.from_matrix(transformation[0:3, 0:3])
	angle 	= r.as_euler('zyx', degrees = True)
	angle_x	= angle[2]
	angle_y	= angle[1]
	angle_z	= angle[0]

	return angle_x, angle_y, angle_z






