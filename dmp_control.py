# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dmp_encodetraj import EncodeTraj
from dmp_gentraj import GenerateTraj
from fk_baxter import FKine
from ik_baxter import ikine
import pandas as pd

parser = argparse.ArgumentParser(description="Use DMP to control the left arm of the Baxter robot")
parser.add_argument('-rf', '--input-file', type=str, default='pick.txt',
                        help="assign the name of input trajecoty file")
parser.add_argument('-wf', '--output-file', type=str, default='pick_gen.txt',
                        help="assign the name of output trajecoty file")
parser.add_argument('-t', '--tau', type=float, default=1.0,
                        help="time scale tau")
parser.add_argument('-nb', '--n-bfs', type=int, default=1000,
                        help="number of Gaussian basis functions")
arg = parser.parse_args()

input_file = arg.input_file
output_file = arg.output_file


### read the joint trajectory file ###
df_in = pd.read_csv(input_file, sep=',')
#cut = []    
#for ii in range(0,df_in.shape[0]-2,3):
#    cut.append(ii+1)
#    cut.append(ii+2)
#df_out = df_in.drop(cut)
df_out = df_in

dd = df_out.values
joint_traj = dd[:,[1,2,3,4,5,6,7]]
print('read done')

### fk to get position trajectory ###
a = 0
pos_traj = joint_traj * 0
fk = FKine()
for joint_angles in joint_traj:
    fk_pos = fk.fkine(joint_angles = [angle for angle in joint_angles], arm = 'left')
    #q = np.array([angle for angle in joint_angles])
    pos_traj[a] = fk_pos
    a = a+1
print('')
print('fk done')

### dmp encode trajecotory ###
y_des = pos_traj.T

encode = EncodeTraj(n_dmps=7, n_bfs=arg.n_bfs)
y_track = []
dy_track = []
ddy_track = []
draw_2 = encode.encode_trajectory(y_des=y_des, dmp_name='draw_2')

### dmp generate trajecotory ###
traj = GenerateTraj(draw_2)
goal=y_des[:,-1]
#goal[2] = goal[2]+0.2
y0=y_des[:,0]
y_track, dy_track, ddy_track = traj.rollout(y0=y0,goal=goal)
print('dmp done')

### ik to get joint trajectory ###
b = 0
joint_ik = y_track * 0
seed = joint_traj[0]
#print(seed)
for pos in y_track:
    ik_result = ikine(pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6],seed = seed, arm='left')
    joint_ik[b] = ik_result
    #seed = [float('%.8f'%j) for j in seed]
    seed = ik_result
    b = b+1
    #print(seed)
    #print(ik_result)
print('ik done')

### output the trajectory ###
#df_pos = pd.DataFrame(columns = ['x','y','z','qx','qy','qz','qw']) 
#for i in range(pos_traj.shape[1]):
#    df_pos.iloc[:,i] = pos_traj[:,i]
#df_pos.to_csv('test1_pos.txt', sep=',',index=False)

for i in range(joint_ik.shape[1]):
    df_out.iloc[:,i+1] = joint_ik[:,i]

df_out.to_csv(output_file, sep=',',index=False)
print('write done')

