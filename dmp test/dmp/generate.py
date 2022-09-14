# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from dmp_encodetraj import EncodeTraj
from dmp_gentraj import GenerateTraj
from fk_baxter import FKine
from ik_baxter import ikine

import pandas as pd

df_in = pd.read_table('pick.txt', sep=',')

#cut = []    
#for ii in range(0,df_in.shape[0]-2,3):
#    cut.append(ii+1)
#    cut.append(ii+2)
#df_out = df_in.drop(cut)
df_out = df_in

dd = df_out.values
#print(dd[0])
joint_traj = dd[:,[1,2,3,4,5,6,7]]

a = 0
pos_traj = joint_traj * 0
fk = FKine()
for joint_angles in joint_traj:
    fk_pos = fk.fkine(joint_angles = [angle for angle in joint_angles], arm = 'left')
    #q = np.array([angle for angle in joint_angles])
    pos_traj[a] = fk_pos
    a = a+1


y_des = pos_traj.T

encode = EncodeTraj(n_dmps=7, n_bfs=1000)
y_track = []
dy_track = []
ddy_track = []

draw_2 = encode.encode_trajectory(y_des=y_des, dmp_name='draw_2')
traj = GenerateTraj(draw_2)

goal=y_des[:,-1]
y0=y_des[:,0]

y_track, dy_track, ddy_track = traj.rollout(y0=y0,goal=goal,tau=1)

b = 0
joint_ik = y_track * 0
seed = joint_traj[0]
#print(seed)
for pos in y_track:
    ik_result = ikine(pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6],seed = seed, arm='left')
    joint_ik[b] = ik_result
    seed = [float('%.8f'%j) for j in seed]
    b = b+1
    #print(b)

df_pos = pd.DataFrame(columns = ['x','y','z','qx','qy','qz','qw']) 
for i in range(pos_traj.shape[1]):
    df_pos.iloc[:,i] = pos_traj[:,i]

df_pos.to_csv('test1_pos.txt', sep=',',index=False)

for i in range(joint_ik.shape[1]):
    df_out.iloc[:,i+1] = joint_ik[:,i]

df_out.to_csv('test1_gen.txt', sep=',',index=False)

print(df_out.values[0])