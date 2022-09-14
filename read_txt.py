# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from dmp_encodetraj import EncodeTraj
from dmp_gentraj import GenerateTraj
import pandas as pd

df_in = pd.read_table('test1.txt', sep=',')
df_out = df_in
dd = df_in.values
y_des = dd[:,[1,2,3,4,5,6,7]].T

dd = EncodeTraj(n_dmps=7, n_bfs=1000)
y_track = []
dy_track = []
ddy_track = []

draw_2 = dd.encode_trajectory(y_des=y_des, dmp_name='draw_2')
traj = GenerateTraj(draw_2)

#goal= [0.4,-0.5]
goal=y_des[:,-1]
goal[0] = goal[0]+0.3
goal[1] = goal[1]+0.2
y0=y_des[:,0]

y_track, dy_track, ddy_track = traj.rollout(y0=y0,goal=goal,tau=1)

#traj.plot_traj()

for i in range(y_track.shape[1]):
    df_out.iloc[:,i+1] = y_track[:,i]
    


df_out.to_csv('test1_2.txt', sep=',',index=False)
