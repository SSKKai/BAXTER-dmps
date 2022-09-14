# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from dmp.dmp_encodetraj import EncodeTraj
from dmp.dmp_gentraj import GenerateTraj

#obstacles = np.array([[3,0],[3,0.25],[3,0.5],[3,-0.5],[3,-0.25],[3,-0.75],[3,-1],[3,-1.25],[3,-1.5]])
# obstacles = np.zeros([20,2])
# for j in range(20):
#     obstacles[j,0] = 4
#     obstacles[j,1] = -2 + 0.15*j
obstacles = np.array([[0.84,0.3,0]])

df_in = pd.read_table('test1_pos.txt', sep=',')
df_out = df_in
dd = df_out.values
y_des = dd[:,[0,1,2]].T

# y_des = np.zeros([3,66])
# for i in range(y_des.shape[1]):
#     y_des[0,i] = i*0.1
#     y_des[1,i] = 0
#     y_des[2,i] = 0
    


dmp = EncodeTraj(n_dmps=3, n_bfs=50)
avoidance_test = dmp.encode_trajectory(y_des=y_des, dmp_name='avoidance_test')

traj = GenerateTraj(avoidance_test, beta=10/np.pi, gamma=500, beta_dist=2)
#goal= [1.5,-0.5]
y0=y_des[:,0]
goal=y_des[:,-1]

y_track, dy_track, ddy_track = traj.rollout(y0=y0,goal=goal,obstacles=obstacles)


fig = plt.figure(1, figsize=(6, 6))
ax = fig.gca(projection='3d')

ax.plot(y_des[0, :], y_des[1, :], y_des[2,:], "b", lw=2)
ax.plot(y_track[:, 0], y_track[:, 1], y_track[:, 2], "r--", lw=2)
#plt.title("DMP system - draw number 2")

for obstacle in obstacles:
    ax.scatter(obstacle[0], obstacle[1], obstacle[2])

for i in range(y_track.shape[1]):
    df_out.iloc[:,i] = y_track[:,i]

df_out.to_csv('pick_pos2.txt', sep=',',index=False)
#ax.axis([-1,7,-4,4,-4,4])
# ax.xlim([-1, 7])
# ax.ylim([-4, 4])
# plt.show()
