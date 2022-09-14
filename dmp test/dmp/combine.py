# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from dmp.dmp_encodetraj import EncodeTraj
from dmp.dmp_gentraj import GenerateTraj
import pandas as pd
#from utils import *

df_in = pd.read_table('data/demo3.txt', sep=',')
df_out = df_in
dd = df_in.values
#y_des = dd[1150:1350,[1,2,3,4,5,6,7]].T

y_des1 = dd[:,[2]].T
y_des2 = dd[:,[4]].T

# data = load_demo('data/demo3.txt')
# t, q = parse_demo(data)

# t = normalize_vector(t)

# # Compute velocity and acceleration
# dq, ddq = np.zeros((2, q.shape[0], q.shape[1]))

# for i in range(q.shape[1]):
#     q[:, i] = smooth_trajectory(q[:, i], 5)
#     dq[:, i] = vel(q[:, i], t)
#     ddq[:, i] = vel(dq[:, i], t)

# # Filter the position velocity and acceleration signals
# f_q, f_dq, f_ddq = np.zeros((3, q.shape[0], q.shape[1]))

# for i in range(q.shape[1]):
#     f_q[:, i] = blend_trajectory(q[:, i], dq[:, i], t, 50)
#     f_dq[:, i] = vel(f_q[:, i], t)
#     f_ddq[:, i] = vel(f_dq[:, i], t)

# y_des = q.T

dd = EncodeTraj(n_dmps=1, n_bfs=2000,ay=np.ones(1)*55)
#dd2 = EncodeTraj(n_dmps=7, n_bfs=2000,ay=np.ones(7)*55)

draw_1 = dd.encode_trajectory(y_des=y_des1, dmp_name='draw_1')
draw_2 = dd.encode_trajectory(y_des=y_des2, dmp_name='draw_2')
traj1 = GenerateTraj(draw_1)
traj2 = GenerateTraj(draw_2)

goal1= [0.9]
goal2=[-1.2]
y1=[0.8]
y2=[-0.82485937]
y_track1, dy_track1, ddy_track1 = traj1.rollout()
y_track2, dy_track2, ddy_track2 = traj2.rollout(y0=y2,goal=goal2,tau=1)
#tt_track, dtt_track, ddtt_track = traj_tt.rollout()

traj1.plot_traj()
traj2.plot_traj()


n = 100
m = 1747
y = np.zeros([m*2-n,1])
dy = np.zeros([m*2-n,1])
dy[0:m-n] = dy_track1[0:m-n]
dy[m-n:m] = (dy_track1[m-n:m]+dy_track2[0:n])/2
dy[m:2*m-n] = dy_track2[n:m]

j=0
yy=-0.81375734
for a in dy:
    yy = yy+a*0.01*0.057240984544934176
    y[j] = yy
    j = j+1
plt.figure()
plt.plot(y, "b", lw=2)
# xx2 = range(200-n,400-n)
# xx1 = range(0,200)
# plt.plot(xx1,y_track1)
# plt.plot(xx2,y_track2)
plt.figure()
plt.plot(dy)
plt.figure()
plt.plot(dy_track1)
plt.figure()
plt.plot(dy_track2)