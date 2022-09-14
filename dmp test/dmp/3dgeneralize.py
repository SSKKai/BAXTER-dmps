# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from dmp.dmp_encodetraj import EncodeTraj
from dmp.dmp_gentraj import GenerateTraj
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

df_in = pd.read_table('test1_pos.txt', sep=',')
df_out = df_in
dd = df_in.values
y_des = dd[:,[0,1,2]].T

# a, b = (2, 0)
# r = 2.0
# x = np.arange(a-r, a+r, 0.01)
# y = np.arange(a-r, a+r, 0.01)
# z = b + np.sqrt(r**2 - (x - a)**2)
# y_des = np.array([x,y,z])
# y_des = y_des[:,50:350]

dd = EncodeTraj(n_dmps=3, n_bfs=2000)

draw_2 = dd.encode_trajectory(y_des=y_des, dmp_name='draw_2')
traj = GenerateTraj(draw_2)

goal= [ 0.73138157, -0.03538419,  0.17863395]
#goal1=y_des[:,-1]
goal1 = [ 0.73138157, -0.23538419,  0.17863395]
goal2 = [ 0.73138157, 0.23538419,  0.17863395]
goal3 = [ 0.73138157, 0.23538419,  0.16863395]
goal4 = [ 0.73138157, -0.23538419,  0.16863395]
goal5= [ 0.73138157, -0.03538419,  0.16863395]

goal_w = [4, 4, 1.5]
y0_w = [1,1,1]


y_track, dy_track, ddy_track = traj.rollout(goal=goal_w,tau=1)
#traj.reset_state()
y_track1, dy_track1, ddy_track1 = traj.rollout(goal=goal1,tau=1)
y_track2, dy_track2, ddy_track2 = traj.rollout(goal=goal2,tau=1)
y_track3, dy_track3, ddy_track3 = traj.rollout(goal=goal3,tau=1)
y_track4, dy_track3, ddy_track3 = traj.rollout(goal=goal4,tau=1)
y_track5, dy_track3, ddy_track3 = traj.rollout(goal=goal5,tau=1)

#traj.plot_traj()

fig = plt.figure(1, figsize=(6, 6))
ax = fig.gca(projection='3d')

ax.plot(y_des[0, :], y_des[1, :], y_des[2,:], "r--", lw=2)
#ax.plot(y_track[:, 0], y_track[:, 1], y_track[:,2], "b", lw=2)
ax.plot(y_track1[:, 0], y_track1[:, 1], y_track1[:,2], lw=2)
ax.legend(['demonstrated','generalized'],loc='right')
ax.plot(y_track2[:, 0], y_track2[:, 1], y_track2[:,2], lw=2)
ax.plot(y_track3[:, 0], y_track3[:, 1], y_track3[:,2], lw=2)
ax.plot(y_track4[:, 0], y_track4[:, 1], y_track4[:,2], lw=2)
ax.plot(y_track5[:, 0], y_track5[:, 1], y_track5[:,2], lw=2)
ax.plot([ 0.73138157, 0.73138157],[ -0.23538419, 0.23538419],[ 0.17863395, 0.17863395],"b--", lw=1)
ax.plot([ 0.73138157, 0.73138157],[ -0.23538419, 0.23538419],[ 0.16863395, 0.16863395],"b--", lw=1)
ax.plot([ 0.73138157, 0.73138157],[ -0.23538419, -0.23538419],[ 0.16863395, 0.17863395],"b--", lw=1)
ax.plot([ 0.73138157, 0.73138157],[ 0.23538419, 0.23538419],[ 0.16863395, 0.17863395],"b--", lw=1)

