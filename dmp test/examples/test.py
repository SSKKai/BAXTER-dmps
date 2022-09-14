# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from dmp.dmp_encodetraj import EncodeTraj
from dmp.dmp_gentraj import GenerateTraj
from dmp.rl import RL_dmp

x = np.linspace(0,2,79)
y_des = np.array([x,x])




#y_des = np.load("2.npz")["arr_0"].T
#y_des = y_des[0,:]
#y_des -= y_des[:, 0][:, None]

#obstacles = np.array([[1,-0.5]])

# test normal run
#ay=np.ones(2) * 10.0
dd = EncodeTraj(n_dmps=2, n_bfs=20,ay=np.ones(2) * 10.0)
y_track = []
dy_track = []
ddy_track = []

draw_2 = dd.encode_trajectory(y_des=y_des, dmp_name='draw_2')
draw_22 = dd.encode_trajectory(y_des=y_des, dmp_name='draw_22')
traj = GenerateTraj(draw_2)
traj2 = GenerateTraj(draw_22)
#goal= [1.5,0]
#goal=y_des[:,-1]
y_track, dy_track, ddy_track = traj.rollout()
y_track2, dy_track2, ddy_track2 = traj2.rollout(y0=[0.33,-0.96],goal=[2,-3])


#rl = RL_dmp(draw_2)
#ddy_rl, dy_rl, y_rl, w_rl = rl.train()



plt.figure(1, figsize=(6, 6))

#plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2)
#plt.plot(y_track2[:, 0], y_track2[:, 1], "b", lw=2)
plt.title("DMP system - draw number 2")

plt.axis("equal")
plt.xlim([-4, 4])
plt.ylim([-4, 4])
# for obstacle in obstacles:
#     (plot_obs,) = plt.plot(obstacle[0], obstacle[1], "rx", mew=3)
# plt.show()

#traj.plot_traj()
y = np.zeros([149,2])
dy = np.zeros([149,2])
dy[0:70,:] = dy_track[0:70,:]
dy[70:79,:] = dy_track[70:79,:]+dy_track2[0:9,:]
dy[79:149,:] = dy_track2[9:79,:]

j=0
yy=[-0.5,0.372385]
for a in dy:
    yy = yy+a*0.01
    y[j] = yy
    j = j+1
plt.plot(y[:, 0], y[:, 1], "b", lw=2)