# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from dmp.dmp_encodetraj import EncodeTraj
from dmp.dmp_gentraj import GenerateTraj
from dmp.rl import RL_dmp

y_des = np.load("2.npz")["arr_0"].T
y_track = y_des.T

dd = EncodeTraj(n_dmps=2, n_bfs=100)
# y_track = []
# dy_track = []
# ddy_track = []

draw_2 = dd.encode_trajectory(y_des=y_des, dmp_name='draw_2')

traj = GenerateTraj(draw_2)

#goal= [0.4,-0.5]
goal=y_des[:,-1]
y0=y_des[:,0]

y_t, dy_t, ddy_t = traj.rollout(y0=y0,goal=goal,tau=1)

s = traj.x_track
w = dd.w.T
psi = traj.psi_track.T
t = np.array(range(0,traj.timesteps))

x_r, dx_r, ddx_r = np.zeros((3, y_track.shape[0], y_track.shape[1]))
w_a = np.zeros((dd.n_bfs, dd.n_dmps))
gain = []

my_rldmp = RL_dmp(dd.ay[0],dd.n_bfs,False)

 
for i in range(y_track.shape[1]):
    ddx_r[:, i], dx_r[:, i], x_r[:, i], w_a[:, i], g = my_rldmp.adapt(w[:, i], 
            y_track[0, i], y_track[-1,i], t, s, psi, 10, 0.5)
    gain.append(g)


import matplotlib.pyplot as plt
plt.figure()
for i in range(traj.n_dmps):
    plt.subplot(traj.n_dmps,1,i+1)
    plt.plot(x_r[:, i],"r--", lw=2)
    plt.plot(y_t[:, i],"g--", lw=2)
    plt.plot(y_des[i], lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    
plt.tight_layout()
plt.show()
