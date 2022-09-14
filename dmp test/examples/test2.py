# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from dmp.dmp_encodetraj import EncodeTraj
from dmp.dmp_gentraj import GenerateTraj

# test imitation of path run
plt.figure(2, figsize=(6, 4))
#n_bfs = [10, 30, 50, 100, 10000]
n_bfs = 30

# a straight line to target
path1 = np.sin(np.arange(0, 1, 0.01) * 5)
# a strange path to target
path2 = np.zeros(path1.shape)
path2[int(len(path2) / 2.0) :] = 1


dmp = EncodeTraj(n_dmps=2, n_bfs=n_bfs)
    
y_des=np.array([path1, path2])

primitive = dmp.encode_trajectory(y_des=np.array([path1, path2]),dmp_name='test')

traj = GenerateTraj(primitive)

#goal = [3,1]
goal = y_des[:,-1]
y_track, dy_track, ddy_track = traj.rollout(goal=goal,tau=1.2)

traj.plot_traj()
