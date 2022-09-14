import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
#import dmp.point_obstacle
import math
from dmp.point_obstacle import obstacle



# To use the codes in the main folder
import sys
sys.path.insert(0, '../codes/')
sys.path.insert(0, 'codes/')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)

import pdb

from dmp import dmp_cartesian, obstacle_ellipse, point_obstacle

"""
Here we create the trajectory to learn
"""
t_f = 1 * np.pi # final time
t_steps = 10 ** 3 # time steps
t = np.linspace(0, t_f, t_steps)
a_x = 1. / np.pi
b_x = 1.
a_y = 1. / np.pi
b_y = 1.


x = a_x * t * np.cos(b_x*t+0.3)
y = a_y * t * np.sin(b_y*t)




x_des = np.ndarray([t_steps, 2])
x_des[:, 0] = x
x_des[:, 1] = y
x_des -= x_des[0]




# print(x_des[0])
# print(x_des[-1])

# Learning of the trajectory
dmp = dmp_cartesian.DMPs_cartesian(n_dmps=2, n_bfs=40, K = 1060 * np.ones(2), dt = .01, alpha_s = 3.)

dmp.imitate_path(x_des=x_des)

x_track, dx_track, ddx_track = dmp.rollout()
woxzq=x_track
dwoxzq=dx_track
ddwoxzq=ddx_track
x_classical = x_track
dx_classical = dx_track

"""
Volumetric Obstacle
"""
x_track_s = x_track[0]

x_c_1 = 0.09
y_c_1 = 0.3
n = 2
a_1 = .2
b_1 = .1


center_1 = np.array([x_c_1, y_c_1])
axis_1 = np.array([a_1, b_1])

A = 50.
eta = 1

fig = plt.figure(1)
plt.clf()
plt.figure(1, figsize=(6,6))
plt.plot(x_classical[:,0], x_classical[:, 1], '--c', lw=2, label = 'without obstacle')



"""
Point cloud obstacle
"""

dmp.reset_state()
x_track = np.zeros((1, dmp.n_dmps))
dx_track = np.zeros((1, dmp.n_dmps))
ddx_track = np.zeros((1, dmp.n_dmps))


dmp.dx_old = np.zeros(dmp.n_dmps)
dmp.ddx_old = np.zeros(dmp.n_dmps)
flag = False
dmp.t = 0
dmp.tol = 5e-02
# Obstacle definition
num_obst_1 = 50
t_1 = np.linspace(0., np.pi * 2., num_obst_1)


number=0
obst_list_1 = []
for n in range(num_obst_1):
    x_obst = np.array([x_c_1 + a_1*np.cos(t_1[n]), y_c_1 + b_1*np.sin(t_1[n])])
    obst = point_obstacle.obstacle(x_obst = x_obst, dx_obst = np.zeros(2))
    obst_list_1.append(obst)


while (not flag):
    if (dmp.t == 0):
        dmp.first = True
    else:
        dmp.first = False
    # run and record timestep
    F_1 = np.zeros([2])

    for n in range(num_obst_1):
        f_n = obst_list_1[n].gen_external_force(dmp.x, dmp.dx, dmp.goal)

        F_1 += f_n
    F = F_1

    # print(dmp.x)
    # print(F_2)
    # print(F)
    # print(dmp.x)
    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force=F)
    x_track = np.append(x_track, [x_track_s], axis=0)
    # print(dmp.x)


    dx_track = np.append(dx_track, [dx_track_s],axis=0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis=0)
    dmp.t += 1
    flag = (dmp.t >= dmp.cs.timesteps) & (np.linalg.norm(x_track_s - dmp.goal) / np.linalg.norm(dmp.goal - dmp.x0) <= dmp.tol)


plt.plot(x_track[:,0], x_track[:,1], color = 'orange', linestyle = '-.', lw=2, label = 'with obstacle')

c=x_track[:,0]
d=x_track[:,1]




x_plot_1 = x_c_1 + a_1*np.cos(t_1)
y_plot_1 = y_c_1 + b_1 * np.sin(t_1)
plt.plot (x_plot_1, y_plot_1, ':b', lw=2, label = 'obstacle')
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
plt.axis('equal')
plt.text(dmp.x0[0]-0.05, dmp.x0[1]-0.05, r'$\mathbf{x}_0$', fontsize = 16)
plt.text(dmp.goal[0]+0.01, dmp.goal[1]-0.05, r'$\mathbf{g}$', fontsize = 16)

plt.show()