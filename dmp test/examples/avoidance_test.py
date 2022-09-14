# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from dmp.dmp_encodetraj import EncodeTraj
from dmp.dmp_gentraj import GenerateTraj

#obstacles = np.array([[3,0],[3,0.25],[3,0.5],[3,-0.5],[3,-0.25],[3,-0.75],[3,-1],[3,-1.25],[3,-1.5]])
#obstacles = np.array([[3,0],[2.75,-0.25],[2.5,0.25],[3.5,-0.5],[3.25,0.25],[4,-0.75],[3.3,1.7],[2.75,-0.25],[3,1.7]])
obstacles = np.array([[3,0]])
# obstacles = np.zeros([20,2])
# for j in range(20):
#     obstacles[j,0] = 4
#     obstacles[j,1] = -2 + 0.15*j
#obstacles = np.array([[3,0]])


y_des = np.zeros([2,66])
for i in range(y_des.shape[1]):
    y_des[0,i] = i*0.1
    y_des[1,i] = 0
    


dmp = EncodeTraj(n_dmps=2, n_bfs=50)
avoidance_test = dmp.encode_trajectory(y_des=y_des, dmp_name='avoidance_test')

traj = GenerateTraj(avoidance_test, beta=10/np.pi, gamma=1000, beta_dist=2) #10 1000 / 40 1000
#goal= [1.5,-0.5]
y0=y_des[:,0]
goal=y_des[:,-1]

y_track, dy_track, ddy_track = traj.rollout(y0=y0,goal=goal,obstacles=obstacles)


plt.figure(1, figsize=(6, 4))

plt.plot(y_des[0, :], y_des[1, :], color='black', lw=1)
plt.plot(y_track[:, 0], y_track[:, 1], "r--", lw=2)
plt.legend(['orignal path','avoidance path'])
#plt.title("DMP system - draw number 2")

for obstacle in obstacles:
    (plot_obs,) = plt.plot(obstacle[0], obstacle[1], "bx",mew=3)

plt.axis("equal")
plt.xlim([-1, 7])
plt.ylim(0,2)
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.plot(6.5,0,'bo',mew=3)
plt.show()

# beta=20/np.pi
# phi = np.linspace(-180,180,1000)/100
# y = np.zeros(1000)
# dphi = 1000 * phi * np.exp(-beta * abs(phi))
# plt.figure()
# plt.plot(phi,dphi)
# plt.plot(phi,y)
# plt.xlabel('phi')
# plt.ylabel('dphi')

#%%
plt.figure(1, figsize=(6, 2))

plt.plot(y_des[0, :], y_des[1, :], color='black', lw=1)
plt.plot(y1[:, 0], y1[:, 1], "--", lw=2)
plt.plot(y2[:, 0], y2[:, 1], "--", lw=2)
plt.plot(y3[:, 0], y3[:, 1], "--", lw=2)
plt.legend(['orignal path','beta_d=0','beta_d=1','beta_d=2'])
for obstacle in obstacles:
    (plot_obs,) = plt.plot(obstacle[0], obstacle[1], "bx",mew=3)

plt.axis("equal")
plt.xlim([-1, 7])
plt.ylim(0,2)
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.plot(6.5,0,'bo',mew=3)
plt.show()