import numpy as np
import matplotlib.pyplot as plt
from dmp.dmp_encodetraj import EncodeTraj
from dmp.dmp_gentraj import GenerateTraj
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

df_in = pd.read_table('pick_pos2.txt', sep=',')
df_out = df_in
ddd = df_in.values
y_des = ddd[0:600,[3,4,5,6]].T

dd = EncodeTraj(n_dmps=4, n_bfs=2000,ay =np.ones(4)*200)

draw_2 = dd.encode_trajectory(y_des=y_des, dmp_name='draw_2')
traj = GenerateTraj(draw_2)
y_track, dy_track, ddy_track = traj.rollout(tau=1)
traj.plot_traj()