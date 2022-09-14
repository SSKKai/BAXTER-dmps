# -*- coding: utf-8 -*-
import numpy as np

from dmp.cs import CanonicalSystem


class EncodeTraj(object):
    """Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(
        self, n_dmps, n_bfs, dt=0.01, y0=0, goal=1, w=None, ay=None, by=None, **kwargs
    ):
        """
        n_dmps int: number of dynamic motor primitives
        n_bfs int: number of basis functions per DMP
        dt float: timestep for simulation
        y0 list: initial state of DMPs
        goal list: goal state of DMPs
        w list: tunable parameters, control amplitude of basis functions
        ay int: gain on attractor term y dynamics
        by int: gain on attractor term y dynamics
        """

        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dmps) * y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps) * goal
        self.goal = goal
        if w is None:
            # default is f = 0
            w = np.zeros((self.n_dmps, self.n_bfs))
        self.w = w

        self.ay = np.ones(n_dmps) * 25.0 if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4.0 if by is None else by  # Schaal 2012

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = int(self.cs.run_time / self.dt)
        
        self.gen_centers()
        
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.cs.ax
        
        self.primitives = {'Construct' : 'weights, psi_centers, psi_variance, ay, y_des(for plotting)'}

        # set up the DMP system
        self.reset_state()
        
        self.check_offset()

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if abs(self.y0[d] - self.goal[d]) < 1e-4:
                self.goal[d] += 1e-4

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        """x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]"""

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_goal(self, y_des):
        return np.copy(y_des[:, -1])

    def gen_psi(self,x):
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c) ** 2)

    def encode_trajectory(self, y_des, dmp_name):
        """Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.

        y_des list/array: the desired trajectories of each DMP
                          should be shaped [n_dmps, run_time]
        """

        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y0 = y_des[:, 0].copy()
        self.y_des = y_des.copy()
        self.goal = self.gen_goal(y_des)

        # self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate

        path = np.zeros((self.n_dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])

        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(self.timesteps):
                path[d, t] = path_gen(t * self.dt)
        y_des = path

        # calculate velocity of y_des with central differences
        dy_des = np.gradient(y_des, axis=1) / self.dt

        # calculate acceleration of y_des with central differences
        ddy_des = np.gradient(dy_des, axis=1) / self.dt

        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        # find the force required to move along this trajectory
        for d in range(self.n_dmps):
            f_target[:, d] = ddy_des[d] - self.ay[d] * (
                self.by[d] * (self.goal[d] - y_des[d]) - dy_des[d]
            )

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            k = self.goal[d] - self.y0[d]
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track ** 2 * psi_track[:, b])
                self.w[d, b] = numer / denom
                if abs(k) > 1e-5:
                    self.w[d, b] /= k

        self.w = np.nan_to_num(self.w)

        self.f_target = f_target
        self.reset_state()
        
        primitive = (self.w, self.c, self.h, self.ay, self.y_des)
        self.primitives[dmp_name] = primitive
        return primitive

    def plot_basisfunction(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(211)
        psi_track = self.gen_psi(self.cs.rollout())
        plt.plot(psi_track)
        plt.title("basis functions")

        #plot the desired forcing function vs approx
        for ii in range(self.n_dmps):
            plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
            plt.plot(self.f_target[:, ii], "--", label="f_target %i" % ii)
        for ii in range(self.n_dmps):
            plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
            print("w shape: ", self.w.shape)
            plt.plot(
                np.sum(psi_track * self.w[ii], axis=1) * self.dt,
                label="w*psi %i" % ii,
            )
            plt.legend()
        plt.title("DMP forcing function")
        plt.tight_layout()
        plt.show()
        
    def plot_bfs(self):
        import matplotlib.pyplot as plt

        plt.figure()
        psi_track = self.gen_psi(self.cs.rollout())
        self.psi_track = psi_track
        plt.plot(psi_track)
        plt.title("basis functions")
        plt.xlabel("time(s)")
        plt.ylabel("basis funtion value")
        x = range(0,101,20)
        plt.xticks(x,('0','1','2','3','4','5'))
        
    def reset_state(self):
        """Reset the system state"""
        #self.y = self.y0.copy()
        #self.dy = np.zeros(self.n_dmps)
        #self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()



# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test imitation of path run
    plt.figure(2, figsize=(6, 4))
    #n_bfs = [10, 30, 50, 100, 10000]
    n_bfs = 30

    # a straight line to target
    path1 = np.sin(np.arange(0, 1, 0.01) * 5)
    # a strange path to target
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.0) :] = 0.5


    dmp = EncodeTraj(n_dmps=2, n_bfs=n_bfs)
        
    #y_des=np.array([path1, path2])

    primitive = dmp.encode_trajectory(y_des=np.array([path2, path1]),dmp_name='test')
    
    dmp.plot_basisfunction()


