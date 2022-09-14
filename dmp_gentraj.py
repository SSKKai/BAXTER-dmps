# -*- coding: utf-8 -*-
import numpy as np

from cs import CanonicalSystem


class GenerateTraj(object):
    """Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(
        self, primitive, dt=0.01, y0=0, goal=1, by=None, **kwargs
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

        self.w = primitive[0]
        self.c = primitive[1]
        self.h = primitive[2]
        self.n_dmps = int(self.w.shape[0])
        self.n_bfs = int(self.w.shape[1])
        self.dt = dt
        
        self.y_trained = primitive[4]


        self.ay = primitive[3]  # Schaal 2012
        self.by = self.ay / 4.0 if by is None else by  # Schaal 2012

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        
        self.timesteps = int(self.cs.run_time / self.dt)
        #self.timesteps = self.y_trained.shape[1]


    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if abs(self.y0[d] - self.goal[d]) < 1e-4:
                self.goal[d] += 1e-4
    
    def gen_front_term(self, x, dmp_num):
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_psi(self,x):
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c) ** 2)



    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()
    
    def rollout(self,y0=0, goal=1, timesteps=None, tau = 1, **kwargs):
        """Generate a system trial, no feedback is incorporated."""
        
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dmps) * y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps) * goal
        self.goal = goal
        
        self.reset_state()
        self.check_offset()
        
        tau = (float(self.timesteps) / self.y_trained.shape[1]) * tau
        
        if timesteps is None:
            timesteps = int(self.timesteps / tau)

        # if timesteps is None:
        #     if "tau" in kwargs:
        #         timesteps = int(self.timesteps / (tau*kwargs["tau"]))
        #     else:
        #         timesteps = self.timesteps / tau

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):

            # run and record timestep
            y_track[t], dy_track[t], ddy_track[t] = self.step(tau = tau, **kwargs)

        self.y_track = y_track
        return y_track, dy_track, ddy_track


    def step(self, tau=1.0, error=0.0, external_force=None):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        """

        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # generate basis function activation
        psi = self.gen_psi(x)



        for d in range(self.n_dmps):

            # generate the forcing term
            f = self.gen_front_term(x, d) * (np.dot(psi, self.w[d])) / np.sum(psi)


            # DMP acceleration
            self.ddy[d] = (
                self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + f)
            
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * tau * self.dt * error_coupling

        return self.y, self.dy, self.ddy
    
    def plot_traj(self):
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(self.n_dmps):
            plt.subplot(self.n_dmps,1,i+1)
            plt.plot(self.y_track[:, i], lw=2)
            
            plt.plot(self.y_trained[i], "r--", lw=2)
            plt.title("DMP imitate path")
            plt.xlabel("time (ms)")
            plt.ylabel("system trajectory")
            
        plt.tight_layout()
        plt.show()
            

# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dmp_encodetraj import EncodeTraj

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
    
    goal = [3,1]
    y_track, dy_track, ddy_track = traj.rollout(goal=goal)
    
    traj.plot_traj()
            
