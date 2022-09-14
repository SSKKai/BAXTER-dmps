# -*- coding: utf-8 -*-
import numpy as np

from dmp.cs import CanonicalSystem


class GenerateTraj(object):
    """Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(
        self, primitive, dt=0.01, by=None, beta=20.0/np.pi, gamma=1000, beta_dist=2, **kwargs
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
        
        #parameters for obstacle avoidance
        self.beta = beta
        self.gamma = gamma
        self.beta_dist = beta_dist
        self.R_halfpi = np.array(
            [
                [np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)],
            ]
        )
        
    

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
    
    def avoid_obstacles(self):
        p = np.zeros(2)
    
        for obstacle in self.obstacles:
            # based on (Hoffmann, 2009)
    
            # if we're moving
            if np.linalg.norm(self.dy) > 1e-5:
    
                # get the angle we're heading in
                phi_dy = -np.arctan2(self.dy[1], self.dy[0])
                R_dy = np.array(
                    [[np.cos(phi_dy), -np.sin(phi_dy)], [np.sin(phi_dy), np.cos(phi_dy)]]
                )
                # calculate vector to object relative to body
                obj_vec = obstacle - self.y
                # rotate it by the direction we're going
                obj_vec = np.dot(R_dy, obj_vec)
                # calculate the angle of obj relative to the direction we're going
                phi = np.arctan2(obj_vec[1], obj_vec[0])
    
                dphi = self.gamma * phi * np.exp(-self.beta * abs(phi))
                print(dphi)
                R = np.dot(self.R_halfpi, np.outer(obstacle - self.y, self.dy))
                pval = -np.nan_to_num(np.dot(R, self.dy) * dphi)
    
                # check to see if the distance to the obstacle is further than
                # the distance to the target, if it is, ignore the obstacle
                if np.linalg.norm(obj_vec) > np.linalg.norm(self.goal - self.y):
                    pval = 0
    
                p += pval
        return p
    
    def rotation_matrix(self, theta, u):
        """
        Compute the roation matrix of a rotation of theta around the direction
        given by u (if self.n_dim = 3)
        """
        c = np.cos(theta)
        s = np.sin(theta)
        if (self.n_dmps == 2):
            R = np.array([[c, -s], [s, c]])
        elif (self.n_dmps == 3):
            if (np.shape(u)[0] != self.n_dmps):
                raise ValueError ('dimension of u incompatible with self.n_dim')
            # u has to be a versor in the l2-norm
            u = u / np.linalg.norm(u)
            x = u[0]
            y = u[1]
            z = u[2]
            C = 1. - c
            R = np.array([
                [(x * x * C + c), (x * y * C - z * s), (x * z * C + y * s)],
                [(y * x * C + z * s), (y * y * C + c), (y * z * C - x * s)],
                [(z * x * C - y * s), (z * y * C + x * s), (z * z * C + c)]])
        else:
            raise ValueError ('Invalid dimension')
        return R
    
    def point_obstacles(self):
        pval = np.zeros(self.n_dmps)
        p = np.zeros(self.n_dmps)
        
        for obstacle in self.obstacles:
            if np.linalg.norm(self.dy) > 1e-5:
            # Computing the steering angle phi_i

                pos = obstacle - self.y    # o_i - x
        
                # Rotate it by the die rection we'rgoing
                # print(self.dx_obst)
                vel = self.dy      # v - \dot o_i
                # print(vel)
                dist = np.linalg.norm(pos)
                # Calculate the steering angle
                phi = np.arccos(np.dot(pos, vel) /
                                        (np.linalg.norm(pos) * np.linalg.norm(vel)))
        
                dphi = self.gamma * phi * np.exp(-self.beta * phi-self.beta_dist*dist)
        
                r = np.cross(pos, self.dy)
        
                R = self.rotation_matrix(np.pi/2, r)
        
                #pval = 3.8*np.dot(R, vel) * dphi
                pval = 0.665*np.dot(R, vel) * dphi/dist
        
                if not (np.abs(phi) < np.pi / 2.0):
                    pval *= 0.0
                    
                p += pval

        return p
    
    def rollout(self,y0=None, goal=None, timesteps=None, tau = 1, obstacles=None, **kwargs):
        """Generate a system trial, no feedback is incorporated."""
        
        # if isinstance(y0, (int, float)):
        #     y0 = np.ones(self.n_dmps) * y0
        # self.y0 = y0
        # if isinstance(goal, (int, float)):
        #     goal = np.ones(self.n_dmps) * goal
        # self.goal = goal
        
        if y0 is None:
            self.y0 = self.y_trained[:,0]
        else:
            if isinstance(y0, (int, float)):
                y0 = np.ones(self.n_dmps) * y0
            self.y0 = y0
        if goal is None:
            self.goal = self.y_trained[:,-1]
        else:
            if isinstance(goal, (int, float)):
                goal = np.ones(self.n_dmps) * goal
            self.goal = goal
        
        self.reset_state()
        self.check_offset()
        
        tau = (float(self.timesteps) / self.y_trained.shape[1]) * tau
        print(tau)
        if timesteps is None:
            timesteps = int(self.timesteps / tau)
        self.timesteps = timesteps
        
        if obstacles is not None:
            self.obstacles = obstacles
            external_force = True
        else:
            external_force = False

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))
        offset_value = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):

            # run and record timestep
            y_track[t], dy_track[t], ddy_track[t], offset_value[t] = self.step(tau = tau, external_force=external_force, **kwargs)

        self.y_track = y_track
        self.offset_value = offset_value
        
        self.x_track = self.cs.rollout(tau=tau)
        self.psi_track = self.gen_psi(self.x_track)
        
        return y_track, dy_track, ddy_track


    def step(self, tau=1.0, error=0.0, external_force=False):
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
        
        if external_force is True:
            #avoidance_offset = self.avoid_obstacles()
            avoidance_offset = self.point_obstacles()
        else:
            avoidance_offset = np.zeros(self.n_dmps)



        for d in range(self.n_dmps):

            # generate the forcing term
            f = self.gen_front_term(x, d) * (np.dot(psi, self.w[d])) / np.sum(psi)


            # DMP acceleration
            self.ddy[d] = (
                self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + f)
            

            self.ddy[d] += avoidance_offset[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * tau * self.dt * error_coupling

        return self.y, self.dy, self.ddy, avoidance_offset
       
    def plot_traj(self):
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(self.n_dmps):
            plt.subplot(self.n_dmps,1,i+1)
            plt.plot(self.y_track[:, i], lw=2)
            
            plt.plot(self.y_trained[i], "r--", lw=2)
            plt.title("DMP imitate path")
            plt.xlabel("time (ms)")
            #plt.ylim(-0.1,0.1)
            
        plt.tight_layout()
        plt.legend(['imitated path','demonstrated path'],loc = 'upper right')
        plt.show()
        
    
    def adapt(self, w, x0, g, t, s, psv, samples, rate):

        print('Trajectory adapted')

        # Initialize the action variables
        a = w
        tau = t[-1]

        # Flag which acts as a stop condition
        met_threshold = False
        counter = 0
        gain = []

        while not met_threshold:
            exploration = np.array([[np.random.normal(0, np.std(psv[j]*a[j]))
                    for j in range(self.ng)] for i in range(samples)])

            actions = np.array([a + e for e in exploration])

            # Generate new rollouts
            ddx, dx, x = np.transpose([self.generate(act, x0, g, t, s, psv) for act in actions], (1, 2, 0))

            # Estimate the Q values
            Q = [sum([self.reward(g, x[j, i], t[j], tau) for j in range(len(t))]) for i in range(samples)]

            # Sample the highest Q values to adapt the action parameters
            sort_Q = np.argsort(Q)[::-1][:np.floor(samples*rate).astype(int)]

            # Update the action parameter
            sumQ_y = sum([Q[i] for i in sort_Q])
            sumQ_x = sum([exploration[i]*Q[i] for i in sort_Q])

            # Update the policy parameters
            a += sumQ_x/sumQ_y

            gain.append(Q[sort_Q[0]])

            # Stopping condition
            if np.abs(x[-1, sort_Q[0]] - g) < 0.01:
                met_threshold = True

        return ddx[:, sort_Q[0]], dx[:, sort_Q[0]], x[:, sort_Q[0]], actions[sort_Q[0]], np.cumsum(gain)

    # Reward function
    def reward(self, goal, position, time, tau, w=0.9, threshold=5):

        dist = goal - position

        if np.abs(time - tau) < threshold:
            rwd = w*np.exp(-np.sqrt(dist*dist.T))
        else:
            rwd = (1-w) * np.exp(-np.sqrt(dist*dist.T))/tau

        return rwd
            

# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dmp.dmp_encodetraj import EncodeTraj

    # test imitation of path run
    #plt.figure(2, figsize=(6, 4))
    #n_bfs = [10, 30, 50, 100, 10000]
    n_bfs = 500

    # a straight line to target
    path1 = np.sin(np.arange(0, 1, 0.01) * 5)
    # a strange path to target
    #path2 = np.zeros(path1.shape)
    #path2[int(len(path2) / 2.0) :] = 1


    dmp = EncodeTraj(n_dmps=1, n_bfs=n_bfs)
        
    y_des=np.array(path1)

    primitive = dmp.encode_trajectory(y_des=y_des,dmp_name='test')
    
    traj = GenerateTraj(primitive)
    
    goal1 = y_des[-1]
    goal2 = [-1.5]
    goal3 = [-0.5]
    goal4 = [-0.75]
    goal5 = [-1.25]
 
#%%    
    y_track, dy_track, ddy_track = traj.rollout(goal=goal1)
    y_track1, dy_track1, ddy_track1 = traj.rollout(goal=goal2)
    y_track2, dy_track2, ddy_track2 = traj.rollout(goal=goal3)
    y_track3, dy_track3, ddy_track3 = traj.rollout(goal=goal4)
    y_track4, dy_track4, ddy_track4 = traj.rollout(goal=goal5)
    #traj.plot_traj()
    
    plt.figure(figsize=(6, 3))
    plt.plot(path1,'black',lw=2)
    plt.plot(y_track,'--',lw=1)
    plt.xlabel("time")
    plt.ylabel("system trajectory")
    plt.xticks([])
    plt.legend(['demonstration','generalized'])  
    plt.plot(y_track1,'--',lw=1)
    plt.plot(y_track2,'--',lw=1)
    plt.plot(y_track3,'--',lw=1)
    plt.plot(y_track4,'--',lw=1)
#%%
    y_track5, dy_track5, ddy_track5 = traj.rollout(goal=goal1,tau=0.6)
    y_track6, dy_track6, ddy_track6 = traj.rollout(goal=goal1,tau=0.8)
    y_track7, dy_track7, ddy_track7 = traj.rollout(goal=goal1,tau=1.2)
    y_track8, dy_track8, ddy_track8 = traj.rollout(goal=goal1,tau=1.4)
    plt.figure(figsize=(6, 3))
    plt.plot(path1,'black',lw=2)
    plt.plot(y_track5,'--',lw=1) 
    plt.xlabel("time")
    plt.ylabel("system trajectory")
    plt.xticks([])
    plt.legend(['demonstration','generalized']) 
    plt.plot(y_track6,'--',lw=1)
    plt.plot(y_track7,'--',lw=1)
    plt.plot(y_track8,'--',lw=1)
    