
import numpy as np

class Obstacle_Ellipse():

    """
    Implementation of an obstacle for DMPs written as a
    general n-ellipsoid
      / x - x_c \ 2n     / y - y_c \ 2n      / z - z_c \ 2n
      |---------|     +  |---------|     +   |---------|     =  1
      \    a    /        \    b    /         \    c    /
    """

    def __init__(self, n_dim = 2, n = 1, center = np.zeros(2), axis = np.ones(2),
            **kwargs):

        if ((np.shape(center)[0] != n_dim) or (np.shape(axis)[0] != n_dim)):
            raise ValueError ("The dimensions of center or axis are not compatible with n_dim")
        else:
            self.n_dim = n_dim
            self.n = n
            self.center = center
            self.axis = axis
        return



    def compute_forcing_term(self, x, A = 1., eta = 1.):

        phi = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            phi[i] = (((x[i] - self.center[i]) ** (2 * self.n - 1)) /
                (self.axis[i] ** (2 * self.n)))
        K = self.compute_isopotential(x)

        phi *= (A * np.exp(-eta*K)) * (eta / K + 1. / K / K) * (2 * self.n)
        return phi

    def compute_potential(self, x, A = 1., eta = 1.):

        K = self.compute_isopotential (x)
        U = A * np.exp(-eta * K) / K
        return U

    def compute_isopotential(self, x):

        K = 0.
        for i in range(self.n_dim):
            K += ((x[i] - self.center[i]) / self.axis[i]) ** (2 * self.n)
        K -= 1
        return K
