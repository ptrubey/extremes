import numpy as np
from genpareto import gpd_fit
from numpy.linalg import norm

class Data(object):
    @staticmethod
    def to_euclidean(theta):
        """ casts angles in radians onto unit hypersphere in Euclidean space """
        coss = np.vstack((np.cos(theta).T, 1)).T
        sins = np.vstack((1, np.sin(theta).T)).T
        sinp = np.cumprod(sins, axis = 1)
        return coss * sinp

    def fill_out(self):
        self.coss  = np.vstack((np.cos(self.A).T, np.ones(self.A.shape[0]))).T
        self.sins  = np.vstack((np.ones(self.A.shape[0]), np.sin(self.A).T)).T
        self.sinp  = np.cumprod(self.sins, axis = 1)
        self.Yl    = self.coss * self.sinp
        self.lsins = np.log(self.sins)
        self.lcoss = np.log(self.coss)
        return

    def __init__(self, path):
        self.A = read.csv(path)
        self.fill_out()
        return

class Data_From_Raw(Data):
    raw = None # raw data
    Z   = None # Standardized Pareto Transformed (for those > 1)
    P   = None # Generalized Pareto Parameters (threshold, scale, extreme index)
    V   = None # Z cast to unit Hypersphere
    R   = None # row maximum under standardized Pareto
    I   = None # index of observation in raw corresponding to observation in V
               # (because we're subsetting to only observations w/ max > 1)
    A   = None # Angular Data

    @staticmethod
    def to_angular(hyp):
        """ Convert data to angular representation. """
        n, k  = hyp.shape
        theta = np.empty((n, k - 1))
        for i in range(k - 1):
            theta[:,i] = np.arccos(hyp[:,i] / (norm(hyp[:,i:], axis = 1) + 1e-7))
        return theta

    @staticmethod
    def to_hypercube(par):
        """ Projects data that is marginally standardized Pareto (for those
        obsv for which the row max > 1) onto the unit hypercube. returns those
        projections, the row max, and the indices in the original data
        corresponding to the observations """
        R = par.max(axis = 1)
        V = (par.T / R).T
        I = np.where(R > 1)
        return V[I], R[I], I

    @staticmethod
    def to_pareto(raw, q = 0.95):
        """ convert data to marginal std pareto -- q is the threshold quantile
        returns an array of observations which > 1 follows standardized pareto,
        as well as an array of GP parameters (univariate threshold, scale, xi)
        """
        def compute_gp_parameters(raw_vector, q):
            b = np.quantile(raw_vector, q)
            a, xi = gpd_fit(raw_vector, b)
            return np.array((b,a,xi))

        P = np.apply_along_axis(lambda x: compute_gp_parameters(x, q), 0, raw)
        Z = (1 + P[2] * (raw - P[0]) / P[1])**(1/P[2])
        Z[Z < 0.] = 0.
        return Z, P

    def __init__(self, raw):
        # if input is pandas dataframe, then take numpy array representation
        try:
            self.raw = raw.values
        # else assume input is numpy array
        except AttributeError:
            self.raw = raw
        # Compute standardized pareto margins
        self.Z, self.P = self.to_pareto(self.raw)
        # Cast to hypercube, keep only observations extreme in >= 1 dimension
        self.V, self.R, self.I = self.to_hypercube(self.Z)
        # proceed with angular representation
        self.A = self.to_angular(self.V)
        # Number of columns for Gamma representation
        self.nCol = self.A.shape[1] + 1
        # Number of rows in data
        self.nDat = self.A.shape[0]
        # Pre-compute the trig components of the likelihood.
        self.fill_out()
        return

# EOF
