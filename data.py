import numpy as np
import sqlite3 as sql
import pandas as pd
import os
from genpareto import gpd_fit
from numpy.linalg import norm
from math import pi, sqrt, exp
from scipy.special import erf, erfinv

epsilon = 1e-30

def angular_to_hypercube(theta):
    """ Assuming data in polar, casts to euclidean then divides by row max
    to achieve projection onto hypercube. """
    euc = angular_to_euclidean(theta)
    return euclidean_to_hypercube(euc)

def angular_to_simplex(theta):
    euc = angular_to_euclidean(theta)
    return euclidean_to_simplex(euc)

def euclidean_to_simplex(euc):
    return ((euc + epsilon).T / (euc + epsilon).sum(axis = 1)).T

def euclidean_to_hypercube(euc):
    return ((euc + epsilon).T / (euc + epsilon).max(axis = 1)).T

def angular_to_euclidean(theta):
    """ casts angles in radians onto unit hypersphere in Euclidean space """
    coss = np.hstack((np.cos(theta), np.ones(shape = (theta.shape[0], 1))))
    sins = np.hstack((np.ones(shape = (theta.shape[0], 1)), np.sin(theta)))
    sinp = np.cumprod(sins, axis = 1)
    return coss * sinp

def euclidean_to_angular(hyp):
    """ Convert data to angular representation. """
    n, k  = hyp.shape
    theta = np.empty((n, k - 1))
    for i in range(k - 1):
        # establish that the denominator is always greater
        # (even if by *tiny* amount) than numerator
        # temp = np.sqrt((hyp[:,i:] * hyp[:,i:]).sum(axis = 1))
        temp = hyp[:,i] / norm(hyp[:,i:], axis = 1)
        temp[temp > (1 - 1e-7)] = 1 - 1e-7
        # tdiff = temp - hyp[:,i]
        # temp[np.where(tdiff < 1e-7)[0]] = 1e-7
        # then theta is arccos of that ratio
        theta[:,i] = np.arccos(temp)
    return theta

def cluster_max_row_ids(series):
    nDat = series.shape[0]
    lst = []
    clu = np.empty(0, dtype = int)
    for i in range(nDat):
        if series[i] > 1:
            clu = np.append(clu, i)
        else:
            if clu.shape[0] > 0:
                lst.append(clu)
                clu = np.empty(0, dtype = int)
            else:
                pass
    else:
        if clu.shape[0] > 0:
            lst.append(clu)
    max_ids = np.empty(0, dtype = int)
    for cluster in lst:
        max_ids = np.append(max_ids, cluster[np.argmax(series[cluster])])
    return max_ids

def scale_pareto(raw, P):
    """ Do the actual Pareto scaling """
    Z = (1 + P[2] * (raw - P[0]) / P[1])**(1/P[2])
    Z[Z < 0.] = 0.
    return Z

def descale_pareto(Z, P):
    """ Given Pareto scaled RV's, return original scale """
    raw = P[0] + P[1] * (Z**P[2] - 1) / P[2]
    return raw

class Transformer(object):
    @staticmethod
    def probit(x):
        return sqrt(2.) * erfinv(2 * x - 1)

    @staticmethod
    def invprobit(y):
        return 0.5 * (1 + erf(y / sqrt(2.)))

    @staticmethod
    def invprobitlogjac(y):
        return (- 0.5 * np.log(2 * pi) - y * y / 2.).sum()

class Data(Transformer):
    def write_empirical(self, path):
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.mkdir(folder)
        ncol   = self.A.shape[1] + 1
        thetas = pd.DataFrame(
            self.A,
            columns = ['theta_{}'.format(i) for i in range(1, ncol)],
            )
        thetas.to_csv(path, index = False)
        return

    @staticmethod
    def to_euclidean(theta):
        return to_euclidean(theta)

    @staticmethod
    def cast_to_cube(A, eps = 1e-6):
        V = A / (pi / 2.)
        V[V > (1 - eps)] = 1 - eps
        V[V < eps] = eps
        return V

    def fill_out(self):
        self.coss  = np.vstack((np.cos(self.A).T, np.ones(self.A.shape[0]))).T
        self.sins  = np.vstack((np.ones(self.A.shape[0]), np.sin(self.A).T)).T
        self.sinp  = np.cumprod(self.sins, axis = 1)
        self.Yl    = self.coss * self.sinp
        self.lsins = np.log(self.sins)
        self.lcoss = np.log(self.coss)
        self.S     = angular_to_simplex(self.A)
        self.Vi    = self.cast_to_cube(self.A)
        self.pVi   = self.probit(self.Vi)
        return

    def __init__(self, path):
        self.A = pd.read_csv(path).values
        self.V = angular_to_hypercube(self.A)
        self.nDat, self.nCol = self.V.shape
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
        return euclidean_to_angular(hyp)

    @staticmethod
    def to_hypercube(par, decluster):
        """ Projects data that is marginally standardized Pareto (for those
        obsv for which the row max > 1) onto the unit hypercube. returns those
        projections, the row max, and the indices in the original data
        corresponding to the observations """
        R = par.max(axis = 1)
        V = (par.T / R).T
        if decluster:
            I = cluster_max_row_ids(R)
        else:
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
        Z = scale_pareto(raw, P)
        return Z, P

    def __init__(self, raw, decluster = False, quantile = 0.95):
        # if input is pandas dataframe, then take numpy array representation
        try:
            self.raw = raw.values
        # else assume input is numpy array
        except AttributeError:
            self.raw = raw
        # Compute standardized pareto margins
        self.Z, self.P = self.to_pareto(self.raw, quantile)
        # Cast to hypercube, keep only observations extreme in >= 1 dimension
        self.V, self.R, self.I = self.to_hypercube(self.Z, decluster)
        # proceed with angular representation
        self.A = self.to_angular(self.V)
        # Number of columns for Gamma representation
        self.nCol = self.A.shape[1] + 1
        # Number of rows in data
        self.nDat = self.A.shape[0]
        # Pre-compute the trig components of the likelihood.
        self.fill_out()
        return

class Data_From_Sphere(Data):
    def fill_out(self):
        self.A = euclidean_to_angular(self.data.V)
        self.S = euclidean_to_simplex(self.data.V)
        return

    def __init__(self, path):
        pV = pd.read_csv(path).values
        self.V = euclidean_to_hypercube(pV)
        self.nDat, self.nCol = pV.shape
        self.fill_out()
        return

if __name__ == '__main__':
    pass

# EOF
