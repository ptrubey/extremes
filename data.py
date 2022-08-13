import numpy as np
import sqlite3 as sql
import pandas as pd
import os
from genpareto import gpd_fit
from numpy.linalg import norm
from math import pi, sqrt, exp
from scipy.special import erf, erfinv

EPS = np.finfo(float).eps

def angular_to_hypercube(theta):
    """ Assuming data in polar, casts to euclidean then divides by row max
    to achieve projection onto hypercube. """
    euc = angular_to_euclidean(theta)
    return euclidean_to_hypercube(euc)

def angular_to_simplex(theta):
    euc = angular_to_euclidean(theta)
    return euclidean_to_simplex(euc)

def euclidean_to_simplex(euc):
    return ((euc + EPS).T / (euc + EPS).sum(axis = 1)).T

def euclidean_to_hypercube(euc):
    return ((euc + EPS).T / (euc + EPS).max(axis = 1)).T

def euclidean_to_psphere(euc, p = 10, epsilon = 1e-6):
    Yp = ((euc.T + EPS) / ((euc + EPS)**p).sum(axis = 1)**(1/p)).T
    Yp[Yp <= epsilon] = epsilon
    return Yp

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
    scratch = np.zeros(raw.shape)
    scratch += raw
    scratch -= P[0]
    scratch /= P[1]
    scratch *= P[2]
    scratch += 1.
    # scratch = 1 + P[2] * (raw - P[0]) / P[1]
    with np.errstate(invalid = 'ignore'):
        np.log(scratch, out = scratch)
    np.nan_to_num(scratch, copy = False, nan = -np.inf)
    scratch /= P[2]
    np.exp(scratch, out = scratch)
    # Z = (1 + P[2] * (raw - P[0]) / P[1])**(1/P[2])
    # Z[Z < 0.] = 0.
    return scratch

def descale_pareto(Z, P):
    """ Given Pareto scaled RV's, return original scale """
    raw = P[0] + P[1] * (Z**P[2] - 1) / P[2]
    return raw

class Outcome(object):
    I = None

    def fill_outcome(self, Y):
        self.Y = Y.ravel()[self.I]
        return

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
        return angular_to_euclidean(theta)

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

class Data_From_Raw(Data, Outcome):
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
        V = par / R[:,None] # (par.T / R).T
        if decluster:
            I = cluster_max_row_ids(R)
        else:
            I = np.where(R > 1)[0]
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
    
    def fill_real(self, raw, decluster, quantile):
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

    def __init__(self, raw, decluster = False, quantile = 0.95, outcome = 'None'):
        self.fill_real(raw, decluster, quantile)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class Data_From_Sphere(Data, Outcome):
    def fill_sphere(self, raw):
        self.V = euclidean_to_hypercube(raw)
        self.nDat, self.nCol = self.V.shape
        self.A = euclidean_to_angular(self.V)
        self.S = euclidean_to_simplex(self.V)
        self.I = np.arange(self.nDat)
        return

    def __init__(self, raw, outcome = 'None'):
        self.fill_sphere(raw)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class Multinomial(Data, Outcome):
    Cats = None  # numpy array indicating number of categories per multinomial variable
    nCat = None  # total number of categories (sum of Cats)
    spheres = None # For each variable, np.array(int) that identifies which 
                   #    columns are associated with that var.

    def fill_multinomial(self, raw, cats = None, index = None):
        if cats is None:
            cats = np.array([raw.shape[1]])
        if index is None:
            index = np.arange(raw.shape[0])
        self.I = index
        self.W = raw[index]
        self.Cats = cats
        self.nCat = self.Cats.sum()
        arr = np.hstack([np.ones(cat, dtype = int) * i for i, cat in enumerate(self.Cats)])
        self.spheres = [np.where(arr == i)[0] for i in range(arr.max() + 1)]
        
        assert self.nCat == self.W.shape[1]
        self.nDat = self.W.shape[0]
        return
    
    def __init__(self, raw, cats, index = None, outcome = 'None'):
        if index is None:
            index = np.arange(raw.shape[0])
        self.fill_multinomial(raw, cats, index)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class Categorical(Multinomial, Outcome):
    Cats = None    # numpy array indicating number of categories per categorical variable
    nCat = None    # Total number of categories (sum of Cats)

    def fill_categorical(self, raw, values, index):
        # If values are supplied, verify that all values in data
        # are represented in supplied.
        if values is not None:
            assert len(values) == raw.shape[1]
            for i in range(raw.shape[1]):
                assert len(set(raw.T[i]).difference(set(values[i]))) == 0
        else:
            values = [np.unique(raw.T[i]) for i in range(raw.shape[1])]
        
        if index is None:
            index = np.arange(raw.shape[0])

        dummies = []
        cats = []
        for i in range(raw.shape[1]):
            dummies.append(np.vstack([raw.T[i] == j for j in values[i]]))
            cats.append(len(values[i]))
        W = np.vstack(dummies).T
        self.fill_multinomial(W, np.array(cats), index)
        return
    
    def __init__(self, raw, values = None, index = None, outcome = 'None'):
        self.fill_categorical(raw, values, index)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class MixedDataBase(Data_From_Sphere, Multinomial, Outcome):
    def __init__(self, raw_sphere, raw_multinomial, cats = None, outcome = 'None'):
        self.fill_sphere(raw_sphere)
        self.fill_multinomial(raw_multinomial, cats)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class MixedData(MixedDataBase, Data_From_Raw, Categorical, Outcome):
    def __init__(self, raw, cat_vars = [], sphere = False,
            decluster = False, quantile = 0.95, values = None,
            outcome = 'None',
            ):
        if type(raw) is pd.DataFrame:
            raw = raw.values
        real_vars = np.array(
            list(set(np.arange(raw.shape[1])).difference(set(cat_vars))), 
            dtype = int
            )
        if sphere:
            self.fill_sphere(raw[:, real_vars])
        else:
            self.fill_real(raw[:, real_vars], decluster, quantile)
        self.fill_categorical(raw[:, cat_vars], values, self.I)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class Projection(object):
    def set_projection(self):
        self.data.Yp = euclidean_to_psphere(self.data.V, self.p)
        return
    pass

if __name__ == '__main__':
    pass

# EOF
