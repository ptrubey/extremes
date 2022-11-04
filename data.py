from dataclasses import dataclass
import numpy as np
import sqlite3 as sql
import pandas as pd
import os
from genpareto import gpd_fit
from numpy.linalg import norm
from math import pi, sqrt, exp
from scipy.special import erf, erfinv
from cdf import ECDF

EPS = np.finfo(float).eps
MAX = np.finfo(float).max

def category_matrix(cats):
    """ Forms a Boolean Category Matrix
        dims = [(# categorical vars), sum(# categories per var)]
    (c, sum(p_i)) """
    catvec = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(cats)))
    CatMat = (catvec[:, None] == np.arange(len(cats))).T
    return CatMat

def angular_to_hypercube(theta):
    """ Assuming data in polar, casts to euclidean then divides by row max
    to achieve projection onto hypercube. """
    euc = angular_to_euclidean(theta)
    return euclidean_to_hypercube(euc)

def angular_to_simplex(theta):
    euc = angular_to_euclidean(theta)
    return euclidean_to_simplex(euc)

# modified to work on 3d arrays
def euclidean_to_simplex(euc):
    """ projects R_+^d to S_1^{d-1} """
    return (euc + EPS) / (euc + EPS).sum(axis = -1)[...,None]

def euclidean_to_hypercube(euc):
    V = (euc + EPS) / (euc + EPS).max(axis = -1)[...,None]
    V[V < EPS] = EPS
    return V

def euclidean_to_psphere(euc, p = 10):
    Yp = (euc + EPS) / (((euc + EPS)**p).sum(axis = -1)**(1/p))[...,None]
    Yp[Yp < EPS] = EPS
    return Yp

def euclidean_to_catprob(euc, catmat):
    """ 
    euc    : (n,s,d) or (s,d)
    catmat : (c,d)
    """
    seuc = (euc @ catmat.T) + EPS
    neuc = np.einsum('...c,cd->...d', seuc, catmat)
    pis = euc / neuc
    return pis

# stuck on 2d
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
        temp[temp > (1 - EPS)] = 1 - EPS
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

class DataBase(Outcome):
    """ Base data class; to be amended with monkey patching. """
    def __init__(self, V = None, R = None, W = None, X = None, Y = None):
        if type(V) is np.ndarray:
            self.V = V
        if type(R) is np.ndarray:
            self.R = R
        if type(W) is np.ndarray:
            self.W = W
        if type(X) is np.ndarray:
            self.X = X
        if type(Y) is np.ndarray:
            self.fill_outcome(Y)
        return

class Data_From_Raw(DataBase):
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
        par[np.isinf(par)] = MAX
        R = par.max(axis = 1)
        with np.errstate(divide = 'ignore'):
            V = par / R[:,None] # (par.T / R).T
            V[V < EPS] = EPS
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

    def to_pareto_new(self, raw, P):
        Z = scale_pareto(raw, P)
        V, R, I = self.to_hypercube(Z)
        return V, R, I
    
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
        # Number of rows, columns
        self.nDat, self.nCol = self.V.shape
        return

    def __init__(self, raw, decluster = False, quantile = 0.95, outcome = 'None'):
        self.fill_real(raw, decluster, quantile)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class Data_From_Sphere(DataBase):
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

class RankTransform(DataBase):
    Fhats = None
    X = None
    Z = None
    V = None
    R = None

    def std_pareto_transform(self, X):
        assert(X.shape[1] == len(self.Fhats))
        Z = np.array([
            Fhat.stdpareto(x)
            for Fhat, x in zip(self.Fhats, X.T)
            ]).T
        return(Z)

    def fill_rank_transform(self, X):
        self.X = X
        self.Fhats = list(map(ECDF, self.X.T))
        self.Z = self.std_pareto_transform(self.X)
        self.R = self.Z.max(axis = 1)
        self.V = self.Z / self.R[:,None]
        self.nDat, self.nCol = self.X.shape
        return

    def __init__(self, raw, outcome = 'None'):
        self.fill_rank_transform(raw)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class Multinomial(DataBase):
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
    
    def to_categorical_new(self, raw_data, raw_out, index = None):
        if index is None:
            index = np.arange(raw_data.shape[0])
        W = Categorical.to_categorical(raw_data, self.values, index)
        Y = raw_out[index]
        return Y, W
    
    @staticmethod
    def to_multinomial(W, index):
        return W[index]
    
    def __init__(self, raw, cats, index = None, outcome = 'None'):
        if index is None:
            index = np.arange(raw.shape[0])
        self.fill_multinomial(raw, cats, index)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class Categorical(Multinomial):
    Cats   = None    # numpy array indicating number of cats per categ variable
    nCat   = None    # Total number of categories (sum of Cats)
    values = None    # particular values indicating which category, per var

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
        self.values = values
        self.fill_multinomial(W, np.array(cats), index)
        return
    
    @staticmethod
    def to_categorical(raw, values, index):
        dummies = []
        assert raw.shape[1] == len(values)
        for i in range(raw.shape[1]):
            dummies.append(np.vstack([raw.T[i] == j for j in values[i]]))
        W = np.vstack(dummies).T
        return Multinomial.to_multinomial(W, index)
    
    def to_categorical_new(self, raw_data, raw_out, index = None):
        if index is None:
            index = np.arange(raw_data.shape[0])
        W = self.to_categorical(raw_data, self.values, index)
        Y = raw_out[index]
        return Y, W

    def __init__(self, raw, values = None, index = None, outcome = 'None'):
        self.fill_categorical(raw, values, index)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class MixedDataBase(Data_From_Sphere, Data_From_Raw, RankTransform, Multinomial):
    def to_mixed_new(self, raw_data, raw_out, decluster = False):
        if self.realtype == 'threshold':
            Z = scale_pareto(raw_data[:,:self.nCol], self.P)
            V, R, I = Data_From_Raw.to_hypercube(Z, decluster = decluster)
            W = Categorical.to_categorical(
                raw_data[:,self.nCol:], self.values, I,
                )
            Y = raw_out[I]
        elif self.realtype == 'sphere':
            assert((raw_data[:,:self.nCol].max() - 1)**2 < 1e-10) # check data on sphere
            V = raw_data[:,:self.nCol]
            W = Categorical.to_categorical(
                raw_data[:,self.nCol:], self.values, np.arange(V.shape[0]),
                )
            Y = raw_out
            R = np.ones(V.shape[0])
        elif self.realtype == 'rank':
            Z = self.std_pareto_transform(raw_data[:,:self.nCol])
            R = Z.max(axis = 1)
            V = Z / R[:,None]
            W = Categorical.to_categorical(
                raw_data[:,self.nCol:], self.values, np.arange(V.shape[0])
                )
            Y = raw_out
        return Y,V,W,R
    
    @classmethod
    def instantiate_from_dict(cls, dict):
        if 'realtype' in dict.keys():
            if dict['realtype'] == 'sphere':
                data = cls(dict['V'], dict['W'], dict['cats'], real_type = 'sphere')
            elif dict['realtype'] == 'threshold':
                data = cls(dict['V'], dict['W'], dict['cats'], real_type = 'threshold', parameters = dict['P'])
            elif dict['realtype'] == 'rank':
                data = cls(dict['X'], dict['W'], dict['cats'], real_type = 'rank')
            else:
                raise ValueError('realtype not recognized')
        elif all([_ in dict.keys() for _ in ('V','P','W','cats')]):
            data = cls(dict['V'], dict['W'], dict['cats'], 
                    real_type = 'threshold', parameters = dict['P'])
        elif all([_ in dict.keys() for _ in ('V','X','W','cats')]):
            data = cls(dict['X'], dict['W'], dict['cats'], real_type = 'rank')
        elif all([_ in dict.keys() for _ in ('V','W','cats')]):
            data = cls(dict['V'], dict['W'], dict['cats'], real_type = 'sphere')
        else:
            raise ValueError('Required Dictionary Contents Not Available')
        if 'Y' in dict.keys():
            data.Y = dict['Y']
        if 'values' in dict.keys():
            data.values = dict['values']
        if 'raw' in dict.keys():
            data.raw = dict['raw']
        return data
    
    def __init__(
            self, raw_real, raw_multinomial, cats = None, real_type = 'sphere', 
            outcome = 'None', parameters = None,
            ):
        self.realtype = real_type
        if real_type == 'sphere':
            self.fill_sphere(raw_real)
        elif real_type == 'rank': 
            self.fill_rank_transform(raw_real)            
        elif real_type == 'threshold':
            self.fill_sphere(raw_real)
            self.P = parameters
        self.fill_multinomial(raw_multinomial, cats)
        if type(outcome) is np.ndarray:
            self.fill_outcome(outcome)
        return

class MixedData(MixedDataBase, Categorical):
    realtype = None
    def write_to_dict(self, dict):
        """ Updates output dictionary in-place with contents of data class """
        if self.realtype == 'sphere':
            dict['realtype'] = 'sphere'
            dict['V'] = self.V
        elif self.realtype == 'rank':
            dict['realtype'] = 'rank'
            dict['V'] = self.V
            dict['Z'] = self.Z
            dict['R'] = self.R
            dict['X'] = self.X
        elif self.realtype == 'threshold':
            dict['realtype'] = 'threshold'
            dict['V'] = self.V
            dict['Z'] = self.Z
            dict['R'] = self.R
            dict['I'] = self.I
            dict['P'] = self.P
        dict['W'] = self.W
        dict['values'] = self.values
        dict['raw']  = self.raw
        if hasattr(self, 'Y'):
            dict['Y'] = self.Y
        return
    def __init__(
            self, raw, cat_vars = [], realtype = 'threshold', decluster = False, 
            quantile = 0.95, values = None, outcome = 'None',
            ):
        self.realtype = realtype
        if type(raw) is pd.DataFrame:
            raw = raw.values
        self.raw = raw
        real_vars = np.array(
            list(set(np.arange(raw.shape[1])).difference(set(cat_vars))), 
            dtype = int
            )
        if realtype == 'sphere':
            self.fill_sphere(raw[:, real_vars])
        elif realtype == 'rank':
            self.fill_rank_transform(raw[:, real_vars])
        elif realtype == 'threshold':
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
