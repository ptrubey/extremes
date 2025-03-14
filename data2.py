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
    if len(cats) == 0:
        return np.array([])
    catvec = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(cats)))
    CatMat = (catvec[:, None] == np.arange(len(cats))).T
    return CatMat

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

def compute_uni_gp_parms(
        rv  : np.ndarray, 
        q   : float,
        ):
    """ Compute Univariate GP parameters given quantile"""
    b = np.quantile(rv, q)
    a, xi = gpd_fit(rv, b)
    return np.array((b,a,xi))

def compute_gp_parameters_2tail(raw : np.ndarray, q : float):
    """ Computes GP parameters for both tails. """
    P = np.zeros((2,3,raw.shape[1]))
    P[0] = np.apply_along_axis(lambda x: compute_uni_gp_parms(x, q), 0, raw)
    P[1] = np.apply_along_axis(lambda x: compute_uni_gp_parms(x, q), 0, -raw)
    return P

def compute_gp_parameters_1tail(raw : np.ndarray, q : float):
    """ Computes GP parameters only for the upper tail """
    P = np.zeros((1,3,raw.shape[1]))
    P[0] = np.apply_along_axis(lambda x: compute_uni_gp_parms(x, q), 0, raw)
    return P

def rescale_pareto(
        Z : np.ndarray,
        P : np.ndarray,
        C : np.ndarray = None,
        ) -> np.ndarray:
    if C is None:
        C = np.zeros(Z.shape, int)
    # Bounds Checking
    assert Z.shape[1] == P.shape[2]
    assert Z.shape == C.shape
    assert P.shape[0] == 2
    assert P.shape[1] == 3
    # Transformation
    scratch = np.zeros((2,*Z.shape))
    scratch = Z[None]**P[:,[2]]
    scratch -= 1
    scratch /= P[:,[2]]
    scratch *= P[:,[1]]
    scratch += P[:,[0]]
    # Data on real scale
    out = np.where(C == 0, scratch[0], -scratch[1])
    return out

def standardize_pareto_2tail(
        raw : np.ndarray, 
        P : np.ndarray,
        ) -> tuple:
    """ Do the Pareto Scaling for both tails """
    # Bounds Checking
    assert raw.shape[1] == P.shape[2]
    assert P.shape[0] == 2
    assert P.shape[1] == 3
    # Transformation
    scratch = np.zeros((2, *raw.shape))
    scratch[0] += raw
    scratch[1] -= raw
    scratch -= P[:,[0]]
    scratch /= P[:,[1]]
    scratch *= P[:,[2]]
    scratch += 1.
    with np.errstate(invalid = 'ignore'):
        np.log(scratch, out = scratch)
    np.nan_to_num(scratch, copy = False, nan = -np.inf)
    scratch /= P[:,[2]]
    np.exp(scratch, out = scratch)
    # Checking which regime to respect: maximum or minimum
    C = np.argmax(scratch, axis = 0)
    Z = np.max(scratch, axis = 0)
    return Z, C

def standardize_pareto_1tail(
        raw : np.ndarray,
        P   : np.ndarray,
        ) -> np.ndarray:
    """ Do the Pareto scaling for 1 tail """
    assert raw.shape[1] == P.shape[2]
    assert P.shape[0] == 1
    assert P.shape[1] == 3
    scratch = np.zeros(raw.shape)
    scratch += raw
    scratch -= P[:,[0]]
    scratch /= P[:,[1]]
    scratch *= P[:,[2]]
    scratch += 1
    with np.errstate(invalid = 'ignore'):
        np.log(scratch, out = scratch)
    np.nan_to_num(scratch, copy = False, nan = -np.inf)
    scratch /= P[:,[2]]
    np.exp(scratch, out = scratch)
    return scratch

class DataBase(object):
    def to_dict(self) -> dict:
        raise NotImplementedError('Overwrite Me!')
    
    @classmethod
    def from_dict(cls, d : dict):
        return cls(**d)
    
    @classmethod
    def from_raw(cls, **kwargs):
        raise NotImplementedError('Overwrite Me!')

class Threshold_2Tail(DataBase):
    raw = None # Raw data (N x D)
    P   = None # Generalized Pareto Parameters (1 (or 2) x 3 x D)
    Z   = None # Standardized Pareto\
    C   = None # 0 -> Upper tail, 1 -> Lower tail
    Cm  = None # C in one-hot, ravelled (N x (2*D))

    def rescale(self, Z):
        return rescale_pareto(Z, self.P, self.C)

    @classmethod
    def from_raw(cls, raw : np.ndarray, q : float):
        P = compute_gp_parameters_2tail(raw, q)
        Z, C = standardize_pareto_2tail(raw, P)
        Cm = np.stack((C == 0,C == 1), dtype = int).reshape(
            Z.shape[0], 2 * Z.shape[1]
            )
        return cls(raw, P, Z, C, Cm, q)
    
    def to_dict(self) -> dict:
        d = {
            'raw'   : self.raw,
            'P'     : self.P,
            'Z'     : self.Z,
            'C'     : self.C,
            'Cm'    : self.Cm,
            'q'     : self.q,
            }
        return d

    def __init__(self,
            raw : np.ndarray, 
            P   : np.ndarray, 
            Z   : np.ndarray,
            C   : np.ndarray,
            Cm  : np.ndarray,
            q   : float,
            ):
        self.raw = raw
        self.P  = P
        self.Z  = Z
        self.C  = C
        self.Cm = Cm
        self.q  = q
        return
    
    pass

class Threshold_1Tail(Threshold_2Tail):
    @classmethod
    def from_raw(cls, raw : np.ndarray, q : float):
        P = compute_gp_parameters_1tail(raw, q)
        Z = standardize_pareto_1tail(raw, P)
        return cls(raw, P, Z, q)
    
    def __init__(
            self, 
            raw : np.ndarray, 
            P   : np.ndarray, 
            Z   : np.ndarray, 
            q   : float, 
            **kwargs
            ):
        self.raw = raw
        self.P   = P
        self.Z   = Z
        return
    
    pass

class Multinomial(DataBase):
    cats = None # number of categories per multinomial variable
    nCat = None # total number of categories (sum of Cats)
    iCat = None # Int array associating columns with Vars
    raw  = None # Originating Data
    W    = None # Multinomial Data

    @classmethod
    def from_raw(cls, 
            raw  : np.ndarray, 
            cats : np.ndarray = None,
            ):
        if cats is None:
            cats = np.array([raw.shape[1]])
        nCat = cats.sum()
        temp = np.hstack([
            np.ones(cat, dtype = int) * i for i, cat in enumerate(cats)
            ])
        iCat = [np.where(temp == i)[0] for i in range(temp.max() + 1)]
        W    = raw.copy()
        return cls(cats, nCat, iCat, raw, W)
    
    def to_dict(self) -> dict:
        d = {
            'cats' : self.cats,
            'nCat' : self.nCat,
            'iCat' : self.iCat,
            'raw'  : self.raw,
            'W'    : self.W,
            }
        return d

    def __init__(
            self, 
            cats : np.ndarray, 
            nCat : int, 
            iCat : np.ndarray,
            raw  : np.ndarray, 
            W    : np.ndarray,
            ):
        self.cats = cats
        self.nCat = nCat
        self.iCat = iCat
        self.raw  = raw
        self.W    = W
        return

class Categorical(Multinomial):
    values = None

    @classmethod
    def from_raw(
            cls, 
            raw : np.ndarray, 
            vals : list,
            ):
        # Verify Supplied Values
        if vals is not None:
            assert len(vals) == raw.shape[1]
            for i in range(raw.shape[1]):
                assert len(set(raw.T[i]).difference(set(vals[i]))) == 0
        # If not supplied, make new one based on existing data
        else:
            vals = [np.unique(raw.T[i]) for i in range(raw.shape[1])]

        dummies = []
        cats  = []
        for i in range(raw.shape[1]):
            dummies.append(np.vstack([raw.T[i] == j for j in vals[i]]))
            cats.append(len(vals[i]))
        W = np.vstack(dummies.T)
        iCatL = []
    
    def __init__(
            self,
            cats,
            nCat,
            iCat,
            raw, 
            W,
            vals,
            ):
        self.cats = cats
        self.nCat = nCat
        self.iCat = iCat
        self.raw  = raw
        self.W    = W
        self.vals = vals
        return

    pass 




if __name__ == '__main__':
    X = np.random.normal(loc = 1., size = (500, 6))
    P = compute_gp_parameters_2tail(X, 0.95)
    Z, C = standardize_pareto_2tail(X, P)
    Xn = rescale_pareto(Z, P, C)
    raise

# EOF
