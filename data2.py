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

def rank_transform_pareto(
        raw   : np.ndarray, 
        Fhats : list,
        ) -> np.ndarray:
    """ Do Rank-Transform Standard Pareto Scaling """
    # Bounds Checking
    assert raw.shape[1] == len(Fhats)
    # Transformation
    Z = np.array([
        Fhat.stdpareto(x)
        for Fhat, x in zip(Fhats, raw.T)
        ])
    return Z

def rank_invtransform_pareto(
        Z : np.ndarray,
        Fhats : list,
        ) -> np.ndarray:
    # Bounds Checking
    assert Z.shape[1] == len(Fhats)
    # Transformation
    X = np.ndarray(
        Fhat.FhatInv(z)
        for Fhat, z in zip(Fhats, Z.T)
        )
    return X

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
    W   = None # C in one-hot, (N x (2*D))
    cats = None
    nCat = None
    iCat = None
    dCat = None


    def rescale(self, Z):
        return rescale_pareto(Z, self.P, self.C)

    @classmethod
    def from_raw(cls, raw : np.ndarray, q : float):
        P = compute_gp_parameters_2tail(raw, q)
        Z, C = standardize_pareto_2tail(raw, P)
        W = np.stack((C == 0,C == 1), dtype = int).reshape(
            Z.shape[0], 2 * Z.shape[1]
            )
        return cls(raw, P, Z, C, W, q)
    
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
            W   : np.ndarray,
            q   : float,
            ):
        self.raw = raw
        self.P = P
        self.Z = Z
        self.C = C
        self.W = W
        self.q = q
        self.cats = np.repeat(2, self.Z.shape[1])
        return
    
    pass

class Threshold_1Tail(Threshold_2Tail):
    cats = np.array([])
    nCat = 0
    iCat = np.array([])
    dCat = np.array([])

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

class RankTransform(DataBase):
    # Vestigial
    nCat = 0
    cats = np.array([])
    iCat = np.array([])
    dCat = np.array([])
    # Relevant
    raw   = None
    Z     = None
    Fhats = []

    @classmethod
    def from_raw(cls, raw : np.ndarray):
        Fhats = list(map(ECDF, raw.T))
        Z     = rank_transform_pareto(X)
        return cls(raw, Z, Fhats)

    def std_pareto_transform(self, X : np.ndarray) -> np.ndarray:
        return rank_transform_pareto(X, self.Fhats)
    
    def to_dict(self) -> dict:
        d = {
            'raw'   : self.raw,
            'Z'     : self.Z,
            'Fhats' : self.Fhats,
            }
        return d

    def __init__(self, raw : np.ndarray, Z : np.ndarray, Fhats : list):
        self.raw = raw
        self.Z = Z
        self.Fhats = Fhats
        return

class Multinomial(DataBase):
    cats = None # number of categories per multinomial variable
    nCat = None # total number of categories (sum of Cats)
    iCat = None # Int Vector associating columns with Vars 
    dCat = None # Bool Array associating columns with Vars
    raw  = None # Originating Data
    W    = None # Multinomial Data

    @classmethod
    def from_raw(
            cls, 
            raw  : np.ndarray, 
            cats : np.ndarray,
            ):
        nCat = cats.sum()
        try:
            assert nCat == raw.shape[1]
        except AssertionError:
            print('Total number of categories must equal number of columns')
            raise
        iCat = np.hstack([
            np.ones(cat, dtype = int) * i for i, cat in enumerate(cats)
            ])
        dCat = np.vstack([
            np.where(iCat == i)[0] for i in range(iCat.max() + 1)
            ])
        W    = raw.copy()
        return cls(cats, nCat, iCat, dCat, raw, W)
    
    def to_dict(self) -> dict:
        d = {
            'cats' : self.cats,
            'nCat' : self.nCat,
            'iCat' : self.iCat,
            'dCat' : self.dCat,
            'raw'  : self.raw,
            'W'    : self.W,
            }
        return d

    def __init__(
            self, 
            cats : np.ndarray, 
            nCat : int, 
            iCat : np.ndarray,
            dCat : np.ndarray,
            raw  : np.ndarray, 
            W    : np.ndarray,
            ):
        self.cats = cats
        self.nCat = nCat
        self.iCat = iCat
        self.dCat = dCat
        self.raw  = raw
        self.W    = W
        return

class Categorical(Multinomial):
    vals = None

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
        cats = np.array(cats, dtype = int)
        nCat = cats.sum()
        iCat = np.hstack([
            np.ones(cat, dtype = int) * i for i, cat in enumerate(cats)
            ])
        dCat = np.vstack([
            np.where(iCat == i)[0] for i in range(iCat.max() + 1)
            ])
        W = np.vstack(dummies.T)
        return cls(cats, nCat, iCat, dCat, raw, W, vals)
    
    def to_dict(self):
        d = {
            'cats' : self.cats,
            'nCat' : self.nCat,
            'iCat' : self.iCat, 
            'dCat' : self.dCat,
            'raw'  : self.raw,
            'W'    : self.W,
            'vals' : self.vals
            }
        return d

    def __init__(
            self,
            cats : np.ndarray,
            nCat : int,
            iCat : np.ndarray,
            dCat : np.ndarray,
            raw  : np.ndarray, 
            W    : np.ndarray,
            vals : list,
            ):
        self.vals = vals
        super().__init__(cats, nCat, iCat, dCat, raw, W)
        return

    pass 

class Data(DataBase):
    xh1t = None
    xh2t = None
    rank = None
    cate = None

    dcls = None

    Z = None
    W = None
    V = None
    R = None
    I = None
    
    def to_dict(self) -> dict:
        d = dict()
        for key in ['xh1t','xh2t','rank','cate']:
            if self.__dict__[key] is not None:
                d[key] = self.__dict__[key].to_dict()
        for key in ['Z','W','V','R','I','dcls']:
            d[key] = self.__dict__[key]
        return d
    
    @classmethod
    def from_dict(cls, d : dict):
        if 'xh1t' in d.keys():
            xh1t = Threshold_1Tail.from_dict(d['xh1t'])
        else:
            xh1t = None
        if 'xh2t' in d.keys():
            xh2t = Threshold_2Tail.from_dict(d['xh2t'])
        else: 
            xh2t = None
        if 'rank' in d.keys():
            rank = RankTransform.from_dict(d['rank'])
        else: 
            rank = None
        if 'cate' in d.keys():
            cate = Categorical.from_dict(d['cate'])
        else:
            cate = None
        addlkeys = dict()
        for key in ['Z','W','V','R','I','dcls']:
            if key in d.keys():
                addlkeys[key] = d[key]
        return cls(xh1t = xh1t, xh2t = xh2t, 
                   rank = rank, cate = cate, **addlkeys)
    
    @classmethod
    def from_raw(
            cls,
            xh1t : Threshold_1Tail = None, 
            xh2t : Threshold_2Tail = None, 
            rank : RankTransform = None,
            cate : Categorical = None,
            dcls : bool = False,
            ):
        inputs = [xh1t, xh2t, rank, cate]
        filted = [x for x in inputs if x is not None]
        assert len(np.unique([x.raw.shape[0] for x in filted])) == 1
        N = filted[0].raw.shape[0]
        Z = np.empty((N,0), dtype = float)
        W = np.empty((N,0), dtype = int)
        if xh1t is not None:
            Z = np.hstack((Z, xh1t.Z))
        if xh2t is not None:
            Z = np.hstack((Z, xh2t.Z))
            W = np.hstack((W, xh2t.W))
        if rank is not None:
            Z = np.hstack((Z, rank.Z))
        if cate is not None:
            W = np.hstack((W, cate.W))




    def __init__(
            self, 
            xh1t : Threshold_1Tail, 
            xh2t : Threshold_2Tail, 
            rank : RankTransform, 
            cate : Categorical, 
            Z    : np.ndarray, 
            W    : np.ndarray, 
            V    : np.ndarray, 
            R    : np.ndarray, 
            I    : np.ndarray, 
            dcls : bool,
            ):
        self.xh1t = xh1t
        self.xh2t = xh2t
        self.rank = rank
        self.cate = cate
        self.Z    = Z
        self.W    = W
        self.V    = V
        self.R    = R
        self.I    = I
        self.dcls = dcls
        return



if __name__ == '__main__':
    X = np.random.normal(loc = 1., size = (500, 6))
    P = compute_gp_parameters_2tail(X, 0.95)
    Z, C = standardize_pareto_2tail(X, P)
    Xn = rescale_pareto(Z, P, C)
    raise

# EOF
