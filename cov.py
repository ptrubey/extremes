""" online updating of covariance matrices """

import numpy as np
from numpy.linalg import cholesky
from numpy.random import normal


def initial_mean(data):
    return data.sum(axis = 0) / data.shape[0]

def initial_cov(data, mean):
    return np.einsum('ij,il->jl', data - mean, data - mean) / data.shape[0]

def update_mean(mean, n, new):
    return (mean * n + new) / (n + 1)

def update_mean_inplace(mean, n, new):
    mean *= n
    mean += new
    mean /= (n + 1)
    return

def update_cov(cov, n, mean, new):
    out = np.zeros(cov.shape)
    out += n/(n+1) * cov
    out += n/(n+1)/(n+1) * (new - mean) * (new - mean)[:,None]
    return out

def update_cov_inplace(cov, n, mean, new):
    cov *= n/(n+1)
    cov += n/(n+1)/(n+1) * (new - mean) * (new - mean)[:,None]
    return

def tempered_initial_mean(data):
    """ same as initial_mean, assuming axis 0 is axis of time """
    return data.sum(axis = 0) / data.shape[0]

def tempered_initial_cov(data, mean):
    """ same as initial_cov, assuming axis 0 is axis of time """
    return np.einsum('itj,itl->tjl', data - mean, data - mean) / data.shape[0]

def tempered_update_mean(mean, n, new):
    """
    mean : (t,d)
    n    : (scalar)
    new  : (t,d)
    """
    return (mean * n + new) / (n + 1)

def tempered_update_cov(cov, n, mean, new):
    """ 
    same as update_cov, assuming axis 0 is axis of time, axis 1 is axis of 
        temperature. 
    cov  : (t,d,d)
    n    : scalar
    mean : (t, d)
    new  : (t, d)
    """
    out = np.zeros(cov.shape)
    out += n / (n+1) * cov
    out += n / (n+1) / (n+1) * np.einsum(
        'tj,tl->tjl',
        new - mean, new - mean,
        )
    return out

def tempered_update_mean_inplace(mean, n, new):
    mean *= n
    mean += new
    mean /= (n + 1)
    return

def tempered_update_cov_inplace(cov, n, mean, new):
    cov *= n / (n + 1)
    cov += n / (n + 1) / (n + 1) * np.einsum(
        'tj,tl->tjl', new - mean, new - mean,
        )
    return

def per_obs_tempered_update_mean(mean, n, new):
    """
    mean : (t, n, d)
    n    : scalar 
    new  : (t, n, d)
    """
    return (mean * n + new) / (n + 1)

def per_obs_tempered_update_cov(cov, n, mean, new):
    """
    cov : (t, n, d, d)
    n   : scalar
    mean : (t, n, d)
    new  : (t, n, d)
    """
    out = np.zeros(cov.shape)
    out += n / (n + 1) * cov
    out += n / (n + 1) / (n + 1) * np.einsum(
        'tnj,tnl->tnjl',
        new - mean, new - mean
        )
    return out

def per_obs_tempered_update_mean_inplace(mean, n, new):
    mean *= n
    mean += new
    mean /= (n + 1)
    return

def per_obs_tempered_update_cov_inplace(cov, n, mean, new):
    """
    cov : (t, n, d, d)
    n   : scalar
    mean : (t, n, d)
    new  : (t, n, d)
    """
    cov *= n / (n + 1)
    cov += n / (n + 1) / (n + 1) * np.einsum(
        'tnj,tnl->tnjl', new - mean, new - mean,
        )
    return

if __name__ == "__main__":
    starting_cov = np.array([3,0.3,-1.3,0.3,2,0.7,-1.3,0.7,2.5]).reshape(3,3)
    starting_cho = cholesky(starting_cov)

    z = normal(size = (1000, 3))
    x = (starting_cho @ z.T).T







