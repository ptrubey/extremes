""" Functions relating to density of Projected Gamma.  All functions are
parameterized such that E(x) = alpha / beta (treat beta as rate parameter). """

import numpy as np
np.seterr(under = 'ignore', over = 'raise')
from numpy.linalg import norm
# from math import cos, sin, log, acos, exp
# from scipy.stats import gamma, uniform, norm as normal
from scipy.special import gammaln
from functools import lru_cache
from genpareto import gpd_fit
from collections import namedtuple
# from slice import univariate_slice_sample, skip_univariate_slice_sample

from scipy.integrate import nquad, tplquad

from numpy import log, exp
from numpy.random import gamma, normal, uniform
from scipy.stats import dirichlet

# Tuples for storing priors

GammaPrior     = namedtuple('GammaPrior', 'a b')
DirichletPrior = namedtuple('DirichletPrior', 'a')
BetaPrior      = namedtuple('BetaPrior', 'a b')

def d_projgamma_p(Y, alpha, beta, p, logd = True):
    """
    Y     : (n * d) array
    alpha : (d) array
    beta  : (d) array
    p     : scalar (norm of hypersphere projection)
    """
    Y = Y.reshape(-1, alpha.shape[0]) # Account for 1-dimensional input
    ld = (
        + (alpha * log(beta)).sum()
        - gammaln(alpha).sum()
        + ((alpha - 1) * log(Y)).sum(axis = 1)
        + log(Y.T[-1] + Y.T[-1]**((1-p)/p) * (Y.T[:-1].T**p).sum(axis = 1))
        # + (1 - p) / p * log(1 - (Y.T[:-1]**p).sum(axis = 0))
        + gammaln(alpha.sum())
        - alpha.sum() * log(np.einsum('ik,k->i', Y, beta))
        )
    if logd:
        return ld
    else:
        return exp(ld)
    pass

def d_projgamma_p2(Y, alpha, beta, p, logd = True):
    """
    Y     : (n * d) array
    alpha : (d) array
    beta  : (d) array
    p     : scalar (norm of hypersphere projection)
    """
    Y = Y.reshape(-1, alpha.shape[0]) # Account for 1-dimensional input
    ld = (
        + (alpha * log(beta)).sum()
        - gammaln(alpha).sum()
        + ((alpha - 1) * log(Y)).sum(axis = 1)
        # + log(Y.T[-1] + Y.T[-1]**((1-p)/p) * (Y.T[:-1].T**p).sum(axis = 1))
        + (1 - p) / p * log(1 - (Y.T[:-1]**p).sum(axis = 0))
        + gammaln(alpha.sum())
        - alpha.sum() * log(np.einsum('ik,k->i', Y, beta))
        )
    if logd:
        return ld
    else:
        return exp(ld)
    pass

def wrapper1(y1, y2, y3, alpha, beta, p):
    """ Y is d-1 vector; alpha,beta are d vectors """
    Y_ = np.array([y1,y2,y3])
    if (Y_**p).sum() > 1:
        return 0.
    Yd = (1 - (Y_**p).sum())**(1/p)
    return d_projgamma_p(np.hstack((Y_, Yd)).reshape(1,-1), alpha, beta, p, False)

def wrapper2(y1, y2, y3, alpha, beta, p):
    Y_ = np.array([y1,y2,y3])
    """ Y is d-1 vector; alpha,beta are d vectors """
    if (Y_**p).sum() > 1:
        return 0.
    Yd = (1 - (Y_**p).sum())**(1/p)
    return d_projgamma_p2(np.hstack((Y_, Yd)).reshape(1,-1), alpha, beta, p, False)


if __name__ == '__main__':
    alpha = np.array((0.5, 2., 5., 3.))
    gbeta  = np.array((1., 0.7, 1.2, 0.9))
    Y     = gamma(alpha, scale = 1 / gbeta, size = (20, 4))
    Y_1   = (Y.T / Y.sum(axis = 1)).T
    Y_2   = (Y.T / np.sqrt((Y**2).sum(axis = 1))).T
    Y_i   = (Y.T / Y.max(axis = 1)).T
    Y_10  = (Y.T / (Y**10).sum(axis = 1)**(1/10)).T

    gfun = lambda x: 0.
    hfun = lambda x: (1 - x**2)**(1/2)
    qfun = lambda x,y: 0.
    rfun = lambda x,y: (1 - x**2 - y**2)**(1/2)
    args = {'gfun' : gfun, 'hfun' : hfun,
            'qfun' : qfun, 'rfun' : rfun,
            'args' : (alpha, gbeta, 2)}

    # i1 = nquad(wrapper1, ranges = ((0., 1.), (0.,1.), (0., 1.)), args = (alpha, gbeta, 2))
    i1 = tplquad(wrapper1, a = 0, b = 1, **args)
    print(i1)
    # i2 = nquad(wrapper2, ranges = ((0., 1.), (0.,1.), (0., 1.)), args = (alpha, gbeta, 2))
    i2 = tplquad(wrapper2, a = 0, b = 1, **args)
    print(i2)


    # print('Dirichlet')
    # print(dirichlet(alpha).logpdf(Y_1.T))
    # print('Projected Gamma L1')
    # print(d_projgamma_p(Y_1, alpha, np.ones(4), 1))
    # print(d_projgamma_p(Y_2, alpha, gbeta, 2))
    # print(d_projgamma_p2(Y_2, alpha, gbeta, 2))
# EOF
