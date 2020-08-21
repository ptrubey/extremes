""" Functions relating to density of Projected Gamma.  All functions are
parameterized such that E(x) = alpha / beta (treat beta as rate parameter). """

import numpy as np
from numpy.linalg import norm
from math import cos, sin, log, acos, exp
from scipy.stats import gamma, uniform, norm as normal
from scipy.special import gammaln
from functools import lru_cache
from genpareto import gpd_fit
from collections import namedtuple
from cSlice import SliceSample
cimport numpy as np


# Tuples for storing priors

GammaPrior     = namedtuple('GammaPrior', 'a b')
DirichletPrior = namedtuple('DirichletPrior', 'a')

## Functions related to projected gamma density

cpdef double logdprojgamma(
        np.ndarray[dtype = np.float64_t, ndim = 2] coss,
        np.ndarray[dtype = np.float64_t, ndim = 2] sins,
        np.ndarray[dtype = np.float64_t, ndim = 2] sinp,
        np.ndarray[dtype = np.float64_t, ndim = 1] alpha,
        np.ndarray[dtype = np.float64_t, ndim = 1] beta
        ):
    """ Log-density of projected gamma.  Inputs have already been
    pre-formatted for use.
    coss = matrix of cos(theta), last col = 1  (n x k)
    sins = matrix of sin(theta), first col = 1 (n x k)
    sinp = cumulative product of sins, by column (n x k)
    alpha = vector of shape parameters for underlying gamma distributions
    beta  = vector of rate parameters for underlying gamma distributions
    """
    cdef np.ndarray[dtype = np.float_t, ndim = 2] Yl
    cdef np.ndarray[dtype = np.float_t, ndim = 1] B, asinv
    cdef double A, lp

    Yl    = coss * sinp
    A     = alpha.sum()
    B     = (beta * Yl).sum(axis = 1)
    asinv = np.cumsum(alpha[1:][::-1])[::-1]

    lp = (
        + gammaln(A)
        - A * log(B)
        + (alpha * log(beta) - gammaln(alpha)).sum()
        + (log(coss[:,:(k-1)]) * (alpha[:(k-1)] - 1)).sum(axis = 1)
        + (log(sins[:,1:]) * (asinv - 1)).sum(axis = 1)
        )
    return lp

cpdef double logdprojgamma_pre(
        np.ndarray[dtype = np.float64_t, ndim = 2] lcoss,
        np.ndarray[dtype = np.float64_t, ndim = 2] lsins,
        np.ndarray[dtype = np.float64_t, ndim = 2] Yl,
        np.ndarray[dtype = np.float64_t, ndim = 1] alpha,
        np.ndarray[dtype = np.float64_t, ndim = 1] beta
        ):
    """ Log-density of projected gamma.  Inputs have been pre-computed as
    much as possible.
    lcoss = log(matrix of cos(theta), last col = 1  (n x k))
    lsins = log(matrix of sin(theta), first col = 1 (n x k))
    Yl    = latent Y matrix--projection of direction vector onto unit hypersphere
           in Euclidean space. (n * k)
    alpha = vector of shape parameters for underlying gamma distributions
    beta  = vector of rate parameters for underlying gamma distributions
    """
    cdef np.ndarray[dtype = np.float_t, ndim = 1] B, asinv
    cdef double A, lp
    cdef int k

    A = alpha.sum()
    B = (beta * Yl).sum(axis = 1)
    k = lcoss.shape[1]
    asinv = np.cumsum(alpha[1:][::-1])[::-1]
    lp = (
        + gammaln(A)
        - A * np.log(B)
        + (alpha * np.log(beta) - gammaln(alpha)).sum()
        + (lcoss[:,:(k - 1)] * (alpha[:(k - 1)] - 1)).sum(axis = 1)
        + (lsins[:,1:] * (asinv - 1)).sum(axis = 1)
        )
    return lp

cpdef double dprojgamma(
        np.ndarray[dtype = np.float64_t, ndim = 2] theta,
        np.ndarray[dtype = np.float64_t, ndim = 1] alpha,
        np.ndarray[dtype = np.float64_t, ndim = 1] beta,
        bint logd = False,
        ):
    cdef int k
    cdef np.ndarray[dtype = np.float64_t, ndim = 2] coss, sins, sinp
    cdef double ld

    k = len(alpha)
    assert all(len(theta) == k - 1, len(beta) == k)

    coss = np.vstack((np.cos(theta).T, 1)).T
    sins = np.vstack((1,np.sin(theta).T)).T
    sinp = np.cumprod(sins, axis = 1)

    ld = logdprojgamma(coss, sins, sinp, alpha, beta)

    if logd:
        return ld
    else:
        return exp(ld)

cpdef double dprojgamma_trig(
        np.ndarray[dtype = np.float64_t, ndim = 2] s_theta,
        np.ndarray[dtype = np.float64_t, ndim = 2] c_theta,
        np.ndarray[dtype = np.float64_t, ndim = 1] alpha,
        np.ndarray[dtype = np.float64_t, ndim = 1 ]beta,
        bint logd = False
        ):
    cdef np.ndarray[dtype = np.float64_t, ndim = 2] coss, sins, sinp
    cdef double ld

    coss = np.vstack((c_theta.T, 1)).T
    sins = np.vstack((1, s_theta.T)).T
    sinp = np.cumprod(sins, axis = 1)

    ld = logdprojgamma(coss, sins, sinp, alpha, beta)

    if logd:
        return ld
    else:
        return exp(ld)

# def dprojgamma_latent(Y, alpha, beta):
#     pass

## Function for sampling from projected gamma

def rprojgamma():
    pass

## Functions related to sampling for parameters from posterior, assuming
## a projected gamma likelihood.

# @lru_cache(maxsize = 32)
cpdef double log_post_log_alpha_1(
        double log_alpha_1,
        np.ndarray[dtype = np.float_t, ndim = 1] y_1,
        GammaPrior prior,
        ):
    """ Log posterior for log-alpha_1 assuming a gamma distribution,
    with beta assumed to be 1. """

    alpha_1 = exp(log_alpha_1)
    n_1     = y_1.shape[0]
    lp = (
        + prior.a * log_alpha_1
        - prior.b * alpha_1
        + (alpha_1 - 1) * np.log(y_1).sum()
        - n_1 * gammaln(alpha_1)
        )
    return lp

cpdef double sample_alpha_1_mh(
        double curr_alpha_1,
        np.ndarray[dtype = np.float64_t, ndim = 1] y_1,
        GammaPrior prior,
        proposal_sd = 0.1,
        ):
    """ Sampling function for shape parameter, with gamma likelihood and
    gamma prior.  Assumes rate parameter = 1.  uses Metropolis Hastings
    algorithm with random walk for sampling. """
    if len(y_1) <= 1:
        return gamma.rvs(prior.a, scale = 1./prior.b)

    curr_log_alpha_1 = log(curr_alpha_1)
    prop_log_alpha_1 = curr_log_alpha_1 + normal.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha_1(curr_log_alpha_1, y_1, prior)
    prop_lp = log_post_log_alpha_1(prop_log_alpha_1, y_1, prior)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_1)
    else:
        return curr_alpha_1

cpdef double sample_alpha_1_slice(
        double curr_alpha_1,
        np.ndarray[dtype = np.float64_t, ndim = 1] y_1,
        GammaPrior prior,
        double increment_size = 0.2
        ):
    cdef double f = lambda log_alpha_1: log_post_log_alpha_1(log_alpha_1, y_1, prior)
    return exp(univariate_slice_sample(f, log(curr_alpha_1), increment_size))

#@lru_cache(maxsize = 128)
cpdef double log_post_log_alpha(
        double log_alpha,
        np.ndarray[dtype = np.float64_t, ndim = 1] y,
        GammaPrior prior_a,
        GammaPrior prior_b,
        ):
    """ Log posterior for log-alpha assuming a gamma distribution,
    beta integrated out of the posterior. """
    alpha = exp(log_alpha)
    n     = y.shape[0]
    lp = (
        + (alpha - 1) * np.log(y).sum()
        - n * gammaln(alpha)
        + prior_a.a * log_alpha
        - prior_a.b * alpha
        + gammaln(n * alpha + prior_b.a)
        - (n * alpha + prior_b.a) * log(y.sum() + prior_b.b)
        )
    return lp

cpdef double sample_alpha_k_mh(
        double curr_alpha_k,
        np.ndarray[dtype = np.float64_t, ndim = 1] y_k,
        GammaPrior prior_a,
        gammaPrior prior_b,
        double proposal_sd = 0.1
        ):
    """ Sampling Function for shape parameter, with Gamma likelihood and Gamma
    prior, with rate (with gamma prior) integrated out. """
    if len(y_k) <= 1:
        return gamma.rvs(prior_a.a, scale = 1./prior_a.b)

    curr_log_alpha_k = log(curr_alpha_k)
    prop_log_alpha_k = curr_log_alpha_k + normal.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha(curr_log_alpha_k, y_k, prior_a, prior_b)
    prop_lp = log_post_log_alpha(prop_log_alpha_k, y_k, prior_a, prior_b)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_k)
    else:
        return curr_alpha_k

def sample_alpha_k_slice(curr_alpha_k, y_k, prior_a, prior_b, increment_size = 0.2):
    f = lambda log_alpha_k: log_post_log_alpha_k(log_alpha_k, y_k, prior_a, prior_b)
    return exp(univariate_slice_sample(f, log(curr_alpha_k), increment_size))

def sample_beta_fc(alpha, y, prior):
    aa = len(y) * alpha + prior.a
    bb = sum(y) + prior.b
    return gamma.rvs(aa, scale = 1. / bb)

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
