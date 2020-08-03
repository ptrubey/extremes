""" Functions relating to density of Projected Gamma.  All functions are
parameterized such that E(x) = alpha / beta (treat beta as rate parameter). """

import numpy as np
from numpy.linalg import norm
from numpy import arccos
from math import cos, sin, log, acos
from scipy.stats import gamma, norm, uniform
from functools import lru_cache

## Functions related to projected gamma density

def logdprojgamma(coss, sins, sinp, alpha, beta):
    """ Log-density of projected gamma.  Inputs have already been
    pre-formatted for use.
    coss = matrix of cos(theta), last col = 1  (n x k)
    sins = matrix of sin(theta), first col = 1 (n x k)
    sinp = cumulative product of sins, by column (n x k)
    alpha = vector of shape parameters for underlying gamma distributions
    beta  = vector of rate parameters for underlying gamma distributions
    """
    Yl = coss * sinp
    A = alpha.sum()
    B = (beta * Yl).sum(axis = 1)

    #asinv = np.cumsum(alpha[(k+1):1])[k:0]
    asinv = np.cumsum(alpha[:1:-1])[::-1]

    lp = (
        + lgamma(A)
        - A * log(B)
        + (alpha * log(beta) - lgamma(alpha)).sum()
        + (log(coss[:k]) * (alpha[:k] - 1)).sum(axis = 1)
        + (log(sins[1:]) * (asinv - 1)).sum(axis = 1)
        )
    return lp

def logdprojgamma_pre(lcoss, lsins, Yl, alpha, beta):
    """ Log-density of projected gamma.  Inputs have been pre-computed as
    much as possible.
    lcoss = log(matrix of cos(theta), last col = 1  (n x k))
    lsins = log(matrix of sin(theta), first col = 1 (n x k))
    Yl    = latent Y matrix--projection of direction vector onto unit hypersphere
           in Euclidean space. (n * k)
    alpha = vector of shape parameters for underlying gamma distributions
    beta  = vector of rate parameters for underlying gamma distributions
    """
    A = alpha.sum()
    B = (beta * Yl) * sum(axis = 1)
    asinv = np.cumsum(alpha[:1:-1])[::-1]
    lp = (
        + lgamma(A)
        - A * log(B)
        + (alpha * log(beta) - lgamma(alpha)) * sum()
        + (lcoss[:,:k] * (alpha[:,:k] - 1)).sum(axis = 1)
        + (lsins[:,1:] * (asinv - 1)).sum(axis = 1)
        )
    return lp

def dprojgamma(theta, alpha, beta, logd = False):
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

def dprojgamma_trig(s_theta, c_theta, alpha, beta, logd = False):
    coss = np.vstack((c_theta.T, 1)).T
    sins = np.vstack((1, s_theta.T)).T
    sinp = np.cumprod(sins, axis = 1)

    ld = logdprojgamma(coss, sins, sinp, alpha, beta)

    if logd:
        return ld
    else:
        return exp(ld)

def dprojgamma_latent(Y, alpha, beta):
    pass

## Function for sampling from projected gamma

def rprojgamma():
    pass

## Functions related to sampling for parameters from posterior, assuming
## a projected gamma likelihood.

@lru_cache(max.size = 32)
def log_post_log_alpha_1(log_alpha_1, y_1, prior):
    """ Log posterior for log-alpha_1 assuming a gamma distribution,
    with beta assumed to be 1. """
    alpha_1 = exp(log_alpha_1)
    n_1     = length(y_1)
    lp = (
        + prior.a * log_alpha_1
        - prior.b * alpha_1
        + (alpha_1 - 1) * sum(log(y_1))
        - n_1 * lgamma(alpha_1)
        )
    return lp

def sample_alpha_1_mh(curr_alpha_1, y_1, prior, proosal_sd = 0.1):
    """ Sampling function for shape parameter, with gamma likelihood and
    gamma prior.  Assumes rate parameter = 1.  uses Metropolis Hastings
    algorithm with random walk for sampling. """
    if len(y) == 1:
        return gamma.rvs(prior.a, scale = 1./prior.b)

    curr_log_alpha_1 = log(curr_alpha_1)
    prop_log_alpha_1 = curr_log_alpha_1 + norm.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha_1(curr_log_alpha_1, y_1, prior)
    prop_lp = log_post_log_alpha_1(prop_log_alpha_1, y_1, prior)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_1)
    else:
        return curr_alpha_1

@lru_cache(max.size = 128)
def log_post_log_alpha(log_alpha, y, prior):
    """ Log posterior for log-alpha assuming a gamma distribution,
    beta integrated out of the posterior. """
    alpha = exp(log_alpha)
    n     = length(y)
    lp = (
        + (alpha - 1) * sum(log(y))
        - n * lgamma(alpha)
        + prior.a * log_alpha
        - prior.b * alpha
        + lgamma(n * a + prior.c)
        - (n * a + prior.c) * log(sum(y) * prior.d)
        )
    return lp

def sample_alpha_k_mh(curr_alpha, y, prior, proposal_sd = 0.1):
    """ Sampling Function for shape parameter, with Gamma likelihood and Gamma
    prior, with rate (with gamma prior) integrated out. """
    if len(y) == 1:
        return gamma.rvs(prior.a, scale = 1./prior.b)

    curr_log_alpha_k = log(curr_alpha)
    prop_log_alpha = curr_log_alpha + norm.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha(curr_log_alpha, y, prior)
    prop_lp = log_post_log_alpha(prop_log_alpha, y, prior)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha)
    else:
        return curr_alpha

def sample_beta_fc(alpha, y, prior):
    aa = len(y) * alpha + prior.c
    bb = sum(y) + prior.d
    return gamma.rvs(aa, scale = 1. / bb)

# EOF
