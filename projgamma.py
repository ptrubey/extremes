""" Functions relating to density of Projected Gamma.  All functions are
parameterized such that E(x) = alpha / beta (treat beta as rate parameter). """

import numpy as np
np.seterr(under = 'ignore', over = 'raise')
from numpy.linalg import norm
from math import cos, sin, log, acos, exp
from scipy.stats import gamma, uniform, norm as normal
from scipy.special import gammaln
from functools import lru_cache
from genpareto import gpd_fit
from collections import namedtuple
from slice import univariate_slice_sample, skip_univariate_slice_sample

# Tuples for storing priors

GammaPrior     = namedtuple('GammaPrior', 'a b')
DirichletPrior = namedtuple('DirichletPrior', 'a')
BetaPrior      = namedtuple('BetaPrior', 'a b')

def to_angular(hyp):
    """ Convert data to angular representation. """
    n, k  = hyp.shape
    theta = np.empty((n, k - 1))
    hyp = np.maximum(hyp, 1e-7)
    for i in range(k - 1):
        theta[:,i] = np.arccos(hyp[:,i] / (norm(hyp[:,i:], axis = 1)))
    return theta

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
    asinv = np.cumsum(alpha[1:][::-1])[::-1]
    lp = (
        + gammaln(A)
        - A * log(B)
        + (alpha * log(beta) - gammaln(alpha)).sum()
        + (log(coss[:,:(k-1)]) * (alpha[:(k-1)] - 1)).sum(axis = 1)
        + (log(sins[:,1:]) * (asinv - 1)).sum(axis = 1)
        )
    return lp

def logdprojgamma_pre_single(lcoss, lsins, Yl, alpha, beta):
    A = alpha.sum()
    B = (Yl * beta).sum()
    k = Yl.shape[0]
    asinv = np.cumsum(alpha[1:][::-1])[::-1]
    lp = (
        + gammaln(A)
        - A * log(B)
        + (alpha * np.log(beta) - gammaln(alpha)).sum()
        + (lcoss[:(k-1)] * (alpha[:(k-1)] - 1)).sum()
        + (lsins[1:] * (asinv - 1)).sum()
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
    # if only a single row, then send it down the single chute.
    if len(Yl.shape) == 1:
        return np.array([logdprojgamma_pre_single(lcoss, lsins, Yl, alpha, beta)])

    # otherwise continue on, calculate logdprojgamma for all
    A = alpha.sum()
    B = (Yl * beta).sum(axis = 1)
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

def rprojgamma(alpha, beta):
    gammas = gamma(alpha, scale = 1/beta).rvs()
    thetas = to_angular(gammas)
    return thetas

## Functions related to sampling for parameters from posterior, assuming
## a projected gamma likelihood.

# @lru_cache(maxsize = 32)
def log_post_log_alpha_1(log_alpha_1, y_1, prior):
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

def sample_alpha_1_mh(curr_alpha_1, y_1, prior, proposal_sd = 0.3):
    """ Sampling function for shape parameter, with gamma likelihood and
    gamma prior.  Assumes rate parameter = 1.  uses Metropolis Hastings
    algorithm with random walk for sampling. """
    if len(y_1) < 1:
        return gamma.rvs(prior.a, scale = 1./prior.b)

    curr_log_alpha_1 = log(curr_alpha_1)
    prop_log_alpha_1 = curr_log_alpha_1 + normal.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha_1(curr_log_alpha_1, y_1, prior)
    prop_lp = log_post_log_alpha_1(prop_log_alpha_1, y_1, prior)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_1)
    else:
        return curr_alpha_1

def sample_alpha_1_slice(curr_alpha_1, y_1, prior, increment_size = 2.):
    f = lambda log_alpha_1: log_post_log_alpha_1(log_alpha_1, y_1, prior)
    return exp(univariate_slice_sample(f, log(curr_alpha_1), increment_size))

#@lru_cache(maxsize = 128)
def log_post_log_alpha_k(log_alpha, y, prior_a, prior_b):
    """ Log posterior for log-alpha assuming a gamma distribution,
    beta integrated out of the posterior. """
    alpha = exp(log_alpha)
    n  = y.shape[0]
    lp = (
        + (alpha - 1) * np.log(y).sum()
        - n * gammaln(alpha)
        + prior_a.a * log_alpha
        - prior_a.b * alpha
        + gammaln(n * alpha + prior_b.a)
        - (n * alpha + prior_b.a) * log(y.sum() + prior_b.b)
        )
    return lp

def sample_alpha_k_mh(curr_alpha_k, y_k, prior_a, prior_b, proposal_sd = 0.3):
    """ Sampling Function for shape parameter, with Gamma likelihood and Gamma
    prior, with rate (with gamma prior) integrated out. """
    if len(y_k) <= 1:
        return gamma.rvs(prior_a.a, scale = 1./prior_a.b)

    curr_log_alpha_k = log(curr_alpha_k)
    prop_log_alpha_k = curr_log_alpha_k + normal.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha_k(curr_log_alpha_k, y_k, prior_a, prior_b)
    prop_lp = log_post_log_alpha_k(prop_log_alpha_k, y_k, prior_a, prior_b)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_k)
    else:
        return curr_alpha_k

def sample_alpha_k_slice(curr_alpha_k, y_k, prior_a, prior_b, increment_size = 2.):
    f = lambda log_alpha_k: log_post_log_alpha_k(log_alpha_k, y_k, prior_a, prior_b)
    return exp(univariate_slice_sample(f, log(curr_alpha_k), increment_size))

def sample_beta_fc(alpha, y, prior):
    aa = len(y) * alpha + prior.a
    bb = sum(y) + prior.b
    return gamma.rvs(aa, scale = 1. / bb)

def density_projgamma_hypercube(y, alpha, beta):
    ld = (
        + gammaln(alpha.sum())
        - alpha.sum() * np.log(beta * y).sum()
        + (alpha * np.log(beta)).sum()
        - gammaln(alpha).sum()
        + ((alpha - 1) * np.log(y)).sum()
        )
    return exp(ld)

# EOF
