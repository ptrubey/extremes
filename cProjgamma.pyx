# cython: language_level=3, boundscheck=False

""" Functions relating to density of Projected Gamma.  All functions are
parameterized such that E(x) = alpha / beta (treat beta as rate parameter). """

from libc.math cimport exp, log, lgamma
from scipy.special.cython_special cimport gamma, gammaincinv
from numpy.random import uniform, normal, gamma as _gamma
cimport numpy as np

# Utility Functions
cdef double sum(double[:] y):
    cdef:
        int i
        double s = 0.
    for i in range(y.shape[0]):
        s += y[i]
    return s

cdef double sumlog(double[:] y):
    cdef:
        int i
        double s = 0.
    for i in range(y.shape[0]):
        s += log(y[i])
    return s

cdef double sumlgamma(double[:] y):
    cdef:
        int i
        double s = 0.
    for i in range(y.shape[0]):
        s += lgamma(y[i])
    return s

cdef double gamma_rvs(double shape, double rate):
    # return gammaincinv(shape, uniform() * gamma(shape)) / rate
    return _gamma(shape = shape, scale = 1. / rate)

## Functions related to projected gamma density

cpdef double logdgamma(double[:] Y, double[:] alpha, double[:] beta):
    "logsum of d independent gamma random variables, with "
    cdef:
        int i, nCol
        double s = 0.
    nCol = Y.shape[0]
    for i in range(nCol):
        s += alpha[i] * log(beta[i]) - lgamma(alpha[i]) + (alpha[i] - 1) * log(Y[i]) - beta[i] * Y[i]
    return s

cpdef double logdgamma_restricted(double[:] Y, double[:] alpha):
    cdef:
        int i
        double s = 0.
    for i in range(Y.shape[0]):
        s += (alpha[i] - 1) * log(Y[i]) - Y[i] - lgamma(alpha[i])
    return s

cpdef double logddirichlet(double[:] X, double[:] alpha):
    cdef:
        int i
        double s = 0.
    for i in range(X.shape[0]):
        s += (alpha[i] - 1) * log(X[i])
    return s + lgamma(sum(alpha)) - sumlgamma(alpha)

## Functions related to sampling for parameters from posterior, assuming
## a projected gamma likelihood.
cpdef double log_post_log_alpha_1(
        double log_alpha_1,
        double [:] y_1,
        double a, double b,
        ):
    """ Log posterior for log-alpha_1 assuming a gamma distribution,
    with beta assumed to be 1. """
    cdef:
        double alpha_1, lp
        int n_1
    alpha_1 = exp(log_alpha_1)
    n_1     = y_1.shape[0]
    lp = (
        + a * log_alpha_1
        - b * alpha_1
        + (alpha_1 - 1) * sumlog(y_1)
        - n_1 * lgamma(alpha_1)
        )
    return lp

cpdef double sample_alpha_1_mh(
        double curr_alpha_1,
        double [:] y_1,
        double a, double b,
        double proposal_sd = 0.3,
        ):
    """ Sampling function for shape parameter, with gamma likelihood and
    gamma prior.  Assumes rate parameter = 1.  uses Metropolis Hastings
    algorithm with random walk for sampling. """
    cdef:
        double curr_log_alpha_1, prop_log_alpha_1, curr_lp, prop_lp

    if y_1.shape[0] < 1:
        return gamma_rvs(a, b)

    curr_log_alpha_1 = log(curr_alpha_1)
    prop_log_alpha_1 = curr_log_alpha_1 + normal(scale = proposal_sd)

    curr_lp = log_post_log_alpha_1(curr_log_alpha_1, y_1, a, b)
    prop_lp = log_post_log_alpha_1(prop_log_alpha_1, y_1, a, b)

    if log(uniform()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_1)
    else:
        return curr_alpha_1

# cdef double _sample_alpha_1_slice(
#         double curr_alpha_1,
#         double[:] y_1,
#         GammaPrior prior,
#         double increment_size
#         ):
#     cdef object f = lambda log_alpha_1: log_post_log_alpha_1(log_alpha_1, y_1, prior)
#     return exp(univariate_slice_sample(f, log(curr_alpha_1), increment_size))

# cpdef double sample_alpha_1_slice(
#         double curr_alpha_1,
#         double[:] y_1,
#         GammaPrior prior,
#         double increment_size = 0.5
#         ):
#     return _sample_alpha_1_slice(curr_alpha_1, y_1, prior, increment_size)

cpdef double log_post_log_alpha_k(
        double log_alpha_k,
        double[:] y_k,
        double a, double b, double c, double d,
        ):
    """ Log posterior for log-alpha assuming a gamma distribution,
    beta integrated out of the posterior. """
    cdef double alpha_k, lp
    cdef int n_k

    alpha_k = exp(log_alpha_k)
    n_k     = y_k.shape[0]
    lp = (
        + (alpha_k - 1) * sumlog(y_k)
        - n_k * lgamma(alpha_k)
        + a * log_alpha_k
        - b * alpha_k
        + lgamma(n_k * alpha_k + c)
        - (n_k * alpha_k + c) * log(sum(y_k) + d)
        )
    return lp

cpdef double log_post_log_alpha_k_summary(
        double log_alpha_k,
        int n_k, double y_ksum, double ly_ksum,
        double a, double b, double c, double d,
        ):
    """ log posterior for log alpha, assuming gamma priors, with rate
    integrated out, and using summary statistics. """
    alpha_k = exp(log_alpha_k)
    lp = (
        + (alpha_k - 1) * ly_ksum
        - n_k * lgamma(alpha_k)
        + a * log_alpha_k
        - b * alpha_k
        + lgamma(n_k * alpha_k + c)
        - (n_k * alpha_k + c) * log(y_ksum + d)
        )
    return lp

cpdef double sample_alpha_k_mh(
        double curr_alpha_k,
        double[:] y_k,
        double a, double b, double c, double d,
        double proposal_sd = 0.3
        ):
    """ Sampling Function for shape parameter, with Gamma likelihood and Gamma
    prior, with rate (with gamma prior) integrated out. """
    cdef double curr_log_alpha_k, prop_log_alpha_k, curr_lp, prop_lp

    if y_k.shape[0] < 1:
        return gamma_rvs(a, b)

    curr_log_alpha_k = log(curr_alpha_k)
    prop_log_alpha_k = curr_log_alpha_k + normal(scale = proposal_sd)

    curr_lp = log_post_log_alpha_k(curr_log_alpha_k, y_k, a, b, c, d)
    prop_lp = log_post_log_alpha_k(prop_log_alpha_k, y_k, a, b, c, d)

    if log(uniform()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_k)
    else:
        return curr_alpha_k

cpdef double sample_alpha_k_mh_summary(
        double curr_alpha_k,
        int n_k,
        double y_ksum,
        double ly_ksum,
        double a, double b, double c, double d,
        double proposal_sd = 0.3,
        double invtemp = 1.
        ):
    """
    Sampling function for shape parameter, 
    with Gamma likelihood and gamma prior,
    with rate (with gamma prior) integrated out, 
    using summary statistics
    """
    cdef double curr_log_alpha_k, prop_log_alpha_k, curr_lp, prop_lp

    curr_log_alpha_k = log(curr_alpha_k)
    prop_log_alpha_k = curr_log_alpha_k + normal(scale = proposal_sd)
    curr_lp = log_post_log_alpha_k_summary(curr_log_alpha_k, n_k, y_ksum, ly_ksum, a, b, c, d)
    prop_lp = log_post_log_alpha_k_summary(prop_log_alpha_k, n_k, y_ksum, ly_ksum, a, b, c, d)

    if log(uniform()) < (prop_lp - curr_lp) * invtemp:
        return exp(prop_log_alpha_k)
    else: 
        return curr_alpha_k
 
# cdef double _sample_alpha_k_slice(
#         double curr_alpha_k,
#         double[:] y_k,
#         GammaPrior prior_a,
#         GammaPrior prior_b,
#         double increment_size
#         ):
#     cdef object f = lambda log_alpha_k: log_post_log_alpha_k(log_alpha_k, y_k, prior_a, prior_b)
#     return exp(univariate_slice_sample(f, log(curr_alpha_k), increment_size))

# cpdef double sample_alpha_k_slice(
#         double curr_alpha_k,
#         double[:] y_k,
#         GammaPrior prior_a,
#         GammaPrior prior_b,
#         double increment_size = 0.5
#         ):
#     return _sample_alpha_k_slice(curr_alpha_k, y_k, prior_a, prior_b, increment_size)

cpdef double sample_beta_fc(
        double alpha,
        double[:] y,
        double c,
        double d,
        ):
    cdef double aa, bb
    aa = y.shape[0] * alpha + c
    bb = sum(y) + d
    # return # gamma.rvs(aa, scale = 1. / bb)
    return gamma_rvs(aa, bb)

cpdef double sample_beta_fc_summary(
        double alpha,
        double ys,
        double n, 
        double c,
        double d,
        ):
    return gamma_rvs(n * alpha + c, ys + d)
# EOF
