""" Functions relating to density of Projected Gamma.  All functions are
parameterized such that E(x) = alpha / beta (treat beta as rate parameter). """


cimport scipy.special.cython_special as csp
cimport libc.math as math
from libc.math cimport exp, log, lgamma
cimport scipy.linalg.cython_blas as blas
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
    for i in range(y.shape):
        s += log(y[i])
    return s

## Functions related to projected gamma density

cdef double logdgamma(double[:] Y, double[:] alpha, double[:] beta):
    cdef:
        int i, nCol
        double s = 0.
    nCol = Y.shape[0]
    for i in range(nCol):
        s += alpha[i] * log(beta[i]) - lgamma(alpha[i]) + (alpha[i] - 1) * log(Y[i]) - beta[i] * Y[i]
    return s

## Functions related to sampling for parameters from posterior, assuming
## a projected gamma likelihood.
cdef double log_post_log_alpha_1(double log_alpha_1, double [:] y_1, double a, double b):
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

    assert y_1.shape[0] > 0

    curr_log_alpha_1 = log(curr_alpha_1)
    prop_log_alpha_1 = curr_log_alpha_1 + normal.rvs(scale = proposal_sd)

    curr_lp = log_post_log_alpha_1(curr_log_alpha_1, y_1, a, b)
    prop_lp = log_post_log_alpha_1(prop_log_alpha_1, y_1, a, b)

    if log(uniform.rvs()) < prop_lp - curr_lp:
        return exp(prop_log_alpha_1)
    else:
        return curr_alpha_1

cdef double _sample_alpha_1_slice(
        double curr_alpha_1,
        double[:] y_1,
        GammaPrior prior,
        double increment_size
        ):
    cdef object f = lambda log_alpha_1: log_post_log_alpha_1(log_alpha_1, y_1, prior)
    return exp(univariate_slice_sample(f, log(curr_alpha_1), increment_size))

cpdef double sample_alpha_1_slice(
        double curr_alpha_1,
        double[:] y_1,
        GammaPrior prior,
        double increment_size = 0.5
        ):
    return _sample_alpha_1_slice(curr_alpha_1, y_1, prior, increment_size)

cpdef double log_post_log_alpha_k(
        double log_alpha_k,
        double[:] y_k,
        a, b, c, d,
        GammaPrior prior_a,
        GammaPrior prior_b,
        ):
    """ Log posterior for log-alpha assuming a gamma distribution,
    beta integrated out of the posterior. """
    cdef double alpha_k, lp
    cdef int n_k

    alpha_k = exp(log_alpha_k)
    n_k     = y_k.shape[0]
    lp = (
        + (alpha_k - 1) * np.log(y_k).sum()
        - n_k * csp.gammaln(alpha_k)
        + prior_a.a * log_alpha_k
        - prior_a.b * alpha_k
        + csp.gammaln(n_k * alpha_k + prior_b.a)
        - (n_k * alpha_k + prior_b.a) * log(y_k.sum() + prior_b.b)
        )
    return lp

cpdef double sample_alpha_k_mh(
        double curr_alpha_k,
        np.ndarray[dtype = np.float64_t, ndim = 1] y_k,
        GammaPrior prior_a,
        GammaPrior prior_b,
        double proposal_sd = 0.1
        ):
    """ Sampling Function for shape parameter, with Gamma likelihood and Gamma
    prior, with rate (with gamma prior) integrated out. """
    cdef double curr_log_alpha_k, prop_log_alpha_k, curr_lp, prop_lp
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

cdef double _sample_alpha_k_slice(
        double curr_alpha_k,
        double[:] y_k,
        GammaPrior prior_a,
        GammaPrior prior_b,
        double increment_size
        ):
    cdef object f = lambda log_alpha_k: log_post_log_alpha_k(log_alpha_k, y_k, prior_a, prior_b)
    return exp(univariate_slice_sample(f, log(curr_alpha_k), increment_size))

cpdef double sample_alpha_k_slice(
        double curr_alpha_k,
        double[:] y_k,
        GammaPrior prior_a,
        GammaPrior prior_b,
        double increment_size = 0.5
        ):
    return _sample_alpha_k_slice(curr_alpha_k, y_k, prior_a, prior_b, increment_size)

cpdef double sample_beta_fc(double alpha, double[:] y, GammaPrior prior):
    cdef double aa, bb
    aa = y.shape[0] * alpha + prior.a
    bb = vector_sum(y) + prior.b
    return gamma.rvs(aa, scale = 1. / bb)

# EOF
