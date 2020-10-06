""" Functions relating to density of Projected Gamma.  All functions are
parameterized such that E(x) = alpha / beta (treat beta as rate parameter). """

import numpy as np
from numpy.linalg import norm
from math import cos, sin, log, acos, exp
from scipy.stats import gamma, uniform, norm as normal
from functools import lru_cache
from collections import namedtuple
cimport scipy.special.cython_special as csp
cimport numpy as np
from cSlice import univariate_slice_sample #, f_type_1

# Tuples for storing priors

# GammaPrior     = namedtuple('GammaPrior', 'a b')
# DirichletPrior = namedtuple('DirichletPrior', 'a')
cdef class GammaPrior:
    cdef double a, b

    @property
    def a(self):
        return self.a

    @property
    def b(self):
        return self.b

    def __init__(self, a, b):
        self.a = a
        self.b = b
        return

cdef class DirichletPrior:
    cdef double a

    @property
    def a(self):
        return self.a

    def __init__(self, a):
        self.a = a
        return

# Utility functions
cdef double vector_sum(double[:] Y):
    """ sum over one-dimensional vector Y """
    cdef int N = Y.shape[0]
    cdef double s = Y[0]
    cdef int i
    for i in range(1,N):
      s += Y[i]
    return s

cdef double [:] vector_cumsum(double[:] Y):
    """ cumulative sum over one-dimensional vector Y """
    cdef int N = Y.shape[0]
    cdef double s = Y[0]
    cdef int i
    cdef np.ndarray[dtype = np.float_t, ndim = 1] S = np.empty(N)
    S[0] = s
    for i in range(1, N):
        s += Y[i]
        S[i] = s
    return S

cdef double [:,:] hadamard_product(double[:,:] mat1, double[:,:] mat2):
    """ hadamard product for matrix multiplication => Z[i,j] = X[i,j] * Y[i,j] for all i,j """
    assert mat1.shape == mat2.shape
    cdef int n = mat1.shape[0]
    cdef int k = mat1.shape[1]
    cdef np.ndarray[dtype = np.float_t, ndim = 2] out = np.empty(mat1.shape)
    for i in range(n):
        for j in range(k):
            out[i,j] = mat1[i,j] * mat2[i,j]
    return out

cdef double [:,:] dot_product(double[:,:] mat1, double[:,:] mat2):
    assert mat1.shape[1] == mat2.shape[0]
    cdef int n = mat1.shape[0]
    cdef int m = mat2.shape[1]
    cdef int o = mat1.shape[1]
    cdef double s
    cdef np.ndarray[dtype = np.float_t, ndim = 2] out = np.zeros((n, m))
    for i in range(n):
       for j in range(m):
          s = 0.
          for k in range(o):
              s += mat1[i,k] * mat2[k,j]
          out[i,j] = s
    return out

cdef double [:] dot_product_s(double[:,:] mat, double[:] vec):
    assert mat.shape[1] == vec.shape[0]
    cdef int n = mat.shape[0]
    cdef int o = mat.shape[1]
    cdef double s
    cdef np.ndarray[dtype = np.float_t, ndim = 1] out = np.zeros(n)
    for i in range(n):
        s = 0
        for k in range(o):
            s += mat[i,k] * vec[k]
        out[i] = s
    return out

## Functions related to projected gamma density

cpdef double logdprojgamma(
        np.ndarray[dtype = np.float_t, ndim = 1] coss,
        np.ndarray[dtype = np.float_t, ndim = 1] sins,
        np.ndarray[dtype = np.float_t, ndim = 1] sinp,
        np.ndarray[dtype = np.float_t, ndim = 1] alpha,
        np.ndarray[dtype = np.float_t, ndim = 1] beta,
        ):
    """ Log-density of projected gamma.  Inputs have already been
    pre-formatted for use.
    coss = matrix of cos(theta), last col = 1  (n x k)
    sins = matrix of sin(theta), first col = 1 (n x k)
    sinp = cumulative product of sins, by column (n x k)
    alpha = vector of shape parameters for underlying gamma distributions
    beta  = vector of rate parameters for underlying gamma distributions
    """
    cdef np.ndarray[dtype = np.float_t, ndim = 1] Yl
    cdef np.ndarray[dtype = np.float_t, ndim = 1] B, asinv
    cdef double A, lp
    cdef int k

    Yl    = coss * sinp
    A     = alpha.sum()
    B     = Yl @ beta
    asinv = np.cumsum(alpha[1:][::-1])[::-1]
    k     = Yl.shape[1]

    lp = (
        + csp.gammaln(A)
        - A * log(B)
        + (alpha * log(beta) - csp.gammaln(alpha)).sum()
        + (log(coss[:,:(k-1)]) * (alpha[:(k-1)] - 1)).sum(axis = 1)
        + (log(sins[:,1:]) * (asinv - 1)).sum(axis = 1)
        )
    return lp

cpdef double logdprojgamma_pre_single(
        double[:] lcoss,
        double[:] lsins,
        double[:] Yl,
        double[:] alpha,
        double[:] beta
        ):
    """ Log density of projected gamma for a single (k-1) dimensional obsv. """
    cdef np.ndarray[dtype = np.float_t, ndim = 1] B, asinv
    cdef double A, lp, line1, line2, line3, line4, line5
    cdef int i
    cdef int k = Yl.shape[1]
    A = vector_sum(alpha)
    B = 0.
    for j in range(k):
        B += Yl[j] * beta[j]
    asinv = vector_cumsum(alpha[1:][::-1])[::-1]
    line1 = csp.gammaln(A)
    line2 = - A * log(B)
    line3 = 0
    line4 = 0
    line5 = 0
    for i in range(k):
       line3 += alpha[i] * log(beta[i]) - csp.gammaln(alpha[i])
    for i in range(k - 1):
       line4 += lcoss[i] * (alpha[i] - 1)
       line5 += lsins[i + 1] * (asinv[i] - 1)
    return line1 + line2 + line3 + line4 + line5

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
    cdef double A, lp, line1, line2, line3, line4, line5
    cdef int k
    A = vector_sum(alpha)
    B = dot_product_s(Yl, beta)
    asinv = vector_cumsum(alpha[1:][::-1])[::-1]
    k = lcoss.shape[1]
    lp = (
        + csp.gammaln(A)
        - A * log(B)
        + (alpha * np.log(beta) - csp.gammaln(alpha)).sum()
        + (lcoss[:,:(k - 1)] * (alpha[:(k - 1)] - 1)).sum()
        + (lsins[:,1:] * (asinv - 1)).sum(axis = 1)
        )
    return lp

# cpdef double dprojgamma(
#         np.ndarray[dtype = np.float64_t, ndim = 2] theta,
#         np.ndarray[dtype = np.float64_t, ndim = 1] alpha,
#         np.ndarray[dtype = np.float64_t, ndim = 1] beta,
#         bint logd = False,
#         ):
#     cdef int k
#     cdef np.ndarray[dtype = np.float64_t, ndim = 2] coss, sins, sinp
#     cdef double ld
#
#     k = len(alpha)
#     assert all(len(theta) == k - 1, len(beta) == k)
#
#     coss = np.vstack((np.cos(theta).T, 1)).T
#     sins = np.vstack((1,np.sin(theta).T)).T
#     sinp = np.cumprod(sins, axis = 1)
#
#     ld = logdprojgamma(coss, sins, sinp, alpha, beta)
#
#     if logd:
#         return ld
#     else:
#         return exp(ld)
#
# cpdef double dprojgamma_trig(
#         np.ndarray[dtype = np.float64_t, ndim = 2] s_theta,
#         np.ndarray[dtype = np.float64_t, ndim = 2] c_theta,
#         np.ndarray[dtype = np.float64_t, ndim = 1] alpha,
#         np.ndarray[dtype = np.float64_t, ndim = 1 ]beta,
#         bint logd = False
#         ):
#     cdef np.ndarray[dtype = np.float64_t, ndim = 2] coss, sins, sinp
#     cdef double ld
#
#     coss = np.vstack((c_theta.T, 1)).T
#     sins = np.vstack((1, s_theta.T)).T
#     sinp = np.cumprod(sins, axis = 1)
#
#     ld = logdprojgamma(coss, sins, sinp, alpha, beta)
#
#     if logd:
#         return ld
#     else:
#         return exp(ld)

## Functions related to sampling for parameters from posterior, assuming
## a projected gamma likelihood.
cpdef double log_post_log_alpha_1(double log_alpha_1, double [:] y_1, GammaPrior prior):
    """ Log posterior for log-alpha_1 assuming a gamma distribution,
    with beta assumed to be 1. """
    cdef double alpha_1, lp
    cdef int n_1
    alpha_1 = exp(log_alpha_1)
    n_1     = y_1.shape[0]
    lp = (
        + prior.a * log_alpha_1
        - prior.b * alpha_1
        + (alpha_1 - 1) * np.log(y_1).sum()
        - n_1 * csp.gammaln(alpha_1)
        )
    return lp

cpdef double sample_alpha_1_mh(
        double curr_alpha_1,
        double [:] y_1,
        GammaPrior prior,
        double proposal_sd = 0.1,
        ):
    """ Sampling function for shape parameter, with gamma likelihood and
    gamma prior.  Assumes rate parameter = 1.  uses Metropolis Hastings
    algorithm with random walk for sampling. """
    cdef double curr_log_alpha_1, prop_log_alpha_1, curr_lp, prop_lp

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
