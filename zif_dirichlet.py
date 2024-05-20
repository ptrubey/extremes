"""
zif_dirichlet.py

zero-inflated Dirichlet Distribution
"""
import numpy as np
from scipy.special import loggamma

EPS = 1e-6

def logd_my_mt_zif_dirichlet_1(
        Y     : np.ndarray,  # (n * d)
        nu    : np.ndarray,  # (j * d)
        alpha : np.ndarray   # (j * d)
        ):
    """
    Zero-inflated Dirichlet with dimension activation flags
    """
    N, D = Y.shape
    J    = alpha.shape[0]
    AlphaNew = nu @ alpha.T - nu @ np.ones(D) + D

    logd = np.zeros((N, J))
    logd -= (nu @ loggamma(alpha).T)[None] # (1 * J)
    logd += [None] # (1 * J)

def logd_my_mt_zif_dirichlet_2(
        Y     : np.ndarray, # (n * d)
        pi    : np.ndarray, # (j * d)
        alpha : np.ndarray, # (j * d)
        ):
    """
    Zero-inflated Dirichlet with dimension activation probabilities
    """
