from numpy.random import choice, gamma, beta, uniform, normal, lognormal
from collections import namedtuple, deque
from itertools import repeat
import numpy as np
np.seterr(divide = 'raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log
from scipy.special import gammaln, digamma
from io import BytesIO
from cUtility import pityor_cluster_sampler, generate_indices
from samplers import DirichletProcessSampler
from data import euclidean_to_hypercube, Data_From_Sphere
from projgamma import GammaPrior, logd_projgamma_my_mt_inplace_unstable


Prior = namedtuple('Prior','alpha beta')
class Samples(object):
    r  : deque      # radius (projected gamma)
    nu : deque      # stick-breaking weights (unnormalized)
    delta : deque   # cluster identifiers
    beta  : deque   # rate hyperparameter
    
    def __init__(self, nkeep : int):
        self.r     = deque([], maxlen = nkeep)
        self.nu    = deque([], maxlen = nkeep)
        self.delta = deque([], maxlen = nkeep)
        return
    pass

def gradient_resgammagamma_ln(
        theta   : np.ndarray,  # np.stack((mu, tau))
        lYs     : np.ndarray,  # sum of log(Y)
        n       : np.ndarray,  # number of observations
        a       : np.ndarray,  # hierarchical shape 
        b       : np.ndarray,  # hierarchical rate
        ns = 10,
        ):
    epsilon = normal(size = (ns, *theta.shape[1:]))
    ete = np.exp(theta[1]) * epsilon
    alpha = np.exp(theta[0] + ete)

    dtheta = np.zeros(theta.shape)
    dtheta += alpha * lYs
    dtheta -= n * digamma(alpha) * alpha
    dtheta += (a - 1)
    dtheta -= b * alpha
    
    dtheta[1] *= ete
    
    dtheta[0] -= -1
    dtheta[1] -= -1 - np.exp(2 * theta[1])
    return dtheta.mean(axis = 0)

def gradient_gammagamma_ln(
        theta   : np.ndarray,  # np.stack((mu, tau))
        lYs     : np.ndarray,  # sum of log(Y)
        Ys      : np.ndarray,  # sum of Y
        n       : np.ndarray,  # number of observations
        a       : np.ndarray,  # hierarchical (shape) shape 
        b       : np.ndarray,  # hierarchical (shape) rate
        c       : np.ndarray,  # hierarchical (rate) shape
        d       : np.ndarray,  # hierarchical (rate) rate
        ns = 10,
        ):
    epsilon = normal(size = (ns, *theta.shape[1:]))
    ete = np.exp(theta[1]) * epsilon
    alpha = np.exp(theta[0] + ete)

    dtheta = np.zeros(theta.shape)
    dtheta += alpha * lYs
    dtheta -= n * digamma(alpha) * alpha
    dtheta += (a - 1)
    dtheta -= b * alpha
    dtheta += digamma(n * alpha + c)
    dtheta -= (n * alpha + c) * np.log(Ys + d)
    
    dtheta[1] *= ete
    
    dtheta[0] -= -1
    dtheta[1] -= -1 - np.exp(2 * theta[1])
    return dtheta.mean(axis = 0)

class VariationalParameters(object):
    zeta_mutau   : np.ndarray
    alpha_mutau  : np.ndarray

    def __init__(self, J : int, S : int):
        zeta_mutau = normal(size = (2, J, S))
        alpha_mutau = normal(size = (2, S))
        return
    pass

class Chain(DirichletProcessSampler):
    concentration   : float
    discount        : float

    @property
    def curr_r(self):
        return self.samples.r[-1]
    @property
    def curr_nu(self):
        return self.samples.nu[-1]
    @property
    def curr_delta(self):
        return self.samples.delta[-1]
    @property
    def curr_alpha(self):
        return lognormal(self.vp.alpha_mu, scale = np.exp(self.vp.alpha_tau))
    @property
    def curr_zeta(self):
        return lognormal(self.vp.zeta_mu, scale = np.exp(self.vp.zeta_tau))
    
    def update_zeta(self):

        pass
    def update_alpha(self):
        pass
    def sample_beta(self):
        pass
    def sample_r(self, zeta, delta):
        pass
    def sample_chi(self, delta):
        pass
    def sample_delta(self, zeta, chi):
        pass

    def iter_sample(self):
        alpha = self.curr_alpha
        zeta  = self.curr_zeta

    def initialize_sampler(self):

    def __init__(
            self, 
            data, 
            variational_samples = 1, 
            variational_iterations_per = 10,
            iters = 10000,  # number of iterations to run the sampler for
            ):
        self.data = data
        self.initialize_sampler(self):
        return
    