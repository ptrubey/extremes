from lib2to3.pytree import Base
from numpy.random import choice, gamma, uniform, normal
from collections import namedtuple
from itertools import repeat
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
import sqlite3 as sql
from math import ceil, log
from scipy.special import gammaln

import cUtility as cu
from samplers import BaseSampler
from cProjgamma import sample_alpha_k_mh_summary, sample_alpha_1_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube, euclidean_to_simplex, Categorical
from projgamma import GammaPrior

class Samples(object):
    zeta  = None
    sigma = None
    rho   = None
    
    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = np.empty((nSamp + 1, nCol))
        self.sigma = np.empty((nSamp + 1, nCol))
        self.rho   = np.empty((nSamp +1, nDat, nCol))
        return
    
    pass

Prior = namedtuple('Prior', 'zeta sigma')

def log_post_log_zeta_1(lzeta, rho, lrho, W, alpha, beta):
    """
    lzeta : (1)
    Y, lY : (n)
    W     : (n)
    alpha : scalar
    beta  : scalar
    cmat  : (n x J), bool
    """
    zeta = np.exp(lzeta)
    ZW = zeta + W # (n)
    lp = np.zeros(lzeta.shape)
    lp += ((ZW - 1) * lrho).sum()
    lp -= gammaln(ZW).sum()
    lp += alpha * lzeta
    lp -= zeta * beta
    return lp

def log_post_log_zeta_k(lzeta, rho, lrho, W, alpha, beta, xi, tau):
    """
    lzeta : (J)
    Y, lY : (n)
    W     : (n)
    alpha : scalar
    beta  : scalar
    cmat  : (n x J), bool
    """
    zeta = np.exp(lzeta)
    ZW = zeta + W # (n) array
    lp = np.zeros(lzeta.shape) # should be 1d?
    lp += ((ZW - 1) * lrho).sum()
    lp -= gammaln(ZW).sum()
    lp += alpha * lzeta
    lp -= zeta * beta
    lp += gammaln(ZW.sum() + xi)
    lp -= (ZW.sum() + xi) * np.log(rho.sum() + tau)
    return lp

class Chain(BaseSampler):
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_sigma(self):
        return self.samples.sigma[self.curr_iter]
    @property
    def curr_rho(self):
        return self.samples.rho[self.curr_iter]

    def sample_zeta(self, curr_zeta, rho):
        """
        Metropolis Hastings sampler for zeta
        """
        lrho = np.log(rho)
        lp_curr = np.zeros(curr_zeta.shape)
        lp_prop = np.zeros(curr_zeta.shape)
        # declaring current and proposal RV's
        curr_log_zeta = np.log(curr_zeta)
        prop_log_zeta = curr_log_zeta + normal(size = curr_log_zeta.shape, scale = 0.3)
        # indexing xi, tau to same index as alpha, beta
        for i in range(curr_log_zeta.shape[0]):
            if (i == 0):
                lp_curr[i] = log_post_log_zeta_1(
                    curr_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
                    self.priors.zeta.a, self.priors.zeta.b,
                    )
                lp_prop[i] = log_post_log_zeta_1(
                    prop_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
                    self.priors.zeta.a, self.priors.zeta.b,
                    )
            else: 
                lp_curr[i] = log_post_log_zeta_k(
                    curr_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
                    self.priors.zeta.a, self.priors.zeta.b, self.priors.sigma.a, self.priors.sigma.b,
                ) 
                lp_prop[i] = log_post_log_zeta_k(
                    prop_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
                    self.priors.zeta.a, self.priors.zeta.b, self.priors.sigma.a, self.priors.sigma.b,
                    )
        lp_diff = lp_prop - lp_curr
        keep = np.log(uniform(size = lp_curr.shape)) < lp_diff   # metropolis hastings step

        log_zeta = (prop_log_zeta * keep) + (curr_log_zeta * (~keep)) # if keep, then proposal, else current
        return np.exp(log_zeta)

    def sample_sigma(self, zeta, rho):
        nz = self.nDat * zeta[:1] # (d - 1)
        rsv = rho.sum(axis = 0)[1:] # (d - 1)
        wsv = self.data.W.sum(axis = 0)[1:] # (d - 1)
        sigma = np.ones(zeta.shape)

        shape = nz + wsv + self.priors.sigma.a
        rate  = rsv + self.priors.sigma.b
        sigma[1:] = gamma(shape = shape, scale = 1 / rate)
        return sigma
    
    def sample_rho(self, zeta, sigma):
        As = zeta + self.data.W
        Bs = sigma[None, :]
        rho = gamma(shape = As, scale = 1 / Bs)
        return rho
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = self.nCol)
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = self.nCol)
        self.samples.sigma[0,0] = 1.
        self.samples.rho[0] = self.sample_rho(self.samples.zeta[0], self.samples.sigma[0])
        self.curr_iter = 0
        return

    def iter_sample(self):
        zeta  = self.curr_zeta
        sigma = self.curr_sigma
        rho   = self.curr_rho

        self.curr_iter += 1

        self.samples.zeta[self.curr_iter] = self.sample_zeta(zeta, rho)
        self.samples.sigma[self.curr_iter] = self.sample_sigma(self.curr_zeta, rho)
        self.samples.rho[self.curr_iter] = self.sample_rho(self.curr_zeta, self.curr_sigma)
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.mkdir(folder)
        if os.path.exists(path):
            os.remove(path)
        
        zeta = self.samples.zeta[nBurn::nThin]
        sigma = self.samples.sigma[nBurn::nThin]
        rho   = self.samples.rho[nBurn::nThin]

        out = {
            'zetas'  : zeta,
            'sigmas' : sigma,
            'rhos'   : rho,
            'W'      : self.data.C,
            }
        
        try:
            out['Y'] = self.data.Y
        except AttributeError:
            pass
        
        with open(path, 'wb') as file:
            pickle.dump(out, file)
        
        return

    def __init__(
            self, 
            data, 
            prior_zeta = GammaPrior(0.5, 0.5),
            prior_sigma = GammaPrior(0.5, 0.5),
            ):
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_zeta, prior_sigma)
        return
    
    pass

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1):
        new_gammas = []
        for s in range(self.nSamp):
            g = gamma(
                shape = self.samples.zeta[s], 
                scale = 1 / self.samples.sigma[s], 
                size = (n_per_sample, self.nCol),
                )
            new_gammas.append(g)
        return np.vstack(new_gammas)
    
    def generate_posterior_predictive_hypercube(self, n_per_sample = 1):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_hypercube(gammas)
    
    def generate_posterior_predictive_simplex(self, n_per_sample = 1):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_simplex(gammas)
    
    def load_data(self, path):
        with open(path, 'rb') as file:
            out = pickle.load(file)
        
        zetas  = out['zetas']
        sigmas = out['sigmas']
        rhos   = out['rhos']

        self.nSamp = zetas.shape[0]
        self.nDat  = rhos.shape[1]
        self.nCol  = zetas.shape[1]

        self.data = Categorical(out['W'])
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass
        
        self.samples = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.zeta = zetas
        self.samples.sigma = sigmas
        self.samples.rho = rhos
        return 

    pass

# EOF

if __name__ == '__main__':
    from pandas import read_csv
    from data import Multinomial

    raw = read_csv('./simulated/categorical/test.csv').values
    data = Multinomial(raw)
    model = Chain(data)
    model.sample(50000)
    model.write_to_disk('./simulated/categorical/result.pkl', 20000, 30)
    res = Result('./simulated/categorical/result.pkl')

# EOF
