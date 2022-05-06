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

def log_post_log_zeta_1(lzeta, Y, lY, W, alpha, beta, cmat):
    """
    lzeta : (J)
    Y, lY : (n)
    W     : (n)
    alpha : scalar
    beta  : scalar
    cmat  : (n x J), bool
    """
    zeta = np.exp(lzeta)
    ZW = (zeta[:, None] + W[None, :]) # (J x n) array
    lp = np.zeros(lzeta.shape)
    lp += np.einsum('jn,n,nj->j', ZW - 1, lY, cmat)
    lp -= np.einsum('jn,nj->j', gammaln(ZW), cmat)
    lp += alpha * lzeta
    lp -= zeta * beta
    return lp

def log_post_log_zeta_k(lzeta, Y, lY, W, alpha, beta, xi, tau, cmat):
    """
    lzeta : (J)
    Y, lY : (n)
    W     : (n)
    alpha : scalar
    beta  : scalar
    cmat  : (n x J), bool
    """
    zeta = np.exp(lzeta)
    ZW = (zeta[:, None] + W[None, :]) # (J x n) array
    lp = np.zeros(lzeta.shape)
    lp += np.einsum('jn,n,nj->j', ZW - 1, lY, cmat)
    lp -= np.einsum('jn,nj->j', gammaln(ZW), cmat)
    lp += alpha * lzeta
    lp -= zeta * beta
    lp += gammaln(np.einsum('jn,nj->j', ZW, cmat) + xi)
    lp -= (np.einsum('jn,nj->j', ZW, cmat) + xi) * np.log(np.einsum('n,nj->j', Y, cmat) + tau)
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
        return self.samples.r[self.curr_iter]

    def sample_zeta(self, curr_zeta, r, rho, delta, alpha, beta, xi, tau):
        """
        Metropolis Hastings sampler for zeta
        """
        dmat = delta[:,None] == np.arange(delta.max() + 1) # n x J
        Y  = np.hstack((r[:, None] * self.data.Yp, rho)) # n x D
        lY = np.log(Y)
        W  = np.hstack((np.zeros(self.data.Yp.shape), self.data.W))
        # declaring targets for LP
        lp_curr = np.zeros(curr_zeta.shape)
        lp_prop = np.zeros(curr_zeta.shape)
        # declaring current and proposal RV's
        curr_log_zeta = np.log(curr_zeta)
        prop_log_zeta = curr_log_zeta + normal(size = curr_log_zeta.shape, scale = 0.3)
        # indexing xi, tau to same index as alpha, beta
        ixi  = np.ones(alpha.shape)
        itau = np.ones(alpha.shape)
        ixi[~self.sigma_unity] = xi
        itau[~self.sigma_unity] = tau

        for i in range(curr_log_zeta.shape[1]):
            if self.sigma_unity[i]:
                lp_curr.T[i] = log_post_log_zeta_1(
                    curr_log_zeta.T[i], Y.T[i], lY.T[i], W.T[i], alpha[i], beta[i], dmat,
                    )
                lp_prop.T[i] = log_post_log_zeta_1(
                    prop_log_zeta.T[i], Y.T[i], lY.T[i], W.T[i], alpha[i], beta[i], dmat,
                    )
            else:
                lp_curr.T[i] = log_post_log_zeta_k(
                    curr_log_zeta.T[i], Y.T[i], lY.T[i], W.T[i], alpha[i], beta[i], ixi[i], itau[i], dmat,
                    )
                lp_prop.T[i] = log_post_log_zeta_k(
                    prop_log_zeta.T[i], Y.T[i], lY.T[i], W.T[i], alpha[i], beta[i], ixi[i], itau[i], dmat,
                    )
        
        lp_diff = lp_prop - lp_curr
        keep = np.log(uniform(size = lp_curr.shape)) < lp_diff   # metropolis hastings step

        log_zeta = (prop_log_zeta * keep) + (curr_log_zeta * (~keep)) # if keep, then proposal, else current
        return np.exp(log_zeta)

    def sample_sigma(self, zeta, r, rho, delta, xi, tau):
        dmat = delta[:, None] == np.arange(delta.max() + 1)        # (n x J)
        Y = np.hstack((r[:, None] * self.data.Yp, rho))            # (n x d)
        W = np.hstack((np.zeros(self.data.Yp.shape), self.data.W)) # (n x d)

        nZ  = zeta * dmat.sum(axis = 0)[:, None]
        Ysv = np.einsum('nd,nj->jd', Y, dmat)
        Wsv = np.einsum('nd,nj->jd', W, dmat)
        
        ixi = np.ones(zeta.shape[1])    # (d)
        itau = np.ones(zeta.shape[1])   # (d)
        ixi[~self.sigma_unity] = xi
        itau[~self.sigma_unity] = tau

        shape = nZ + Wsv + ixi
        rate  = Ysv + itau

        sigma = gamma(shape, scale = 1 / rate)
        sigma.T[self.sigma_unity] = 1.
        return sigma
    
    def sample_rho(self, delta, zeta, sigma):
        As = np.einsum('il->i', zeta[delta])
        Bs = np.einsum('il,il->i', self.data.Yp, sigma[delta])
        return gamma(shape = As, scale = 1 / Bs)
    
    def sample_rho(self, delta, zeta, sigma):
        """ Sampling the PG_1 gammas for categorical variables

        Args:
            delta ([type]): [description]
            zeta ([type]): [description]
            sigma ([type]): [description]
        """
        As = zeta[:, self.nCol:][delta] + self.data.W
        Bs = sigma[:, self.nCol:][delta]
        rho = gamma(shape = As, scale = 1 / Bs)
        rho[rho < 1e-9] = 1e-9
        return rho
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.xi[0] = 1.
        self.samples.tau[0] = 1.
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 30, self.nCol))
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 30, self.nCol))
        self.samples.eta[0] = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 30, size = self.nDat)
        self.samples.delta[0][-1] = np.arange(self.max_clust_count - 30)[-1]
        self.samples.r[0] = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0], self.samples.sigma[0],
                )
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
    pass
    # from data import Data_From_Raw
    # from projgamma import GammaPrior
    # from pandas import read_csv
    # import os

    # raw = read_csv('./datasets/ivt_nov_mar.csv')
    # data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    # data.write_empirical('./test/empirical.csv')
    # model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    # model.sample(4000)
    # model.write_to_disk('./test/results.pickle', 2000, 2)
    # res = Result('./test/results.pickle')
    # res.write_posterior_predictive('./test/postpred.csv')
    # EOL



# EOF 2
