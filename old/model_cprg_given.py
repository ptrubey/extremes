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
from math import log, exp
from scipy.special import gammaln

import cUtility as cu
from samplers import BaseSampler
from cProjgamma import sample_alpha_k_mh_summary, sample_alpha_1_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube, euclidean_to_simplex, Categorical
from projgamma import GammaPrior

class Samples(object):
    zeta  = None
    # rho   = None
    
    # def __init__(self, nSamp, nCol):
    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = np.empty((nSamp + 1, nCol))
        self.rho   = np.empty((nSamp +1, nDat, nCol))
        self.logp = np.empty((nSamp + 1))
        return
    
    pass

Prior = namedtuple('Prior', 'zeta')

def logd_multinomial_paired(x, pi):
    logd = np.zeros(x.shape[0])
    logd += gammaln(x.sum(axis = 1) + 1)
    logd -= gammaln(x + 1).sum(axis = 1)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        lpi = np.log(pi)
        logd += np.nansum(x * lpi, axis = 1)
    return logd

def logd_dirichlet_mx_sa(pi, alpha):
    logd = np.zeros(pi.shape[0])
    logd += gammaln(alpha.sum())
    logd -= gammaln(alpha).sum()
    logd += ((alpha - 1) * np.log(pi)).sum(axis = 1)
    return logd

def logd_loggamma_mx_sab(x, alpha, beta):
    logd = np.zeros(x.shape)
    logd += alpha * np.log(beta)
    logd -= gammaln(alpha)
    logd += alpha * x
    logd -= beta * np.exp(x)
    return logd

def logd_gamma_mx_sab(x, alpha, beta):
    logd = np.zeros(x.shape)
    logd += alpha * np.log(beta)
    logd -= gammaln(alpha)
    logd += (alpha - 1) * np.log(x)
    logd -= beta * x
    return logd

def logd_dirichlet_multinomial_mx_sa(x, alpha):
    """
    x     : (n x d)
    alpha : (d)
    """
    sa = alpha.sum()
    sx = x.sum(axis = 1)
    logd = np.zeros(x.shape[0])
    logd += gammaln(sa)
    logd += gammaln(sx + 1)
    logd -= gammaln(sx + sa)
    logd += gammaln(x + alpha).sum(axis = 1)
    logd -= gammaln(alpha).sum()
    logd -= gammaln(x + 1).sum(axis = 1)
    return logd

def logd_dirichlet_multinomial_mx_ma(x, alpha):
    """ 
    x     : (n x d)
    alpha : (j x d)
    """
    sa = alpha.sum(axis = 1)
    sx = x.sum(axis = 1)
    logd = np.zeros((x.shape[0], alpha.shape[0]))
    logd += gammaln(sa)[None,:]
    logd += gammaln(sx + 1)[:,None]
    logd -= gammaln(sx[:, None] + sa[None,:])
    logd += gammaln(x[:,None,:] + alpha[None,:,:]).sum(axis = 2)
    logd -= gammaln(alpha).sum(axis = 1)[None,:]
    logd -= gammaln(x + 1).sum(axis = 1)[:, None]
    return logd

# def log_post_log_zeta_1(lzeta, rho, lrho, W, alpha, beta):
#     """
#     lzeta : (1)
#     Y, lY : (n)
#     W     : (n)
#     alpha : scalar
#     beta  : scalar
#     cmat  : (n x J), bool
#     """
#     zeta = np.exp(lzeta)
#     ZW = zeta + W 
#     lp = np.zeros(lzeta.shape)
#     lp += ((ZW - 1) * lrho).sum()
#     lp -= gammaln(ZW).sum()
#     lp += alpha * lzeta
#     lp -= zeta * beta
#     return lp

def log_post_log_zeta_1(lzeta, rho, lrho, W, alpha, beta):
    zeta = np.exp(lzeta)
    ZW = zeta + W
    lp = np.zeros(1)
    lp += ((ZW - 1) * lrho).sum()
    lp -= gammaln(ZW).sum()
    lp += alpha * lzeta
    lp -= beta * zeta
    return lp

def log_post_log_zeta_k(lzeta, rho, lrho, W, alpha, beta, xi, tau):
    zeta = np.exp(lzeta)
    ZW = zeta + W
    lp = np.zeros(1)
    lp += ((ZW - 1) * lrho).sum()
    lp -= gammaln(ZW).sum()
    lp += alpha * lzeta
    lp -= beta * zeta
    lp += gammaln(ZW.sum() + xi)
    lp -= (ZW.sum() + xi) * np.log(rho.sum() + tau)
    return lp

# def log_post_log_zeta_k(lzeta, rho, lrho, W, alpha, beta, xi, tau):
#     """
#     lzeta : (J)
#     Y, lY : (n)
#     W     : (n)
#     alpha : scalar
#     beta  : scalar
#     cmat  : (n x J), bool
#     """
#     zeta = np.exp(lzeta)
#     ZW = zeta + W # (n) array
#     lp = np.zeros(lzeta.shape) # should be 1d?
#     lp += ((ZW - 1) * lrho).sum()
#     lp -= gammaln(ZW).sum()
#     lp += alpha * lzeta
#     lp -= zeta * beta
#     lp += gammaln(ZW.sum() + xi)
#     lp -= (ZW.sum() + xi) * np.log(rho.sum() + tau)
#     return lp

def update_zeta_wrapper(args):
    # parse arguments
    curr_zeta, n, Ys, lYs, alpha, beta, xi, tau = args
    prop_zeta = np.empty(curr_zeta.shape)
    prop_zeta[0] = sample_alpha_1_mh_summary(
        curr_zeta[0], n, Ys[0], lYs[0], alpha, beta
        )
    for i in range(1, curr_zeta.shape[0]):
        prop_zeta[i] = sample_alpha_k_mh_summary(
            curr_zeta[i], n, Ys[i], lYs[i], 
            alpha, beta, xi, tau,
            )
    return prop_zeta

class Chain(BaseSampler):
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_rho(self):
        return self.samples.rho[self.curr_iter]

    # def sample_zeta(self, curr_zeta, rho):
    #     """
    #     Metropolis Hastings sampler for zeta
    #     """
    #     lrho = np.log(rho)
    #     lp_curr = np.zeros(curr_zeta.shape)
    #     lp_prop = np.zeros(curr_zeta.shape)
    #     # declaring current and proposal RV's
    #     curr_log_zeta = np.log(curr_zeta)
    #     prop_log_zeta = curr_log_zeta + normal(size = curr_log_zeta.shape, scale = 0.3)
    #     # indexing xi, tau to same index as alpha, beta
    #     for i in range(curr_log_zeta.shape[0]):
    #         if (i == 0):
    #             lp_curr[i] = log_post_log_zeta_1(
    #                 curr_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
    #                 self.priors.zeta.a, self.priors.zeta.b,
    #                 )
    #             lp_prop[i] = log_post_log_zeta_1(
    #                 prop_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
    #                 self.priors.zeta.a, self.priors.zeta.b,
    #                 )
    #         else: 
    #             lp_curr[i] = log_post_log_zeta_k(
    #                 curr_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
    #                 self.priors.zeta.a, self.priors.zeta.b, self.priors.sigma.a, self.priors.sigma.b,
    #             ) 
    #             lp_prop[i] = log_post_log_zeta_k(
    #                 prop_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
    #                 self.priors.zeta.a, self.priors.zeta.b, self.priors.sigma.a, self.priors.sigma.b,
    #                 )
    #     lp_diff = lp_prop - lp_curr
    #     keep = np.log(uniform(size = lp_curr.shape)) < lp_diff   # metropolis hastings step

    #     log_zeta = (prop_log_zeta * keep) + (curr_log_zeta * (~keep)) # if keep, then proposal, else current
    #     return np.exp(log_zeta)
    
    # def sample_zeta(self, curr_zeta, rho):
    #     srho = rho.sum(axis = 0)
    #     slrho = np.log(rho).sum(axis = 0)
    #     prop_zeta = np.empty(curr_zeta.shape)
    #     for i in range(self.nCol):
    #         prop_zeta[i] = sample_alpha_1_mh_summary(
    #             curr_zeta[i], self.nDat, srho[i], slrho[i],
    #             self.priors.zeta.a, self.priors.zeta.b,
    #             )
    #     return prop_zeta

    def sample_zeta(self, curr_zeta, rho):
        lrho = np.log(rho)
        curr_log_zeta = np.log(curr_zeta)
        prop_log_zeta = curr_log_zeta.copy() + normal(scale = 0.1, size = curr_log_zeta.shape)
        keep_log_zeta = np.empty(curr_log_zeta.shape)
        lunifs = np.log(uniform(size = curr_log_zeta.shape))
        logp = np.empty(2) # (current, proposal)
        logp[0] = log_post_log_zeta_1(curr_log_zeta[0], rho.T[0], lrho.T[0], self.data.W.T[0], 
                                        self.priors.zeta.a, self.priors.zeta.b)
        logp[1] = log_post_log_zeta_1(prop_log_zeta[0], rho.T[0], lrho.T[0], self.data.W.T[0], 
                                        self.priors.zeta.a, self.priors.zeta.b)
        if lunifs[0] < logp[1] - logp[0]:
            keep_log_zeta[0] = prop_log_zeta[0]
        else:
            keep_log_zeta[0] = curr_log_zeta[0]
        for i in range(1, self.nCol):
            logp[0] = log_post_log_zeta_1(curr_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
                                            self.priors.zeta.a, self.priors.zeta.b)
            logp[1] = log_post_log_zeta_1(prop_log_zeta[i], rho.T[i], lrho.T[i], self.data.W.T[i], 
                                            self.priors.zeta.a, self.priors.zeta.b)
            if lunifs[i] < logp[1] - logp[0]:
                keep_log_zeta[i] = prop_log_zeta[i]
            else:
                keep_log_zeta[i] = curr_log_zeta[i]
        return np.exp(keep_log_zeta)

    # def sample_zeta(self, curr_zeta):
    #     curr_log_zeta = np.log(curr_zeta)
    #     eval_log_zeta = curr_log_zeta.copy()
    #     prop_log_zeta = curr_log_zeta + normal(scale = 0.1, size = curr_log_zeta.shape)
    #     lunifs = np.log(uniform(size = curr_log_zeta.shape))
    #     logp = np.zeros(2) # (current, proposal)
    #     for i in range(self.nCol):
    #         logp[0] += logd_dirichlet_multinomial_mx_sa(
    #             self.data.W, np.exp(eval_log_zeta),
    #             ).sum()
    #         logp[0] += logd_loggamma_mx_sab(
    #             eval_log_zeta[i], self.priors.zeta.a, self.priors.zeta.b,
    #             )
    #         eval_log_zeta[i] = prop_log_zeta[i]
    #         logp[1] += logd_dirichlet_multinomial_mx_sa(
    #             self.data.W, np.exp(eval_log_zeta),
    #             ).sum()
    #         logp[1] += logd_loggamma_mx_sab(
    #             eval_log_zeta[i], self.priors.zeta.a, self.priors.zeta.b,
    #             )
    #         if lunifs[i] < logp[1] - logp[0]:
    #             pass # keep eval_log_zeta with updated parameter
    #         else:
    #             eval_log_zeta[i] = curr_log_zeta[i] # return eval_log_zeta to previous 
    #         logp[:] = 0.
    #     return np.exp(eval_log_zeta)

    def sample_rho(self, zeta):
        As = zeta + self.data.W
        rho = gamma(shape = As)
        rho[rho < 1e-200] = 1e-200
        return rho
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = self.nCol)
        self.samples.rho[0] = self.sample_rho(self.samples.zeta[0])
        self.curr_iter = 0
        return

    def iter_sample(self):
        zeta  = self.curr_zeta.copy()
        rho   = self.curr_rho

        self.curr_iter += 1

        self.samples.zeta[self.curr_iter] = self.sample_zeta(zeta, rho)
        self.samples.rho[self.curr_iter] = self.sample_rho(self.curr_zeta)
        
        pis = self.curr_rho / self.curr_rho.sum(axis = 1)[:,None]
        
        self.samples.logp[self.curr_iter] = (
            + logd_multinomial_paired(self.data.W, pis).sum()
            + logd_dirichlet_mx_sa(pis, self.curr_zeta).sum()
            # + logd_dirichlet_multinomial_mx_sa(self.data.W, self.curr_zeta).sum()
            + logd_gamma_mx_sab(self.curr_zeta, self.priors.zeta.a, self.priors.zeta.b).sum()
            )
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.mkdir(folder)
        if os.path.exists(path):
            os.remove(path)
        
        zeta = self.samples.zeta[nBurn::nThin]
        rho   = self.samples.rho[nBurn::nThin]

        out = {
            'zetas'  : zeta,
            'rhos'   : rho,
            'W'      : self.data.W,
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
            ):
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_zeta)
        return
    
    pass

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1):
        new_gammas = []
        for s in range(self.nSamp):
            g = gamma(
                shape = self.samples.zeta[s],
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
        rhos   = out['rhos']

        self.nSamp = zetas.shape[0]
        self.nDat = out['W'].shape[0]
        # self.nDat  = rhos.shape[1]
        self.nCol  = zetas.shape[1]

        self.data = Categorical(out['W'])
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass
        
        self.samples = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.zeta = zetas
        self.samples.rho = rhos
        return 

    def __init__(self, path):
        self.load_data(path)
        return
    
    pass

# EOF

if __name__ == '__main__':
    from pandas import read_csv
    from data import Multinomial

    raw = read_csv('./simulated/categorical/test.csv').values
    data = Multinomial(raw)
    model = Chain(data)
    model.sample(200000)
    model.write_to_disk('./simulated/categorical/result.pkl', 1, 200)
    res = Result('./simulated/categorical/result.pkl')
    print(model.samples.zeta[100000:].mean(axis = 0))
    raise

# EOF
