# from cProjgamma import *
from projgamma import *
from scipy.stats import gamma, multinomial
from numpy.random import choice
from collections import namedtuple
import numpy as np
import sqlite3 as sql
import pandas as pd

MPGPrior   = namedtuple('MPGPrior', 'alpha beta eta')
# GammaPrior = namedtuple('GammaPrior', 'a b')
class MPGSamples(object):
    alpha = None
    beta  = None
    eta   = None
    delta = None
    r     = None

    def __init__(self, nSamp, nDat, nCol, nMix):
        self.alpha = np.empty((nSamp + 1, nMix, nCol))
        self.beta  = np.empty((nSamp + 1, nMix, nCol))
        self.eta   = np.empty((nSamp + 1, nMix))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        return

class MPG(object):
    samples        = None
    priors_alpha   = None
    priors_beta    = None
    def plot_posterior_predictive(self, nburn):
        delta = np.apply_along_axis(
            lambda p: choice(nmix, 1, p = p),
            1,
            self.samples.delta[nburn:],
            ).flatten()
        alphas = np.array([
                self.samples.alpha[nburn:][i,j]
                for i, j in enumerate(delta)
                ])
        betas  = np.array([
                self.samples.beta[nburn:][i,j]
                for i, j in enumerate(delta)
                ])
        theta_new = rprojgamma(alphas, betas)

    def sample_alpha(self, R, curr_alpha, delta):
        Y = (self.data.Yl.T * R).T
        prop_alpha = np.empty(curr_alpha.shape)
        # This part should be parallelized!
        for j in range(self.nMix):
            Yj = Y[delta == j]
            prop_alpha[j,0] = sample_alpha_1_mh(
                curr_alpha[j,0], Yj[:,0], self.priors.alpha,
                )
            for k in range(1, self.nCol):
                prop_alpha[j,k] = sample_alpha_k_mh(
                    curr_alpha[j,k], Yj[:,k], self.priors.alpha, self.priors.beta
                    )
        return prop_alpha

    def sample_beta(self, R, alpha, delta):
        Y = (self.data.Yl.T * R).T
        prop_beta = np.empty(alpha.shape)
        prop_beta[:,0] = 1.
        for j in range(self.nMix):
            Yj = Y[delta == j]
            for k in range(1, self.nCol):
                prop_beta[j,k] = sample_beta_fc(alpha[j,k], Yj[:,k], self.priors.beta)
        return prop_beta

    def sample_eta(self, delta):
        shapes = np.array([
            (delta == j).sum() + self.priors.eta.a
            for j in range(self.nMix)
            ])
        unnormalized = gamma.rvs(a = shapes)
        return unnormalized / unnormalized.sum()

    def sample_r(self, alpha, beta, delta):
        As = (alpha[delta]).sum(axis = 1)
        Bs = (self.data.Yl * beta[delta]).sum(axis = 1)
        return gamma.rvs(As, scale = 1/Bs)

    def log_posterior_delta_j(self, alpha_j, beta_j):
        lp = logdprojgamma_pre(
            self.data.lcoss, self.data.lsins, self.data.Yl, alpha_j, beta_j,
            )
        return lp

    def sample_delta(self, eta, alpha, beta):
        shapes = np.exp(np.vstack([
            self.log_posterior_delta_j(alpha[j], beta[j])
            for j in range(self.nMix)
            ]).T)
        unnormalized = shapes * eta
        normalized = (unnormalized.T / unnormalized.sum(axis = 1)).T
        choices = tuple(range(self.nMix))
        gmat = np.apply_along_axis(
            lambda p: choice(self.nMix, 1, p = p),
            1,
            normalized,
            ).flatten()
        # gmat = np.apply_along_axis(lambda p: multinomial.rvs(1, p), 0, normalized)
        # return np.apply_along_axis(np.where, 1, gmat).flatten().astype(int)
        return gmat

    def set_priors(self):
        self.alpha_prior = gamma(self.priors.alpha.a, scale = 1 / self.priors.alpha.b)
        self.beta_prior  = gamma(self.priors.beta.a,  scale = 1 / self.priors.beta.b)
        return

    def initialize_sampler(self, ns):
        # set sampler target
        self.samples = MPGSamples(ns, self.nDat, self.nCol, self.nMix)
        # self.samples_alpha  = np.empty((ns, self.nMix, self.nCol))
        # self.samples_beta   = np.empty((ns, self.nMix, self.nCol))
        # self.samples_eta    = np.empty((ns, self.nMix))
        # self.samples_delta  = np.empty((ns, self.nDat), dtype = int)
        # self.samples_r      = np.empty((ns, self.nDat))
        # set initial values
        self.samples.alpha[0] = self.alpha_prior.rvs(size = (self.nMix, self.nCol))
        self.samples.beta[0]  = self.beta_prior.rvs(size = (self.nMix, self.nCol))
        self.samples.beta[:,:,0] = 1.
        self.samples.eta[0]  = 1. / self.nMix
        self.samples.r[0]    = 1.
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        for i in range(1, ns + 1):
            self.samples.delta[i] = self.sample_delta(
                self.samples.eta[i-1],
                self.samples.alpha[i-1],
                self.samples.beta[i-1],
                )
            self.samples.r[i] = self.sample_r(
                self.samples.alpha[i-1],
                self.samples.beta[i-1],
                self.samples.delta[i],
                )
            self.samples.eta[i] = self.sample_eta(self.samples.delta[i])
            self.samples.alpha[i] = self.sample_alpha(
                self.samples.r[i],
                self.samples.alpha[i-1],
                self.samples.delta[i],
                )
            self.samples.beta[i] = self.sample_beta(
                self.samples.r[i],
                self.samples.alpha[i],
                self.samples.delta[i],
                )
        return

    def write_to_disk(self, path, nBurn, thin = 1):
        nSamp = self.samples.alpha.shape[0]
        nKeep = nSamp - nBurn - 1
        conn  = sql.connection(path)
        alpha = self.samples.alpha[- nKeep :: thin]
        beta  = self.samples.beta[- nKeep :: thin]
        eta   = self.samples.eta[- nKeep :: thin]
        delta = self.samples.delta[- nKeep :: thin]
        r     = self.samples.r[- nKeep :: thin]
        return

    def __init__(
                self,
                data,
                nMix,
                prior_alpha = GammaPrior(1.,1.),
                prior_beta = GammaPrior(1.,1.),
                prior_eta = DirichletPrior(5.)
                ):
        self.data = data
        self.nMix = nMix
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = MPGPrior(
            prior_alpha,
            prior_beta,
            prior_eta,
            )
        self.set_priors()
        return

# EOF
