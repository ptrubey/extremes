from projgamma import *
from scipy.stats import gamma, multinomial
from numpy.random import choice
from collections import namedtuple
import numpy as np

MPGPrior       = namedtuple('PGPrior', 'alpha beta eta')

class MPG(object):
    samples_alpha  = None
    samples_beta   = None
    samples_eta    = None
    samples_delta  = None
    samples_r      = None
    priors_alpha   = None
    priors_beta    = None

    def sample_alpha(self, R, curr_alpha, delta):
        Y = (self.data.Yl.T * R).T
        prop_alpha = np.empty(curr_alpha.shape)
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
        self.alpha_prior = gamma(self.priors.alpha.a, 1 / self.priors.alpha.b)
        self.beta_prior  = gamma(self.priors.beta.a,  1 / self.priors.beta.b)
        return

    def initialize_sampler(self, ns):
        # set sampler target
        self.samples_alpha  = np.empty((ns, self.nMix, self.nCol))
        self.samples_beta   = np.empty((ns, self.nMix, self.nCol))
        self.samples_eta    = np.empty((ns, self.nMix))
        self.samples_delta  = np.empty((ns, self.nDat), dtype = int)
        self.samples_r      = np.empty((ns, self.nDat))
        # set initial values
        self.samples_alpha[0] = self.alpha_prior.rvs(size = (self.nMix, self.nCol))
        self.samples_beta[0]  = self.beta_prior.rvs(size = (self.nMix, self.nCol))
        self.samples_beta[:,:,0] = 1.
        self.samples_eta[0]  = 1. / self.nMix
        self.samples_r[0]    = 1.
        return

    def sample(self, ns):
        self.initialize_sampler(ns + 1)
        for i in range(1, ns + 1):
            self.samples_delta[i] = self.sample_delta(
                self.samples_eta[i-1],
                self.samples_alpha[i-1],
                self.samples_beta[i-1],
                )
            self.samples_r[i] = self.sample_r(
                self.samples_alpha[i-1],
                self.samples_beta[i-1],
                self.samples_delta[i],
                )
            self.samples_eta[i] = self.sample_eta(self.samples_delta[i])
            self.samples_alpha[i] = self.sample_alpha(
                self.samples_r[i],
                self.samples_alpha[i-1],
                self.samples_delta[i],
                )
            self.samples_beta[i] = self.sample_beta(
                self.samples_r[i],
                self.samples_alpha[i],
                self.samples_delta[i],
                )
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
