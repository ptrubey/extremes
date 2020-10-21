# from cProjgamma import *
from projgamma import *
from data import *
from scipy.stats import gamma, multinomial
from numpy.random import choice
from collections import namedtuple
import numpy as np
np.seterr(under = 'ignore')
import sqlite3 as sql
import pandas as pd
from cUtility import generate_indices
import os

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

    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter].copy()

    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter].copy()

    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()

    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()

    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()

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
        lps = np.vstack([
            self.log_posterior_delta_j(alpha[j], beta[j])
            for j in range(self.nMix)
            ]).T
        lps[np.isnan(lps)] = - np.inf
        lps = (lps.T - lps.max(axis = 1)).T
        shapes = np.exp(lps)
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
        # set initial values
        self.samples.alpha[0] = self.alpha_prior.rvs(size = (self.nMix, self.nCol))
        self.samples.beta[0]  = self.beta_prior.rvs(size = (self.nMix, self.nCol))
        self.samples.beta[:,:,0] = 1.
        self.samples.eta[0]  = 1. / self.nMix
        self.samples.r[0]    = 1.
        self.curr_iter = 0
        return

    def iter_sample(self):
        eta   = self.curr_eta
        alpha = self.curr_alpha
        beta  = self.curr_beta

        self.curr_iter += 1
        self.samples.delta[self.curr_iter] = self.sample_delta(eta, alpha, beta)
        self.samples.r[self.curr_iter] = self.sample_r(alpha, beta, self.curr_delta)
        self.samples.eta[self.curr_iter] = self.sample_eta(self.curr_delta)
        self.samples.alpha[self.curr_iter] = self.sample_alpha(
            self.curr_r, alpha, self.curr_delta
            )
        self.samples.beta[self.curr_iter] = self.sample_beta(
            self.curr_r, self.curr_alpha, self.curr_delta
            )
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        print_string = '\rSampling {:.1%} Completed'
        print(print_string.format(self.curr_iter / ns), end = '')
        while (self.curr_iter < ns):
            if (self.curr_iter % 10) == 0:
                print(print_string.format(self.curr_iter / ns), end = '')
            self.iter_sample()
        print('\rSampling 100% Completed            ')
        return

    def write_to_disk(self, path, nBurn, thin = 1):
        nTail = self.samples.alpha.shape[0] - nBurn - 1
        if os.path.exists(path):
            os.remove(path)
        conn  = sql.connect(path)

        alphas = self.samples.alpha[-nTail :: thin]
        betas  = self.samples.beta[-nTail :: thin]
        etas   = self.samples.eta[- nTail :: thin]
        deltas = self.samples.delta[- nTail :: thin]
        rs     = self.samples.r[- nTail :: thin]

        nSamp = etas.shape[0]
        assert (alphas.shape[0] == etas.shape[0] and betas.shape[0] == deltas.shape[0])

        meta = np.array((self.nDat, self.nCol, self.nMix, nSamp)).reshape(1,-1)
        meta_df = pd.DataFrame(meta, columns = ('nDat','nCol','nMix','nSamp'))

        alphas_df = pd.DataFrame(
            alphas.reshape(nSamp * self.nMix, self.nCol),
            columns = ['alpha_{}'.format(i) for i in range(self.nCol)],
            )
        betas_df = pd.DataFrame(
            betas.reshape(nSamp * self.nMix, self.nCol),
            columns = ['beta_{}'.format(i) for i in range(self.nCol)],
            )
        etas_df = pd.DataFrame(
            etas,
            columns = ['eta_{}'.format(i) for i in range(self.nMix)],
            )
        deltas_df = pd.DataFrame(
            deltas,
            columns = ['delta_{}'.format(i) for i in range(self.nDat)],
            )
        rs_df = pd.DataFrame(
            rs,
            columns = ['r_{}'.format(i) for i in range(self.nDat)],
            )

        alphas_df.to_sql('alphas', conn, index = False)
        betas_df.to_sql('betas', conn, index = False)
        etas_df.to_sql('etas', conn, index = False)
        deltas_df.to_sql('deltas', conn, index = False)
        rs_df.to_sql('rs', conn, index = False)
        meta_df.to_sql('meta', conn, index = False)
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

class MPGResult(object):
    samples = None
    nSamp = None
    nDat = None
    nCol = None
    nMix = None

    def generate_posterior_predictive_gammas(self, n_per_sample):
        """ Generates the posterior predictive distribution in Gamma space """
        dnew = np.array(list(map(
                lambda x: generate_indices(x, n_per_sample),
                self.samples.eta,
                )))
        gnew = np.vstack([
            gamma.rvs(alpha[delta], scale = 1 / beta[delta])
            for alpha, beta, delta
            in zip(self.samples.alpha, self.samples.beta, dnew)
            ])
        return gnew

    def generate_posterior_predictive(self, n_per_sample = 10):
        """ Generates posterior prediction, projects to hypercube, then
        casts to angular space """
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample)
        return to_angular(hyp)

    def generate_posterior_predictive_hypercube(self, n_per_sample):
        """ Generates a posterior prediction, and projects it to the hypercube """
        gnew = self.generate_posterior_predictive_gammas(n_per_sample)
        return (gnew.T / gnew.max(axis = 1)).T

    def write_posterior_predictive(self, path):
        thetas = pd.DataFrame(
            self.generate_posterior_predictive(),
            columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
            )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)

        meta = pd.read_sql('select * from meta;', conn)
        self.nSamp = meta.nSamp[0]
        self.nCol  = meta.nCol[0]
        self.nMix  = meta.nMix[0]
        self.nDat  = meta.nDat[0]

        absize = (self.nSamp, self.nMix, self.nCol)
        alphas = pd.read_sql('select * from alphas;', conn).values.reshape(*absize)
        betas  = pd.read_sql('select * from betas;', conn).values.reshape(*absize)
        deltas = pd.read_sql('select * from deltas;', conn).values
        etas   = pd.read_sql('select * from etas;', conn).values
        rs     = pd.read_sql('select * from rs;', conn).values

        self.samples = MPGSamples(self.nSamp, self.nDat, self.nCol, self.nMix)
        self.samples.alpha = alphas
        self.samples.beta = betas
        self.samples.eta = etas
        self.samples.r = rs
        self.samples.delta = deltas

        conn.close()
        return

    def __init__(self, path):
        self.load_data(path)
        return
# EOF
