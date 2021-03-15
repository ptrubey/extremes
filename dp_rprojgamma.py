from projgamma import *
from scipy.stats import gamma, beta, gmean
from numpy.random import choice
from collections import namedtuple
from itertools import repeat
from math import ceil
import numpy as np
np.seterr(under = 'ignore', over = 'raise')
import multiprocessing as mp
import sqlite3 as sql
import pandas as pd
import data as dm
import cUtility as cu
import os

BNPPGPrior = namedtuple('BNPPGPrior', 'alpha beta eta')
Theta      = namedtuple('Theta','alpha beta')

def update_zeta_jl_wrapper(args):
    return sample_alpha_1_mh(*args)
def update_zeta_j(curr_zeta_j, Yj, alpha, beta):
    priors = [GammaPrior(a,b) for a,b in zip(alpha, beta)]
    args = zip(curr_zeta_j, Yj.T, priors)
    res = map(update_zeta_jl_wrapper, args)
    return np.array(list(res))
def update_zeta_j_wrapper(args):
    return update_zeta_j(*args)
def update_alpha_l_wrapper(args):
    return sample_alpha_k_mh(*args)
def update_beta_l_wrapper(args):
    return sample_beta_fc(*args)
def log_density_delta_i(args):
    return gamma.logpdf(args[0], a = args[1]).sum()

class SamplesDPMPG(object):
    zeta  = None # list, each entry is np.array, each row pertains to cluster
    alpha = None # Hierarchical distribution of shape parameter for zeta
    beta  = None # Hierarchical distribution of rate parameter for zeta
    delta = None # numpy array; int; indicates cluster membership
    eta   = None # Dispersion variable for DP algorithm
    r     = None # (latent) observation radius

    def __init__(self, nSamp, nDat, nCol):
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        self.zeta  = [None] * (nSamp + 1)
        self.alpha = np.empty((nSamp + 1, nCol))
        self.beta  = np.empty((nSamp + 1, nCol))
        return

class DPMPG(object):
    samples = None
    fixed_eta = False
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter].copy()
    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter].copy()
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter].copy()
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()
    @property
    def curr_nClust(self):
        return self.samples.delta[self.curr_iter].max() + 1

    def write_to_disk(self, path, nburn, thin = 1):
        """ Write output to disk """
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # Assemble output arrays
        zetas  = np.vstack([
            np.vstack((np.ones(zeta.shape[0]) * i, zeta.T)).T
            for i, zeta in enumerate(self.samples.zeta[nburn::thin])
            ])
        alphas = self.samples.alpha[nburn::thin]
        betas  = self.samples.beta[nburn::thin]
        deltas = self.samples.delta[nburn::thin]
        rs     = self.samples.r[nburn::thin]
        etas    = self.samples.eta[nburn::thin]
        # Assemble output DataFrames
        df_zetas = pd.DataFrame(
            zetas,
            columns = ['iter'] + ['zeta_{}'.format(i) for i in range(self.nCol)],
            )
        df_alphas = pd.DataFrame(
            alphas,
            columns = ['alpha_{}'.format(i) for i in range(self.nCol)],
            )
        df_betas = pd.DataFrame(
            betas,
            columns = ['beta_{}'.format(i) for i in range(self.nCol)],
            )
        df_deltas = pd.DataFrame(
            deltas,
            columns = ['delta_{}'.format(i) for i in range(self.nDat)]
            )
        df_rs = pd.DataFrame(
            rs,
            columns = ['r_{}'.format(i) for i in range(self.nDat)]
            )
        df_etas = pd.DataFrame({'eta' : etas})
        # Write Dataframes to SQL Connection
        df_zetas.to_sql('alpha_shape', conn, index = False)
        df_alphas.to_sql('beta_shape', conn, index = False)
        df_betas.to_sql('alphas', conn, index = False)
        df_deltas.to_sql('deltas', conn, index = False)
        df_rs.to_sql('rs', conn, index = False)
        df_etas.to_sql('eta', conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        return

    def clean_delta(self, delta, zeta, i):
        assert (delta.max() + 1 == zeta.shape[0])
        _delta = np.delete(delta, i)
        nj     = cu.counter(_delta, _delta.max() + 2)
        fz     = cu.first_zero(nj)
        _zeta  = zeta[np.where(nj > 0)[0]]
        if (fz == delta[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _zeta

    def sample_zeta(self, curr_zeta, delta, r, alpha, beta):
        Y = (self.data.Yl.T * r).T
        djs = [np.where(delta == j)[0] for j in range(delta.max() + 1)]
        args = zip(
            curr_zeta,
            [Y[djs[j]] for j in range(delta.max() + 1)],
            repeat(alpha),
            repeat(beta),
            )
        res = self.pool.map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def sample_zeta_new(self, n, alpha, beta):
        return gamma.rvs(a = alpha, scale = 1/beta, size = (n, self.nCol))

    def sample_alpha(self, zeta, curr_alpha):
        args = zip(curr_alpha, zeta.T, repeat(self.priors.alpha), repeat(self.priors.beta))
        res = self.pool.map(update_alpha_l_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        args = zip(alpha, zeta.T, repeat(self.priors.beta))
        res = self.pool.map(update_beta_l_wrapper, args)
        return np.array(list(res))

    def sample_delta_i(self, delta, r, zeta, alpha, beta, eta, i):
        # Clean the delta, zeta  calculate the new max delta
        _delta, _zeta = self.clean_delta(delta, zeta, i)
        _dmax = _delta.max()
        # Compute the prior probabilities for the collapsed sampler.
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * eta / self.m
        # Generate potential new clusters
        zeta_new = self.sample_zeta_new(self.m, alpha, beta)
        zeta_stack = np.vstack((_zeta, zeta_new))
        assert (zeta_stack.shape[0] == ljs.shape[0])
        args = zip(repeat(r[i] * self.data.Yl[i]), zeta_stack)
        res = self.pool.map(log_density_delta_i, args, chunksize = ceil(ljs.shape[0]/8))
        lps = np.array(list(res))
        # lps = np.array(list(map(log_density_gamma_i, args)))
        lps[np.where(np.isnan(lps))] = - np.inf
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + self.m + 1), 1, p = normalized)
        if dnew > _dmax:
            delta = np.insert(_delta, i, _dmax + 1)
            zeta = np.vstack((_zeta, zeta_stack[dnew]))
        else:
            delta = np.insert(_delta, i, dnew)
            zeta  = _zeta.copy()
        return delta, zeta

    def sample_r(self, zeta, delta):
        _zeta = zeta[delta]
        As = _zeta.sum(axis = 1)
        Bs = self.data.Yl.sum(axis = 1)
        return gamma.rvs(As, scale = 1 / Bs)

    def sample_eta(self, curr_eta, nClust):
        g  = beta.rvs(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma.rvs(a = aaa, scale = bb)

    def initialize_sampler(self, ns):
        self.samples = SamplesDPMPG(ns, self.nDat, self.nCol)
        # Initial conditions
        self.samples.delta[0] = np.array(list(range(self.nDat)))
        self.samples.r[0]     = 1.
        self.samples.eta[0]   = 5.
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = self.sample_zeta_new(
                self.nDat, self.samples.alpha[0], self.samples.beta[0],
                )
        self.curr_iter = 0
        return

    def iter_sample(self):
        # Fix the current estimates
        zeta  = self.curr_zeta
        delta = self.curr_delta
        eta   = self.curr_eta
        r     = self.curr_r
        alpha = self.curr_alpha
        beta  = self.curr_beta
        # advance the iterator
        self.curr_iter += 1
        # Compute New Cluster assignments
        for i in range(self.nDat):
            delta, zeta = self.sample_delta_i(delta, r, zeta, alpha, beta, eta, i)
        # Update Sampler with new values; Sample other parameters
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter] = self.sample_r(zeta, self.curr_delta)
        self.samples.zeta[self.curr_iter] = self.sample_zeta(
            zeta, self.curr_delta, self.curr_r, alpha, beta,
            )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter] = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, self.curr_nClust)
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        print_string = '\rSampling {:.1%} Completed, {} Clusters     '
        print(print_string.format(self.curr_iter / ns, self.nDat), end = '')
        while (self.curr_iter < ns):
            if (self.curr_iter % 10) == 0:
                print(print_string.format(self.curr_iter / ns, self.curr_nClust), end = '')
            self.iter_sample()
        print('\rSampling 100% Completed                    ')
        return

    def __init__(
            self,
            data,
            prior_alpha = GammaPrior(0.5,0.5),
            prior_beta = GammaPrior(2.,2.),
            prior_eta = GammaPrior(2.,.5),
            m = 20,
            ):
        self.m = m
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = BNPPGPrior(prior_alpha, prior_beta, prior_eta)
        self.pool = mp.Pool(8)
        return

class ResultDPMPG(object):
    samples = None
    nSamp   = None
    nDat    = None
    nCol    = None

    def generate_posterior_predictive(self, n_per_sample = 10):
        """ Generates posterior prediction, projects to hypercube, then
        casts to angular space """
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample)
        return to_angular(hyp)

    def generate_posterior_predictive_hypercube(self, n_per_sample):
        """ Generates a posterior prediction, and projects it to the hypercube """
        gnew = self.generate_posterior_predictive_gammas(n_per_sample)
        return (gnew.T / gnew.max(axis = 1)).T

    def generate_posterior_predictive_gammas(self, n_per_sample = 10, m = 20):
        new_gammas = []
        for i in range(self.nSamp):
            dmax   = self.samples.delta[i].max()
            njs    = cu.counter(self.samples.delta[i], dmax + 1 + m)
            ljs    = njs + (njs == 0) * self.samples.eta[i] / m
            new_zetas = gamma.rvs(
                a = self.samples.alpha[i], scale = 1/self.samples.beta[i], size = (m, self.nCol),
                )
            prob   = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta   = np.vstack((self.samples.zeta[i], new_zetas))[deltas]
            new_gammas.append(gamma.rvs(a = zeta))
        new_gamma_arr = np.vstack(new_gammas)
        return new_gamma_arr

    def write_posterior_predictive(self, path):
        thetas = pd.DataFrame(
            self.generate_posterior_predictive(),
            columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
            )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn    = sql.connect(path)
        ashapes = pd.read_sql('select * from alpha_shape;', conn).values
        bshapes = pd.read_sql('select * from beta_shape;', conn).values
        alphas  = pd.read_sql('select * from alphas;', conn).values
        betas   = pd.read_sql('select * from betas;', conn).values
        deltas  = pd.read_sql('select * from deltas;', conn).values.astype(int)
        rs      = pd.read_sql('select * from rs', conn).values
        eta     = pd.read_sql('select * from eta;', conn).values.T[0]

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = ashapes.shape[1]

        self.samples = SamplesDPMPG(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta = eta
        self.samples.r = rs
        self.samples.alpha = [
            alphas[np.where(alphas.T[0] == i)[0], 1:]
            for i in range(self.nSamp)
            ]
        self.samples.beta = [
            betas[np.where(betas.T[0] == i)[0], 1:]
            for i in range(self.nSamp)
            ]
        self.samples.alpha_shape = ashapes
        self.samples.beta_shape  = bshapes
        return

    def __init__(self, path):
        self.load_data(path)
        return

# EOF
