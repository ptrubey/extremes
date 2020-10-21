from projgamma import *
from scipy.stats import gamma, beta, gmean
from numpy.random import choice
from collections import namedtuple
from itertools import repeat
from math import ceil
import numpy as np
np.seterr(under = 'ignore')
import multiprocessing as mp
import sqlite3 as sql
import pandas as pd
import data as dm
import cUtility as cu
import os

BNPPGPrior = namedtuple('BNPPGPrior', 'alpha beta eta')
Theta      = namedtuple('Theta','alpha beta')

def log_density_gamma_i(args):
    return logdprojgamma_pre_single(*args)
def update_alpha_wrapper(args):
    if args[0] == 0:
        return sample_alpha_1_mh(*args[1:4])
    elif args[0] > 0:
        return sample_alpha_k_mh(*args[1:])
    else:
        raise ValueError('Something other than col index was passed!')
def update_beta_wrapper(args):
    return sample_beta_fc(*args)

class SamplesDPMPG(object):
    alpha = None # list, each entry is np.array; each row of array pertains to a cluster
    beta  = None # same as alpha
    delta = None # numpy array; int; indicates cluster membership
    eta   = None # Dispersion variable for DP algorithm
    r     = None # (latent) observation radius

    def __init__(self, nSamp, nDat, nCol):
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        self.alpha = []
        self.beta  = []
        return

class DPMPG(object):
    samples = None
    @property
    def curr_alphas(self):
        return self.samples.alpha[self.curr_iter].copy()

    @property
    def curr_betas(self):
        return self.samples.beta[self.curr_iter].copy()

    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()

    @property
    def curr_deltas(self):
        return self.samples.delta[self.curr_iter].copy()

    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()

    @property
    def curr_nClust(self):
        return self.curr_deltas.max() + 1

    def write_to_disk(self, path, nburn, thin = 1):
        """ Write output to disk """
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # Assemble output arrays
        alphas = np.vstack([
            np.vstack((np.ones(alpha.shape[0]) * i, alpha.T)).T
            for i, alpha in enumerate(self.samples.alpha[nburn::thin])
            ])
        betas  = np.vstack([
            np.vstack((np.ones(beta.shape[0]) * i, beta.T)).T
            for i, beta in enumerate(self.samples.beta[nburn::thin])
            ])
        deltas = self.samples.delta[nburn::thin]
        rs     = self.samples.r[nburn::thin]
        eta    = self.samples.eta[nburn::thin]
        # Assemble output DataFrames
        df_alphas = pd.DataFrame(
            alphas,
            columns = ['iter'] + ['alpha_{}'.format(i) for i in range(self.nCol)],
            )
        df_betas  = pd.DataFrame(
            betas,
            columns = ['iter'] + ['beta_{}'.format(i) for i in range(self.nCol)],
            )
        df_deltas = pd.DataFrame(
            deltas,
            columns = ['delta_{}'.format(i) for i in range(self.nDat)]
            )
        df_rs = pd.DataFrame(
            rs,
            columns = ['r_{}'.format(i) for i in range(self.nDat)]
            )
        df_eta = pd.DataFrame({'eta' : eta})
        # Write Dataframes to SQL Connection
        df_alphas.to_sql('alphas', conn, index = False)
        df_betas.to_sql('betas', conn, index = False)
        df_deltas.to_sql('deltas', conn, index = False)
        df_rs.to_sql('rs', conn, index = False)
        df_eta.to_sql('eta', conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        return

    def clean_delta(self, deltas, alphas, betas, i):
        assert (deltas.max() + 1 == alphas.shape[0])
        _delta = np.delete(deltas, i)
        nj     = cu.counter(_delta, _delta.max() + 2)
        fz     = cu.first_zero(nj)
        _alpha = alphas[np.where(nj > 0)[0]]
        _beta  = betas[np.where(nj > 0)[0]]
        if (fz == deltas[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _alpha, _beta

    def sample_delta_i(self, deltas, alphas, betas, eta, i):
        # Clean the deltas, alphas, and betas.  calculate the new max delta
        _delta, _alpha, _beta = self.clean_delta(deltas, alphas, betas, i)
        _dmax = _delta.max()
        # Compute the prior probabilities for the collapsed sampler.
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        # njs = np.array([(_delta == j).sum() for j in range(_dmax + 1 + self.m)])
        ljs = njs + (njs == 0) * eta / self.m
        # Generate potential new clusters
        alpha_new, beta_new = self.sample_alpha_beta_new(self.m)
        alpha_stack = np.vstack((_alpha, alpha_new))
        beta_stack  = np.vstack((_beta, beta_new))
        assert (alpha_stack.shape[0] == ljs.shape[0])
        # Calculate log-posteriors under each cluster
        args = zip(
            repeat(self.data.lcoss[i]),
            repeat(self.data.lsins[i]),
            repeat(self.data.Yl[i]),
            alpha_stack,
            beta_stack,
            )
        res = self.pool.map(log_density_gamma_i, args, chunksize = ceil(ljs.shape[0]/8))
        lps = np.array(list(res))
        # lps = np.array(list(map(log_density_gamma_i, args)))
        lps[np.where(np.isnan(lps))] = - np.inf
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + self.m + 1), 1, p = normalized)
        if dnew > _dmax:
            alpha = np.vstack((_alpha, alpha_stack[dnew]))
            beta  = np.vstack((_beta,  beta_stack[dnew]))
            delta = np.insert(_delta, i, _dmax + 1)
        else:
            delta = np.insert(_delta, i, dnew)
            alpha = _alpha.copy()
            beta  = _beta.copy()
        return delta, alpha, beta

    def sample_alpha_beta_new(self, size):
        alphas = self.alpha_prior.rvs(size = (size, self.nCol))
        betas  = np.hstack((np.ones((size, 1)), self.beta_prior.rvs(size = (size, self.nCol - 1))))
        return alphas, betas

    def update_alpha_beta(self, curr_alphas, deltas, Y):
        nClust = deltas.max() + 1
        djs  = [(deltas == j) for j in range(nClust)]
        Yjks = [Y[djs[j], k] for j in range(nClust) for k in range(self.nCol)]
        idxs = list(range(self.nCol)) * nClust
        alpha_args = zip(
            idxs,
            curr_alphas.reshape(-1),
            Yjks,
            repeat(self.priors.alpha),
            repeat(self.priors.beta)
            )
        prop_alphas = np.array(list(self.pool.map(update_alpha_wrapper, alpha_args))).reshape(curr_alphas.shape)
        # prop_alphas = np.array(list(map(update_alpha_wrapper, alpha_args))).reshape(curr_alphas.shape)
        Yjks = [Y[djs[j], k] for k in range(1, self.nCol) for j in range(nClust)]
        beta_args = zip(prop_alphas[:,1:].reshape(-1), Yjks, repeat(self.priors.beta))
        prop_betas = np.hstack((
            np.ones((nClust, 1)),
            np.array(list(self.pool.map(update_beta_wrapper, beta_args))).reshape(nClust, self.nCol - 1),
            # np.array(list(map(update_beta_wrapper, beta_args))).reshape(nClust, self.nCol - 1),
            ))
        return prop_alphas, prop_betas

    def sample_r(self, alphas, betas, deltas):
        alpha = alphas[deltas]
        beta  = betas[deltas]
        As = alpha.sum(axis = 1)
        Bs = (self.data.Yl * beta).sum(axis = 1)
        return gamma.rvs(As, scale = 1/Bs)

    def sample_eta(self, curr_eta, nClust):
        g  = beta.rvs(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma.rvs(aaa, bb)

    def set_prior_samplers(self):
        self.alpha_prior = gamma(self.priors.alpha.a, scale = 1. / self.priors.alpha.b)
        self.beta_prior  = gamma(self.priors.beta.a,  scale = 1. / self.priors.beta.b)
        return

    def initialize_sampler(self, ns):
        self.samples = SamplesDPMPG(ns, self.nDat, self.nCol)
        # Initial conditions
        self.samples.delta[0] = np.array(list(range(self.nDat)))
        self.samples.r[0]     = 1.
        self.samples.eta[0]   = 5.
        alpha_start, beta_start = self.sample_alpha_beta_new(self.nDat)
        self.samples.alpha.append(alpha_start)
        self.samples.beta.append(beta_start)
        self.curr_iter = 0
        return

    def iter_sample(self):
        # Fix the current estimates
        alphas, betas = self.curr_alphas, self.curr_betas
        deltas, eta   = self.curr_deltas, self.curr_eta
        # advance the iterator
        self.curr_iter += 1
        # Sample cluster assignments
        for i in range(self.nDat):
            deltas, alphas, betas = self.sample_delta_i(deltas, alphas, betas, eta, i)
        self.samples.delta[self.curr_iter] = deltas
        # Compute new latent radii
        self.samples.r[self.curr_iter] = self.sample_r(alphas, betas, deltas)
        # Sample new estimates based on new cluster assignments
        Y = (self.data.Yl.T * self.curr_r).T
        alphas, betas = self.update_alpha_beta(alphas, deltas, Y)
        self.samples.alpha.append(alphas)
        self.samples.beta.append(betas)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, alphas.shape[0])
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        print_string = '\rSampling {:.1%} Completed, {} Clusters'
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
            prior_alpha = GammaPrior(1.,1.),
            prior_beta = GammaPrior(1.,1.),
            prior_eta = GammaPrior(2.,15.),
            m = 20
            ):
        self.m = m
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = BNPPGPrior(prior_alpha, prior_beta, prior_eta)
        self.set_prior_samplers()
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
    
    def generate_posterior_predictive_gammas(self, n_per_sample = 10):
        new_gammas = []
        for i in range(self.nSamp):
            dmax = self.samples.delta[i].max()
            njs = cu.counter(self.samples.delta[i], dmax + 1)
            # njs = np.array([
            #     (self.samples.delta[i] == j).sum()
            #     for j in range(dmax + 1)
            #     ], dtype = int)
            prob = njs / njs.sum()
            # deltas = np.random.choice(range(dmax + 1), size = n_per_sample, p = prob)
            deltas = cu.generate_indices(prob, n_per_sample)
            alpha = self.samples.alpha[i][deltas]
            beta  = self.samples.beta[i][deltas]
            new_gammas.append(gamma.rvs(alpha, scale = 1 / beta))
        new_gamma_arr = np.vstack(new_gammas)
        return dm.to_angular(new_gamma_arr)

    def write_posterior_predictive(self, path):
        thetas = pd.DataFrame(
            self.generate_posterior_predictive(),
            columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
            )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)
        alphas = pd.read_sql('select * from alphas;', conn).values
        betas  = pd.read_sql('select * from betas;', conn).values
        deltas = pd.read_sql('select * from deltas;', conn).values.astype(int)
        rs     = pd.read_sql('select * from rs', conn).values
        eta    = pd.read_sql('select * from eta;', conn).values.T[0]

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = alphas.shape[1] - 1

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
        return

    def __init__(self, path):
        self.load_data(path)
        return

# EOF
