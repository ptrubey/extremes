from numpy.random import choice, gamma, beta, normal, uniform
from collections import namedtuple
from itertools import repeat
import numpy as np
import pandas as pd
import os
import sqlite3 as sql
from math import ceil, log
# from multiprocessing import Pool

import cUtility as cu
from cProjgamma import sample_alpha_1_mh, sample_alpha_k_mh, sample_beta_fc, \
                        logddirichlet, logdgamma, logdgamma_restricted
from data import euclidean_to_simplex, euclidean_to_hypercube, euclidean_to_angular, Data
from projgamma import GammaPrior, DirichletPrior

def logdgamma_restricted_wrapper(args):
    return logdgamma_restricted(*args)

def sample_delta_i(args):
    Yi = args[0]
    zeta = args[1]
    pi = args[2]
    argset = zip(repeat(Yi), zeta)
    lps = np.array(list(map(logdgamma_restricted_wrapper, argset)))
    lps[np.isnan(lps)] = - np.inf
    lps = lps - lps.max()
    unnormalized = np.exp(lps) * pi
    normalized = unnormalized / unnormalized.sum()
    return choice(pi.shape[0], 1, p = normalized)

def update_alpha_l_wrapper(args):
    """ wrapper for projgamma.sample_alpha_k_mh

    sample_alpha_k_mh assumes a gamma likelihood with gamma priors for both
    shape and rate parameters.
     """
    return sample_alpha_k_mh(*args)

def update_beta_l_wrapper(args):
    """ Wrapper for projgamma.sample_beta_fc

    sample_beta_fc assumes a gamma likelihood with gamma prior for the rate parameter.
    sampling is done via full conditional (which has form of a gamma).
    """
    return sample_beta_fc(*args)

def update_zeta_jl_wrapper(args):
    return sample_alpha_1_mh(*args)

def update_zeta_j(curr_zeta_j, Yj, alpha, beta):
    """
    Wrapper for projgamma.sample_alpha_1_mh
    sample_alpha_1_mh assumes a Gamma likelihood with rate parameter = 1, and a gamma
    prior for the shape parameter.
    """
    args = zip(curr_zeta_j, Yj.T, alpha, beta)
    res = map(update_zeta_jl_wrapper, args)
    return np.array(list(res))

def update_zeta_j_wrapper(args):
    return update_zeta_j(*args)

# Functions and definitions relating to DP Mixture

def log_density_delta_i(args):
    return logdgamma_restricted(args[0], args[1])

class Samples(object):
    zeta  = None  # List, each entry is np.array, each row of array pertains to cluster
    delta = None # np.array, int, indicates cluster membership
    alpha = None # np.array; Shape parameter prior for zeta
    beta  = None # np.array; rate parameter prior for zeta
    eta   = None # dispersion parameter for DP algorithm
    r     = None # Latent observation radius

    def __init__(self, nSamp, nDat, nCol):
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        self.alpha = np.empty((nSamp + 1, nCol))
        self.beta  = np.empty((nSamp + 1, nCol))
        self.zeta  = [None] * (nSamp + 1)
        return

Prior = namedtuple('Prior', 'alpha beta eta')

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 10, m = 20):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = cu.counter(self.samples.delta[s], int(dmax + 1 + m))
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            new_zetas = gamma(
                    shape = self.samples.alpha[s],
                    scale = 1. / self.samples.beta[s],
                    size = (m, self.nCol),
                    )
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            new_gammas.append(gamma(shape = zeta))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 10):
        gnew = self.generate_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_hypercube(gnew)

    def generate_posterior_predictive(self, n_per_sample = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample)
        return euclidean_to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 10):
        thetas = pd.DataFrame(
            self.generate_posterior_predictive(n_per_sample),
            columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
            )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)

        zetas  = pd.read_sql('select * from zetas;', conn).values
        deltas = pd.read_sql('select * from deltas;', conn).values.astype(int)
        alphas = pd.read_sql('select * from alphas;', conn).values
        betas  = pd.read_sql('select * from betas;', conn).values
        rs     = pd.read_sql('select * from rs', conn).values
        etas   = pd.read_sql('select * from etas', conn).values.reshape(-1)

        if len(deltas.shape) > 1:
            self.nDat  = deltas.shape[1]
            self.nSamp = deltas.shape[0]
        else:
            self.nSamp = etas.shape[0]
            deltas = deltas.reshape(self.nSamp, -1)
            self.nDat = deltas.shape[1]

        self.nCol  = alphas.shape[1]

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.r     = rs
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

class Chain(object):
    """ DP clustering of Dirichlet RV's on the Simplex """
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter].copy()
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter].copy()
    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter].copy()
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()

    def clean_delta(self, delta, zeta, i):
        assert (delta.max() + 1 == zeta.shape[0])
        _delta = np.delete(delta, i)
        nj = cu.counter(_delta, _delta.max() + 2)
        fz = cu.first_zero(nj)
        _zeta = zeta[np.where(nj > 0)[0]]
        if (fz == delta[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _zeta

    def sample_zeta_new(self, m, alpha, beta):
        return gamma(shape = alpha, scale = 1. / beta, size = (m, self.nCol))

    def sample_zeta(self, curr_zeta, delta, r, alpha, beta):
        Y = (self.data.V.T * r).T
        djs = [np.where(delta == j)[0] for j in range(delta.max() + 1)]
        args = zip(
            curr_zeta,
            [Y[djs[j]] for j in range(delta.max() + 1)],
            repeat(alpha),
            repeat(beta),
            )
        res = map(update_zeta_j_wrapper, args)
        # res = self.pool.map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def sample_alpha(self, zeta, curr_alpha):
        # args = zip(curr_alpha, zeta.T, repeat(self.priors.alpha), repeat(self.priors.beta))
        args = zip(
            curr_alpha,
            zeta.T,
            repeat(self.priors.alpha.a),
            repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a),
            repeat(self.priors.beta.b),
            )
        # res = self.pool.map(update_alpha_l_wrapper, args)
        res = map(update_alpha_l_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        # args = zip(alpha, zeta.T, repeat(self.priors.beta))
        args = zip(alpha, zeta.T, repeat(self.priors.beta.a), repeat(self.priors.beta.b))
        # res = self.pool.map(update_beta_l_wrapper, args)
        res = map(update_beta_l_wrapper, args)
        return np.array(list(res))

    def sample_r(self, delta, zeta):
        shapes = zeta[delta].sum(axis = 1)
        rates  = self.data.V.sum(axis = 1)
        return gamma(shape = shapes, scale = 1. / rates)

    def sample_eta(self, curr_eta, delta):
        nClust = delta.max() + 1
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_delta_i(self, delta, zeta, r, alpha, beta, eta, i):
        _delta, _zeta = self.clean_delta(delta, zeta, i)
        _dmax = _delta.max()
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * eta / self.m
        _zeta_new = self.sample_zeta_new(self.m, alpha, beta)
        zeta_stack = np.vstack((_zeta, _zeta_new))
        assert (zeta_stack.shape[0] == ljs.shape[0])
        args = zip(
            repeat(r[i] * self.data.V[i]),
            zeta_stack,
            )
        # res = self.pool.map(log_density_delta_i, args, chunksize = ceil(ljs.shape[0] / 8.))
        res = map(log_density_delta_i, args)
        lps = np.array(list(res))
        lps[np.where(np.isnan(lps))[0]] = - np.inf
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + 1 + self.m), 1, p = normalized)
        if dnew > _dmax:
            zeta = np.vstack((_zeta, zeta_stack[dnew]))
            delta = np.insert(_delta, i, _dmax + 1)
        else:
            delta = np.insert(_delta, i, dnew)
            zeta = _zeta.copy()
        return delta, zeta

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.delta[0] = np.array(list(range(self.nDat)), dtype = int)
        self.samples.r[0]     = 1.
        self.samples.eta[0]   = 40.
        self.samples.alpha[0] = 1.
        self.samples.beta[0]  = 1.
        self.samples.zeta[0]  = self.sample_zeta_new(
            self.nDat, self.samples.alpha[0], self.samples.beta[0],
            )
        self.curr_iter = 0
        return

    def iter_sample(self):
        """ Advance the sampler one iteration """
        # Parse the current values
        zeta  = self.curr_zeta
        delta = self.curr_delta
        eta   = self.curr_eta
        r     = self.curr_r
        alpha = self.curr_alpha
        beta  = self.curr_beta
        # Advance the iterator
        self.curr_iter += 1
        # Compute new cluster assignments based on current values of zeta (and new as needed)
        for i in range(self.nDat):
            delta, zeta = self.sample_delta_i(delta, zeta, r, alpha, beta, eta, i)
        # Update sampler with new values
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter] = self.sample_r(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter] = self.sample_zeta(
                zeta, self.curr_delta, self.curr_r, alpha, beta,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter] = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, self.curr_delta)
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        print_string = '\rSampling {:.1%} Completed, {} Clusters     '
        print(print_string.format(self.curr_iter / ns, self.nDat), end = '')
        while (self.curr_iter < ns):
            if (self.curr_iter % 10) == 0:
                print(print_string.format(self.curr_iter / ns, self.curr_delta.max() + 1), end = '')
            self.iter_sample()
        print('\rSampling 100% Completed                    ')
        return

    def write_to_disk(self, path, nburn, thin = 1):
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # Gather output data
        zetas = np.vstack([
            np.vstack((np.ones(zeta.shape[0]) * i, zeta.T)).T
            for i, zeta in enumerate(self.samples.zeta[nburn::thin])
            ])
        alphas = self.samples.alpha[nburn::thin]
        betas  = self.samples.beta[nburn::thin]
        etas   = self.samples.eta[nburn::thin]
        deltas = self.samples.delta[nburn::thin]
        rs     = self.samples.r[nburn::thin]
        # Assemble Output DataFrames
        df_zeta  = pd.DataFrame(
                zetas, columns = ['iter'] + ['zeta_{}'.format(i) for i in range(self.nCol)],
                )
        df_alpha = pd.DataFrame(alphas, columns = ['alpha_{}'.format(i) for i in range(self.nCol)])
        df_beta  = pd.DataFrame(betas, columns = ['beta_{}'.format(i) for i in range(self.nCol)])
        df_eta   = pd.DataFrame({'eta' : etas})
        df_r     = pd.DataFrame(rs, columns = ['r_{}'.format(i) for i in range(self.nDat)])
        df_delta = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        # Write DataFrames to SQL Connection
        df_zeta.to_sql('zetas', conn, index = False)
        df_alpha.to_sql('alphas', conn, index = False)
        df_beta.to_sql('betas', conn, index = False)
        df_eta.to_sql('etas', conn, index = False)
        df_r.to_sql('rs', conn, index = False)
        try:
            df_delta.to_sql('deltas', conn, index = False)
        except sql.OperationalError:
            deltas_dft = pd.DataFrame({'delta' : deltas.reshape(-1)})
            deltas_dft.to_sql('deltas', conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        pass

    def __init__(
            self,
            data,
            prior_alpha = GammaPrior(1.,1.),
            prior_beta = GammaPrior(1.,1.),
            prior_eta = GammaPrior(2.,0.5),
            m = 20,
            ):
        self.m = m
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_alpha, prior_beta, prior_eta)
        # self.pool = Pool(8)
        return

# EOF
