from numpy.random import choice, gamma, beta, normal, uniform
from collections import namedtuple
from itertools import repeat
import numpy as np
import pandas as pd
import os
import sqlite3 as sql
from math import ceil, log
from multiprocessing import Pool
from energy import limit_cpu

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

class Samples(object):
    zeta  = None
    delta = None
    pi    = None
    alpha = None
    beta  = None
    r     = None

    def __init__(self, nSamp, nDat, nCol, nMix):
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.zeta  = np.empty((nSamp + 1, nMix, nCol))
        self.alpha = np.empty((nSamp + 1, nCol))
        self.beta  = np.empty((nSamp + 1, nCol))
        self.pi    = np.empty((nSamp + 1, nMix))
        self.r     = np.empty((nSamp + 1, nDat))
        return

Prior = namedtuple('Prior', 'alpha beta pi')

class Chain(object):
    samples = None
    priors = None

    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter].copy()
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()
    @property
    def curr_pi(self):
        return self.samples.pi[self.curr_iter].copy()
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter].copy()
    @property
    def  curr_beta(self):
        return self.samples.beta[self.curr_iter].copy()

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
        # res = map(update_alpha_l_wrapper, args)
        res = self.pool.map(update_alpha_l_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        # args = zip(alpha, zeta.T, repeat(self.priors.beta))
        args = zip(alpha, zeta.T, repeat(self.priors.beta.a), repeat(self.priors.beta.b))
        # res = map(update_beta_l_wrapper, args)
        res = self.pool.map(update_beta_l_wrapper, args)
        return np.array(list(res))

    def sample_pi(self, delta):
        shapes = cu.counter(delta, self.nMix) + self.priors.pi.a
        unnormalized = gamma(shape = shapes)
        return unnormalized / unnormalized.sum()

    def sample_r(self, zeta, delta):
        shapes = zeta[delta].sum(axis = 1)
        rates  = self.data.V.sum(axis = 1)
        return gamma(shape = shapes, scale = 1. / rates)

    def sample_zeta(self, curr_zeta, r, delta, alpha, beta):
        Y = (self.data.V.T * r).T
        # djs = [np.where(delta == j)[0] for j in range(self.nMix)]
        args = zip(
            curr_zeta,
            [Y[np.where(delta == j)[0]] for j in range(self.nMix)],
            # [Y[djs[j]] for j in range(self.nMix)],
            repeat(alpha),
            repeat(beta),
            )
        # res = map(update_zeta_j_wrapper, args)
        res = self.pool.map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def sample_delta(self, r, zeta, pi):
        Y = (self.data.V.T * r).T
        args = zip(Y, repeat(zeta), repeat(pi))
        # res = map(sample_delta_i, args)
        res = self.pool.map(sample_delta_i, args)
        return np.array(list(res)).reshape(-1)

    def initialize_sampler(self, ns):
        self.samples          = Samples(ns, self.nDat, self.nCol, self.nMix)
        self.samples.alpha[0] = 1.
        self.samples.beta[0]  = 1.
        self.samples.zeta[0]  = gamma(shape = 1., size = (self.nMix, self.nCol))
        self.samples.pi[0]    = 1. / self.nMix
        self.samples.delta[0] = self.sample_delta(
                np.ones(self.nDat),self.samples.zeta[0], self.samples.pi[0],
                )
        self.samples.r[0]     = self.sample_r(self.samples.zeta[0], self.samples.delta[0])
        self.curr_iter        = 0
        return

    def iter_sample(self):
        # Pull current values
        zeta  = self.curr_zeta
        pi    = self.curr_pi
        delta = self.curr_delta
        r     = self.curr_r
        alpha = self.curr_alpha
        beta  = self.curr_beta
        # Advance the iterator
        self.curr_iter += 1
        # Compute new values
        self.samples.delta[self.curr_iter] = self.sample_delta(r, zeta, pi)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(zeta, r, self.curr_delta, alpha, beta)
        self.samples.pi[self.curr_iter]    = self.sample_pi(self.curr_delta)
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter]  = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_zeta, self.curr_delta)
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        print_string = '\rSampling {:.1%} Completed'
        print(print_string.format(self.curr_iter / ns), end = '')
        while (self.curr_iter < ns):
            if (self.curr_iter % 10) == 0:
                print(print_string.format(self.curr_iter / ns), end = '')
            self.iter_sample()
        print(print_string.format(1))
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        nTail = self.samples.alpha.shape[0] - nBurn - 1
        if not os.path.exists(os.path.split(path)[0]):
            os.mkdir(os.path.split(path)[0])
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # declare the resulting sample set
        zetas  = self.samples.zeta[-nTail :: nThin]
        deltas = self.samples.delta[-nTail :: nThin]
        pis    = self.samples.pi[-nTail :: nThin]
        rs     = self.samples.r[-nTail :: nThin]
        alphas = self.samples.alpha[-nTail :: nThin]
        betas  = self.samples.beta[-nTail :: nThin]
        # get the output sample size
        nSamp = deltas.shape[0]
        # set up resulting data frames
        zetas_df  = pd.DataFrame(zetas.reshape(nSamp * self.nMix, self.nCol),
                                columns = ['zeta_{}'.format(i) for i in range(self.nCol)])
        deltas_df = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        pis_df    = pd.DataFrame(pis, columns = ['pi_{}'.format(i) for i in range(self.nMix)])
        rs_df     = pd.DataFrame(rs, columns = ['r_{}'.format(i) for i in range(self.nDat)])
        alphas_df = pd.DataFrame(alphas, columns = ['alpha_{}'.format(i)  for i in range(self.nCol)])
        betas_df  = pd.DataFrame(betas, columns = ['beta_{}'.format(i) for i in range(self.nCol)])
        # write to disk
        zetas_df.to_sql('zetas', conn, index = False)
        try:
            deltas_df.to_sql('deltas', conn, index = False)
            rs_df.to_sql('rs', conn, index = False)
        except sql.OperationalError:
            deltas_dft = pd.DataFrame({'delta' : deltas.reshape(-1)})
            deltas_dft.to_sql('deltas', conn, index = False)
            rs_dft = pd.DataFrame({'r' : rs.reshape(-1)})
            rs_dft.to_sql('rs', conn, index = False)

        pis_df.to_sql('pis', conn, index = False)
        alphas_df.to_sql('alphas', conn, index = False)
        betas_df.to_sql('betas', conn, index = False)
        conn.commit()
        conn.close()
        return

    def __init__(
            self,
            data,
            nMix,
            prior_alpha = GammaPrior(0.5,0.5),
            prior_beta = GammaPrior(2.,2.),
            prior_pi = DirichletPrior(1.)
            ):
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.nMix = nMix
        self.priors = Prior(prior_alpha, prior_beta, prior_pi)
        self.pool = Pool(processes = 16, initializer = limit_cpu)
        return

class Result(object):
    samples = None
    nSamp = None
    nDat = None
    nCol = None
    nMix = None

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1):
        postpred = np.empty((self.nSamp, n_per_sample, self.nCol))
        for n in range(self.nSamp):
            delta_new = choice(self.nMix, n_per_sample, p = self.samples.pi[n])
            zeta_new = self.samples.zeta[n,delta_new]
            postpred[n] = euclidean_to_simplex(
                    gamma(shape = zeta_new, size = (n_per_sample, self.nCol)),
                    )
        simplex = postpred.reshape(self.nSamp * n_per_sample, self.nCol)
        return euclidean_to_hypercube(simplex)

    def generate_posterior_predictive_angular(self, n_per_sample = 1):
        hypercube = self.generate_posterior_predictive_hypercube(n_per_sample)
        return euclidean_to_angular(hypercube)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        thetas = pd.DataFrame(
                self.generate_posterior_predictive_angular(n_per_sample),
                columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
                )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)

        deltas = pd.read_sql('select * from deltas;', conn).values.astype(int)
        pis    = pd.read_sql('select * from pis;', conn).values
        zetas  = pd.read_sql('select * from zetas;', conn).values
        alphas = pd.read_sql('select * from alphas;', conn).values
        betas  = pd.read_sql('select * from betas;', conn).values

        if len(deltas.shape) > 1:
            self.nSamp = deltas.shape[0]
            self.nDat = deltas.shape[1]
        else:
            self.nSamp = pis.shape[0]
            deltas = deltas.reshape(self.nSamp, -1)
            self.nDat = deltas.shape[1]

        self.nMix = pis.shape[1]
        self.nCol = zetas.shape[1]

        self.samples = Samples(self.nSamp, self.nDat, self.nCol, self.nMix)
        self.samples.delta = deltas
        self.samples.pi = pis
        self.samples.zeta = zetas.reshape((self.nSamp, self.nMix, self.nCol))
        self.samples.alpha = alphas
        self.samples.beta = betas
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

# EOF
