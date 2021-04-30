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

class Samples(object):
    zeta = None
    r    = None

    def __init__(self, nSamp, nDat, nCol):
        self.zeta = np.empty((nSamp + 1, nCol))
        self.r    = np.empty((nSamp + 1, nDat))
        return

Prior = namedtuple('Prior', 'zeta')

class Chain(object):
    samples = None

    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter].copy()
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.curr_iter = 0
        self.samples.zeta[0] = 1.
        self.samples.r[0] = self.sample_r(self.curr_zeta)
        return

    def sample_zeta(self, curr_zeta, r):
        Y = (self.data.Yl.T * r).T
        args = zip(curr_zeta, Y.T, repeat(self.priors.zeta.a), repeat(self.priors.zeta.b))
        res = map(update_zeta_jl_wrapper, args)
        return np.array(list(res))

    def sample_r(self, zeta):
        shapes = zeta.sum()
        rates  = self.data.Yl.sum(axis = 1)
        return gamma(shape = shapes, scale = 1. / rates, size = self.nDat)

    def iter_sample(self):
        r    = self.curr_r
        zeta = self.curr_zeta

        self.curr_iter += 1

        self.samples.zeta[self.curr_iter] = self.sample_zeta(zeta, r)
        self.samples.r[self.curr_iter]    = self.sample_r(self.curr_zeta)
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
        nTail = self.samples.zeta.shape[0] - nBurn - 1
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # declare the resulting sample set
        zetas  = self.samples.zeta[-nTail :: nThin]
        rs     = self.samples.r[-nTail :: nThin]
        # set up resulting data frames
        zetas_df  = pd.DataFrame(zetas, columns = ['zeta_{}'.format(i) for i in range(self.nCol)])
        rs_df     = pd.DataFrame(rs, columns = ['r_{}'.format(i) for i in range(self.nDat)])
        # write to disk
        zetas_df.to_sql('zetas', conn, index = False)
        rs_df.to_sql('rs', conn, index = False)
        conn.commit()
        conn.close()
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
    def load_data(self, path):
        conn = sql.connect(path)
        zetas = pd.read_sql('select * from zetas;', conn).values
        rs = pd.read_sql('select * from rs;', conn).values

        self.nSamp = zetas.shape[0]
        self.nDat = rs.shape[1]
        self.nCol = zetas.shape[1]

        self.samples = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.zeta = zetas
        self.samples.r = rs
        return

    def generate_posterior_predictive_gammas(self, n_per_sample):
        gnew = np.vstack([
            gamma(shape = zeta, size = (n_per_sample, self.nCol))
            for zeta in self.samples.zeta
            ])
        return gnew

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1):
        euc = self.generate_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_hypercube(euc)

    def generate_posterior_predictive_angular(self, n_per_sample = 1):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample)
        return euclidean_to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        thetas = pd.DataFrame(
            self.generate_posterior_predictive_angular(n_per_sample),
            columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
            )
        thetas.to_csv(path, index = False)
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

# EOF
