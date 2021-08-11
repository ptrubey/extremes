from numpy.random import choice, gamma, beta, normal, uniform
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
import pandas as pd
import os
import sqlite3 as sql
from math import ceil, log
# from multiprocessing import Pool

import cUtility as cu
from cProjgamma import sample_alpha_1_mh, sample_alpha_k_mh, sample_beta_fc, \
                        logddirichlet, logdgamma, logdgamma_restricted
from data import euclidean_to_angular, euclidean_to_simplex, euclidean_to_hypercube, Data
from projgamma import GammaPrior, DirichletPrior

def logdgamma_wrapper(args):
    return logdgamma(*args)

def sample_delta_i(args):
    y, zeta, sigma, pi = args
    args = zip(repeat(y), zeta, sigma)
    lps = np.array(list(map(logdgamma_wrapper, args)))
    lps[np.isnan(lps)] = - np.inf
    lps = lps - lps.max()
    unnormalized = np.exp(lps) * pi
    normalized = unnormalized / unnormalized.sum()
    return choice(pi.shape[0], 1, p = normalized)

def update_zeta_j_wrapper(args):
    # parse arguments
    curr_zeta_j, Y_j, alpha, beta, xi, tau = args
    prop_zeta_j = np.empty(curr_zeta_j.shape)
    prop_zeta_j[0] = sample_alpha_1_mh(curr_zeta_j[0], Y_j.T[0], alpha[0], beta[0])
    for i in range(1, curr_zeta_j.shape[0]):
        prop_zeta_j[i] = sample_alpha_k_mh(
                curr_zeta_j[i], Y_j.T[i], alpha[i], beta[i], xi[i-1], tau[i-1],
                )
    return prop_zeta_j

def update_sigma_j_wrapper(args):
    zeta_j, Y_j, xi, tau = args
    prop_sigma_j = np.empty(zeta_j.shape)
    prop_sigma_j[0] = 1.
    for i in range(1, prop_sigma_j.shape[0]):
        prop_sigma_j[i] = sample_beta_fc(zeta_j[i], Y_j.T[i], xi[i-1], tau[i-1])
    return prop_sigma_j

def update_alpha_l_wrapper(args):
    return sample_alpha_k_mh(*args)

def update_beta_l_wrapper(args):
    return sample_beta_fc(*args)

def update_xi_l_wrapper(args):
    return sample_alpha_k_mh(*args)

def update_tau_l_wrapper(args):
    return sample_beta_fc(*args)


Prior = namedtuple('Prior', 'zeta sigma')

class Samples(object):
    zeta  = None
    sigma = None
    r     = None

    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = np.empty((nSamp + 1, nCol))
        self.sigma = np.empty((nSamp + 1, nCol))
        self.r     = np.empty((nSamp + 1, nDat))
        return

class Chain(object):
    samples = None

    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter].copy()
    @property
    def curr_sigma(self):
        return self.samples.sigma[self.curr_iter].copy()
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()

    def sample_zeta(self, curr_zeta, r):
        Y = (self.data.Yp.T * r).T
        args = zip(
                curr_zeta,
                Y.T,
                repeat(self.priors.zeta.a),
                repeat(self.priors.zeta.b),
                repeat(self.priors.sigma.a),
                repeat(self.priors.sigma.b),
                )
        res = map(update_alpha_l_wrapper, args)
        return np.array(list(res))

    def sample_sigma(self, zeta, r):
        Y = (self.data.Yp.T * r).T
        prop_sigma = np.empty(self.nCol)
        prop_sigma[0] = 1.
        args = zip(zeta[1:], Y.T[1:], repeat(self.priors.sigma.a), repeat(self.priors.sigma.b))
        prop_sigma[1:] = np.array(list(map(update_beta_l_wrapper, args)))
        return prop_sigma

    def sample_r(self, zeta, sigma):
        shape = zeta.sum()
        rate  = (self.data.Yp * sigma).sum(axis = 1)
        return gamma(shape = shape, scale = 1 / rate)

    def iter_sample(self):
        zeta = self.curr_zeta
        sigma = self.curr_sigma

        self.curr_iter += 1

        self.samples.r[self.curr_iter] = self.sample_r(sigma, zeta)
        self.samples.zeta[self.curr_iter] = self.sample_zeta(zeta, self.curr_r)
        self.samples.sigma[self.curr_iter] = self.sample_sigma(self.curr_zeta, self.curr_r)
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
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)

        zeta  = self.samples.zeta[nBurn::nThin]
        sigma = self.samples.sigma[nBurn::nThin]
        r     = self.samples.r[nBurn::nThin]

        df_zeta  = pd.DataFrame(zeta,  columns = ['zeta_{}'.format(i) for i in range(self.nCol)])
        df_sigma = pd.DataFrame(sigma, columns = ['sigma_{}'.format(i) for i in range(self.nCol)])
        df_r     = pd.DataFrame(r,     columns = ['r_{}'.format(i) for i in range(self.nDat)])

        df_zeta.to_sql('zetas', conn, index = False)
        df_sigma.to_sql('sigmas', conn, index = False)
        df_r.to_sql('rs', conn, index = False)
        pass

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.curr_iter = 0
        self.samples.zeta[0] = 1.
        self.samples.sigma[0] = 1.
        self.samples.r[0] = self.sample_r(self.curr_zeta, self.curr_sigma)
        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        return

    def __init__(
            self,
            data,
            prior_zeta  = GammaPrior(0.5, 0.5),
            prior_sigma = GammaPrior(2., 2.),
            p = 10,
            ):
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_zeta, prior_sigma)
        self.set_projection()
        return

class Result(object):
    def load_data(self, path):
        conn = sql.connect(path)

        zetas = pd.read_sql('select * from zetas;', conn).values
        sigmas = pd.read_sql('select * from sigmas;', conn).values
        rs     = pd.read_sql('select * from rs;', conn).values

        self.nSamp = zetas.shape[0]
        self.nDat  = rs.shape[1]
        self.nCol  = zetas.shape[1]

        self.samples = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.zeta  = zetas
        self.samples.sigma = sigmas
        self.samples.r     = rs
        return

    def generate_posterior_predictive_gammas(self, n_per_sample):
        gnew = np.vstack([
            gamma(shape = zeta, scale = 1 / sigma, size = (n_per_sample, self.nCol))
            for zeta, sigma in zip(self.samples.zeta, self.samples.sigma)
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
