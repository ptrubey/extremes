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
from data import euclidean_to_simplex, euclidean_to_hypercube, euclidean_to_angular, Data
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

Prior = namedtuple('Prior', 'pi alpha beta xi tau')

class Samples(object):
    zeta  = None
    sigma = None
    alpha = None
    beta  = None
    xi    = None
    tau   = None
    delta = None
    r     = None
    pi    = None

    def __init__(self, nSamp, nDat, nCol, nMix):
        self.pi    = np.empty((nSamp + 1, nMix))
        self.zeta  = np.empty((nSamp + 1, nMix, nCol))
        self.sigma = np.empty((nSamp + 1, nMix, nCol))
        self.alpha = np.empty((nSamp + 1, nCol)) # shape and rate priors for zeta
        self.beta  = np.empty((nSamp + 1, nCol))
        self.xi    = np.empty((nSamp + 1, nCol - 1)) # shape and rate priors for sigma
        self.tau   = np.empty((nSamp + 1, nCol - 1))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        return

class Chain(object):
    samples = None
    priors  = None

    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter].copy()
    @property
    def curr_sigma(self):
        return self.samples.sigma[self.curr_iter].copy()
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter].copy()
    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter].copy()
    @property
    def curr_xi(self):
        return self.samples.xi[self.curr_iter].copy()
    @property
    def curr_tau(self):
        return self.samples.tau[self.curr_iter].copy()
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()
    @property
    def curr_pi(self):
        return self.samples.pi[self.curr_iter].copy()

    def sample_alpha(self, zeta, curr_alpha):
        args = zip(
            curr_alpha,
            zeta.T,
            repeat(self.priors.alpha.a),
            repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a),
            repeat(self.priors.beta.b),
            )
        res = map(update_alpha_l_wrapper, args)
        # res = self.pool.map(update_alpha_l_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        args = zip(
            alpha,
            zeta.T,
            repeat(self.priors.beta.a),
            repeat(self.priors.beta.b),
            )
        res = map(update_beta_l_wrapper, args)
        # res = self.pool.map(update_beta_l_wrapper, args)
        return np.array(list(res))

    def sample_xi(self, sigma, curr_xi):
        args = zip(
            curr_xi,
            sigma.T[1:],
            repeat(self.priors.xi.a),
            repeat(self.priors.xi.b),
            repeat(self.priors.tau.a),
            repeat(self.priors.tau.b),
            )
        res = map(update_xi_l_wrapper, args)
        # res = self.pool.map(update_xi_l_wrapper, args)
        return np.array(list(res))

    def sample_tau(self, sigma, xi):
        args = zip(
            xi,
            sigma.T[1:],
            repeat(self.priors.tau.a),
            repeat(self.priors.tau.b),
            )
        res = map(update_tau_l_wrapper, args)
        # res = self.pool.map(update_beta_l_wrapper, args)
        return np.array(list(res))

    def sample_zeta(self, curr_zeta, r, delta, alpha, beta, xi, tau):
        Y = (self.data.S.T * r).T
        args = zip(
            curr_zeta,
            [Y[np.where(delta == j)[0]] for j in range(self.nMix)],
            repeat(alpha),
            repeat(beta),
            repeat(xi),
            repeat(tau),
            )
        res = map(update_zeta_j_wrapper, args)
        # res = self.pool.map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def sample_sigma(self, zeta, r, delta, xi, tau):
        Y = (self.data.S.T * r).T
        args = zip(
            zeta,
            [Y[np.where(delta == j)[0]] for j in range(self.nMix)],
            repeat(xi),
            repeat(tau),
            )
        res = map(update_sigma_j_wrapper, args)
        return np.array(list(res))

    def sample_pi(self, delta):
        shapes = cu.counter(delta, self.nMix) + self.priors.pi.a
        unnormalized = gamma(shape = shapes)
        return unnormalized / unnormalized.sum()

    def sample_r(self, zeta, sigma, delta):
        As = zeta[delta].sum(axis = 1)
        Bs = (self.data.S * sigma[delta]).sum(axis = 1)
        return gamma(shape = As, scale = 1/Bs)

    def sample_delta(self, zeta, sigma, pi, r):
        Y = (self.data.S.T * r).T
        args = zip(Y, repeat(zeta), repeat(sigma), repeat(pi))
        res = map(sample_delta_i, args)
        # res = self.pool.map(sample_delta_i, args)
        return np.array(list(res)).reshape(-1)

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol, self.nMix)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.xi[0] = 1.
        self.samples.tau[0] = 1.
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = (self.nMix, self.nCol))
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = (self.nMix, self.nCol))
        self.samples.pi[0] = 1. / self.nMix
        self.samples.delta[0] = self.sample_delta(
                self.samples.zeta[0], self.samples.sigma[0], self.samples.pi[0], np.ones(self.nDat),
                )
        self.samples.r[0] = self.sample_r(
                self.samples.zeta[0], self.samples.sigma[0], self.samples.delta[0],
                )
        self.curr_iter = 0
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

    def iter_sample(self):
        pi    = self.curr_pi
        delta = self.curr_delta
        r     = self.curr_r
        zeta  = self.curr_zeta
        sigma = self.curr_sigma
        alpha = self.curr_alpha
        beta  = self.curr_beta
        xi    = self.curr_xi
        tau   = self.curr_tau

        self.curr_iter += 1

        self.samples.delta[self.curr_iter] = self.sample_delta(zeta, sigma, pi, r)
        self.samples.r[self.curr_iter] = self.sample_r(zeta, sigma, self.curr_delta)
        self.samples.pi[self.curr_iter] = self.sample_pi(self.curr_delta)
        self.samples.zeta[self.curr_iter] = self.sample_zeta(
            zeta, self.curr_r, self.curr_delta, alpha, beta, xi, tau,
            )
        self.samples.sigma[self.curr_iter] = self.sample_sigma(
            self.curr_zeta, self.curr_r, self.curr_delta, xi, tau,
            )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter] = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.xi[self.curr_iter] = self.sample_xi(self.curr_sigma, xi)
        self.samples.tau[self.curr_iter] = self.sample_tau(self.curr_sigma, self.curr_xi)
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # Subset to resulting sample set
        zetas  = self.samples.zeta[nBurn::nThin]
        sigmas = self.samples.sigma[nBurn::nThin]
        alphas = self.samples.alpha[nBurn::nThin]
        betas  = self.samples.beta[nBurn::nThin]
        xis    = self.samples.xi[nBurn::nThin]
        taus   = self.samples.tau[nBurn::nThin]
        deltas = self.samples.delta[nBurn::nThin]
        rs     = self.samples.r[nBurn::nThin]
        pis    = self.samples.pi[nBurn::nThin]
        # Set up Data Frames
        zetas_df = pd.DataFrame(
                zetas.reshape(zetas.shape[0] * self.nMix, self.nCol),
                columns = ['zeta_{}'.format(i) for i in range(self.nCol)],
                )
        sigmas_df = pd.DataFrame(
                sigmas.reshape(sigmas.shape[0] * self.nMix, self.nCol),
                columns = ['sigma_{}'.format(i) for i in range(self.nCol)],
                )
        alphas_df = pd.DataFrame(alphas, columns = ['alpha_{}'.format(i) for i in range(self.nCol)])
        betas_df  = pd.DataFrame(betas,  columns = ['beta_{}'.format(i)  for i in range(self.nCol)])
        xis_df    = pd.DataFrame(xis,    columns = ['xi_{}'.format(i)    for i in range(self.nCol-1)])
        taus_df   = pd.DataFrame(taus,   columns = ['tau_{}'.format(i)   for i in range(self.nCol-1)])
        deltas_df = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        rs_df     = pd.DataFrame(rs,     columns = ['r_{}'.format(i)     for i in range(self.nDat)])
        pis_df    = pd.DataFrame(pis,    columns = ['pi_{}'.format(i)    for i in range(self.nMix)])
        # Write to Disk
        zetas_df.to_sql('zetas',   conn, index = False)
        sigmas_df.to_sql('sigmas', conn, index = False)
        alphas_df.to_sql('alphas', conn, index = False)
        betas_df.to_sql('betas',   conn, index = False)
        xis_df.to_sql('xis',       conn, index = False)
        taus_df.to_sql('taus',     conn, index = False)
        deltas_df.to_sql('deltas', conn, index = False)
        rs_df.to_sql('rs',         conn, index = False)
        pis_df.to_sql('pis',       conn, index = False)
        conn.commit()
        conn.close()
        return

    def __init__(
            self,
            data,
            nMix,
            prior_pi    = DirichletPrior(0.5,),
            prior_alpha = GammaPrior(0.5, 0.5),
            prior_beta  = GammaPrior(2., 2.),
            prior_xi    = GammaPrior(0.5, 0.5),
            prior_tau   = GammaPrior(2., 2.),
            ):
        self.data = data
        self.nMix = nMix
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_pi, prior_alpha, prior_beta, prior_xi, prior_tau)
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample):
        dnew = np.array(list(map(lambda x: cu.generate_indices(x, n_per_sample), self.samples.pi)))
        gnew = np.vstack([
            gamma(zeta[delta], scale = 1 / sigma[delta])
            for zeta, sigma, delta in
            zip(self.samples.zeta, self.samples.sigma, dnew)
            ])
        return gnew

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1):
        euc = self.generate_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_hypercube(euc)

    def generate_posterior_predictive_angular(self, n_per_sample):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample)
        return euclidean_to_angular(hyp)

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
        sigmas = pd.read_sql('select * from sigmas;', conn).values
        alphas = pd.read_sql('select * from alphas;', conn).values
        betas  = pd.read_sql('select * from betas;', conn).values
        xis    = pd.read_sql('select * from xis;', conn).values
        taus   = pd.read_sql('select * from taus;', conn).values
        rs     = pd.read_sql('select * from rs;', conn).values

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = alphas.shape[1]
        self.nMix  = pis.shape[1]

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol, self.nMix)
        self.samples.delta = deltas
        self.samples.pi    = pis
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.xi    = xis
        self.samples.tau   = taus
        self.samples.zeta  = zetas.reshape((self.nSamp, self.nMix, self.nCol))
        self.samples.sigma = sigmas.reshape((self.nSamp, self.nMix, self.nCol))
        self.samples.r     = rs
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

# EOF
