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

Prior = namedtuple('Prior', 'eta alpha beta xi tau')

class Samples(object):
    zeta  = None
    sigma = None
    alpha = None
    beta  = None
    xi    = None
    tau   = None
    delta = None
    r     = None
    eta   = None

    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = [None] * (nSamp + 1)
        self.sigma = [None] * (nSamp + 1)
        self.alpha = np.empty((nSamp + 1, nCol))
        self.beta  = np.empty((nSamp + 1, nCol))
        self.xi    = np.empty((nSamp + 1, nCol - 1))
        self.tau   = np.empty((nSamp + 1, nCol - 1))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        return

class Chain(object):
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
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()

    def clean_delta(self, delta, zeta, sigma, i):
        assert (delta.max() + 1 == zeta.shape[0])
        _delta = np.delete(delta, i)
        nj = cu.counter(_delta, _delta.max() + 2)
        fz = cu.first_zero(nj)
        _zeta = zeta[np.where(nj > 0)[0]]
        _sigma = sigma[np.where(nj > 0)[0]]
        if (fz == delta[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _zeta, _sigma

    def sample_zeta_sigma_new(self, alpha, beta, xi, tau, m):
        zeta = gamma(shape = alpha, scale = 1/beta, size = (m, self.nCol))
        sigma = np.hstack((
                np.ones((m, 1)),
                gamma(shape = xi, scale = 1 / tau, size = (m, self.nCol - 1)),
                ))
        return zeta, sigma

    def sample_delta_i(self, delta, r, zeta, sigma, alpha, beta, xi, tau, eta, i):
        _delta, _zeta, _sigma = self.clean_delta(delta, zeta, sigma, i)
        _dmax = _delta.max()
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * eta / self.m
        _zeta_new, _sigma_new = self.sample_zeta_sigma_new(alpha, beta, xi, tau, self.m)
        zeta_stack = np.vstack((_zeta, _zeta_new))
        sigma_stack = np.vstack((_sigma, _sigma_new))
        assert (zeta_stack.shape[0] == ljs.shape[0])
        args = zip(
            repeat(r[i] * self.data.V[i]),
            zeta_stack,
            sigma_stack,
            )
        res = map(logdgamma_wrapper, args)
        lps = np.array(list(res))
        lps[np.where(np.isnan(lps))[0]] = - np.inf
        lps -= lps.max()
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(zeta_stack.shape[0]), 1, p = normalized)
        if dnew > _dmax:
            zeta = np.vstack((_zeta, zeta_stack[dnew]))
            sigma = np.vstack((_sigma, sigma_stack[dnew]))
            delta = np.insert(_delta, i, _dmax + 1)
        else:
            delta = np.insert(_delta, i, dnew)
            zeta = _zeta.copy()
            sigma = _sigma.copy()
        return delta, zeta, sigma

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
        # res = self.pool.map(update_tau_l_wrapper, args)
        return np.array(list(res))

    def sample_r(self, delta, zeta, sigma):
        As = zeta[delta].sum(axis = 1)
        Bs = (self.data.V * sigma[delta]).sum(axis = 1)
        return gamma(shape = As, scale = 1/Bs)

    def sample_eta(self, curr_eta, delta):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max() + 1
        bb = self.priors.beta.b - log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_zeta(self, curr_zeta, r, delta, alpha, beta, xi, tau):
        Y = (self.data.V.T * r).T
        args = zip(
            curr_zeta,
            [Y[np.where(delta == j)[0]] for j in range(curr_zeta.shape[0])],
            repeat(alpha),
            repeat(beta),
            repeat(xi),
            repeat(tau),
            )
        res = map(update_zeta_j_wrapper, args)
        # res = self.pool.map(update_zeta_j_wrapper)
        return np.array(list(res))

    def sample_sigma(self, zeta, r, delta, xi, tau):
        Y = (self.data.V.T * r).T
        args = zip(
            zeta,
            [Y[np.where(delta == j)[0]] for j in range(zeta.shape[0])],
            repeat(xi),
            repeat(tau),
            )
        res = map(update_sigma_j_wrapper, args)
        # res = self.pool.map(update_sigma_j_wrapper, args)
        return np.array(list(res))

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.xi[0] = 1.
        self.samples.tau[0] = 1.
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = (self.nDat, self.nCol))
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = (self.nDat, self.nCol))
        self.samples.eta[0] = 40.
        self.samples.delta[0] = np.array(list(range(self.nDat)))
        self.samples.r[0] = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0], self.samples.sigma[0],
                )
        self.curr_iter = 0
        return

    def iter_sample(self):
        eta   = self.curr_eta
        delta = self.curr_delta
        r     = self.curr_r
        zeta  = self.curr_zeta
        sigma = self.curr_sigma
        alpha = self.curr_alpha
        beta  = self.curr_beta
        xi    = self.curr_xi
        tau   = self.curr_tau

        self.curr_iter += 1

        for i in range(self.nDat):
            delta, zeta, sigma = self.sample_delta_i(
                    delta, r, zeta, sigma, alpha, beta, xi, tau, eta, i,
                    )
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter] = self.sample_r(self.curr_delta, zeta, sigma)
        self.samples.zeta[self.curr_iter] = self.sample_zeta(
                zeta, self.curr_r, self.curr_delta, alpha, beta, xi, tau,
                )
        self.samples.sigma[self.curr_iter] = self.sample_sigma(
                zeta, self.curr_r, self.curr_delta, xi, tau,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter] = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.xi[self.curr_iter] = self.sample_xi(self.curr_sigma, xi)
        self.samples.tau[self.curr_iter] = self.sample_tau(self.curr_sigma, self.curr_xi)
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

    def write_to_disk(self, path, nBurn, nThin = 1):
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)

        zetas  = np.vstack([
            np.hstack((np.ones((zeta.shape[0], 1)) * i, zeta))
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        sigmas = np.vstack([
            np.hstack((np.ones((sigma.shape[0], 1)) * i, sigma))
            for i, sigma in enumerate(self.samples.sigma[nBurn :: nThin])
            ])
        alphas = self.samples.alpha[nBurn :: nThin]
        betas  = self.samples.beta[nBurn :: nThin]
        xis    = self.samples.xi[nBurn :: nThin]
        taus   = self.samples.tau[nBurn :: nThin]
        deltas = self.samples.delta[nBurn :: nThin]
        rs     = self.samples.r[nBurn :: nThin]
        etas   = self.samples.eta[nBurn :: nThin]

        zetas_df = pd.DataFrame(
                zetas, columns = ['iter'] + ['zeta_{}'.format(i) for i in range(self.nCol)],
                )
        sigmas_df = pd.DataFrame(
                sigmas, columns = ['iter'] + ['sigma_{}'.format(i) for i in range(self.nCol)],
                )
        alphas_df = pd.DataFrame(alphas, columns = ['alpha_{}'.format(i) for i in range(self.nCol)])
        betas_df  = pd.DataFrame(betas,  columns = ['beta_{}'.format(i)  for i in range(self.nCol)])
        xis_df    = pd.DataFrame(xis,    columns = ['xi_{}'.format(i)    for i in range(self.nCol-1)])
        taus_df   = pd.DataFrame(taus,   columns = ['tau_{}'.format(i)   for i in range(self.nCol-1)])
        deltas_df = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        rs_df     = pd.DataFrame(rs,     columns = ['r_{}'.format(i)     for i in range(self.nDat)])
        etas_df   = pd.DataFrame({'eta' : etas})

        zetas_df.to_sql('zetas',   conn, index = False)
        sigmas_df.to_sql('sigmas', conn, index = False)
        alphas_df.to_sql('alphas', conn, index = False)
        betas_df.to_sql('betas',   conn, index = False)
        xis_df.to_sql('xis',       conn, index = False)
        taus_df.to_sql('taus',     conn, index = False)
        deltas_df.to_sql('deltas', conn, index = False)
        rs_df.to_sql('rs',         conn, index = False)
        etas_df.to_sql('etas',     conn, index = False)
        conn.commit()
        conn.close()
        return

    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_alpha = GammaPrior(0.5, 0.5),
            prior_beta  = GammaPrior(2., 2.),
            prior_xi    = GammaPrior(0.5, 0.5),
            prior_tau   = GammaPrior(2., 2.),
            m = 20,
            ):
        self.data = data
        self.m = m
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_eta, prior_alpha, prior_beta, prior_xi, prior_tau)
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
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
            new_sigmas = np.hstack((
                np.ones((m, 1)),
                gamma(
                    shape = self.samples.xi[s],
                    scale = self.samples.tau[s],
                    size = (m, self.nCol - 1),
                    ),
                ))
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            sigma = np.vstack((self.samples.sigma[s], new_sigmas))[deltas]
            new_gammas.append(gamma(shape = zeta, scale = 1 / sigma))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1, m = 10):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m)
        return euclidean_to_hypercube(gammas)

    def generate_posterior_predictive_angular(self, n_per_sample = 1, m = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample, m)
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
        etas   = pd.read_sql('select * from etas;', conn).values
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

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.xi    = xis
        self.samples.tau   = taus
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.sigma = [sigmas[np.where(sigmas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.r     = rs
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

# EOF
