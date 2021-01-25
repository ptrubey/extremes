# from projgamma import *
from projgamma import sample_beta_fc, logdprojgamma_pre_single, logdprojgamma_pre, to_angular
from scipy.stats import gamma, beta, gmean, norm as normal, invwishart, uniform
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import loggamma
from numpy.random import choice
from collections import namedtuple
from itertools import repeat
from math import ceil, log, lgamma, exp
import numpy as np
np.seterr(under = 'ignore', over = 'raise')
import multiprocessing as mp
import sqlite3 as sql
import pandas as pd
import data as dm
import cUtility as cu
import os
import mpi4py as mpi

import pt
from pointcloud import localcov

GammaPrior      = namedtuple('GammaPrior',  'a b')
DPMPG_Prior     = namedtuple('DPMPG_Prior', 'mu Sigma beta eta')
NormalPrior     = namedtuple('NormalPrior', 'mu SCho SInv')
InvWishartPrior = namedtuple('InvWishartPrior', 'nu psi')

POOL_SIZE = 8

class Bunch(object):
    """
    Bunch object.  Object for holding arbitrary keywords
        (like a dictionary, but neater; like a namedtuple, but less restrictive.)
    """
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        return

def log_density_gamma_i(args):
    lcoss, lsins, Yl, lab = args
    ncol = ceil(lab.shape[0] / 2.)
    alpha = np.exp(lab[:ncol])
    beta = np.append([1.], np.exp(lab[ncol:]))
    return logdprojgamma_pre_single(lcoss, lsins, Yl, alpha, beta)

def log_density_theta_i(args):
    return logdprojgamma_pre_single(*args)

# def log_density_log_alpha_j(log_alpha_j, Y_j, Sigma_cho, Sigma_inv, mu, prior_beta):
#     alpha_j = np.exp(log_alpha_j)
#     sum_y, n = Y_j.sum(axis = 0), Y_j.shape[0]
#     # lp1 = + ((alpha_j - 1) * np.log(Y_j).sum(axis = 0)).sum()
#     # lp2 = - (n * loggamma(alpha_j)).sum()
#     # lp3 = + (loggamma(n * alpha_j[1:] + prior_beta.a)).sum()
#     # lp4 = - ((n * alpha_j[1:] + prior_beta.a) * np.log(Y_j.T[1:].sum(axis = 1) + prior_beta.b)).sum()
#     # lp5 = + log_density_mvnormal(log_alpha_j, mu, Sigma_cho, Sigma_inv)
#     # try:
#     #     lp = lp1+lp2+lp3+lp4+lp5
#     # except:
#     #     print('lp1: {}'.format(lp1))
#     #     print('lp2: {}'.format(lp2))
#     #     print('lp3: {}'.format(lp3))
#     #     print('lp4: {}'.format(lp4))
#     #     print('lp5: {}'.format(lp5))
#     #     raise
#     # return lp
#     lp = (
#         + ((alpha_j - 1) * np.log(Y_j).sum(axis = 0)).sum()
#         - (n * loggamma(alpha_j)).sum()
#         + (loggamma(n * alpha_j[1:] + prior_beta.a)).sum()
#         - ((n * alpha_j[1:] + prior_beta.a) * np.log(Y_j.T[1:].sum(axis = 1) + prior_beta.b)).sum()
#         + log_density_mvnormal(log_alpha_j, mu, Sigma_cho, Sigma_inv)
#         )
#     return lp

def log_density_mvnormal(x, mu, cov_chol, cov_inv):
    lp = (
        - 0.5 * 2 * np.log(np.diag(cov_chol[0])).sum()
        - 0.5 * ((x - mu).T @ cov_inv @ (x - mu)).sum()
        )
    return lp

def log_density_mvnormal_wrapper(args):
    return log_density_mvnormal(*args)

def log_density_log_alphabeta_j(log_alphabeta_j, Sigma_cho, Sigma_inv, mu, lcoss_j, lsins_j, Yl_j):
    ncol = ceil(log_alphabeta_j.shape[0] / 2.)
    alpha_j = np.exp(log_alphabeta_j[:ncol])
    beta_j = np.append([1.], np.exp(log_alphabeta_j[ncol:]))
    lp = (
        + logdprojgamma_pre(lcoss_j, lsins_j, Yl_j, alpha_j, beta_j).sum()
        + log_density_mvnormal(log_alphabeta_j, mu, Sigma_cho, Sigma_inv)
        )
    return lp

class DPMPG_Samples(object):
    log_alphabeta = None
    delta = None # numpy array; int; indicates cluster membership
    eta   = None # Dispersion variable for DP algorithm
    mu    = None # Hierarchical Mean of shapes
    Sigma = None # Covariance Matrix for Cluster Dispersion from Hierarchial Mean
    log_alpha = None
    accepted   = None
    labstack = None

    def __init__(self, nSamp, nDat, nCol):
        mu_cols = 2 * nCol - 1
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.eta   = np.empty(nSamp + 1)
        self.log_alphabeta = [None] * (nSamp + 1)
        self.mu    = np.empty((nSamp + 1, mu_cols))
        self.Sigma = np.empty((nSamp + 1, mu_cols, mu_cols))
        self.accepted = None
        self.labstack = np.empty((0, mu_cols))
        return

class DPMPG_Result(object):
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

    def generate_log_alphabeta_new(self, mu, Sigma, m):
        deviations = (cholesky(Sigma) @ normal.rvs(size = (2 * self.nCol - 1, m))).T
        return mu.reshape(1,-1) + deviations

    def generate_posterior_predictive_gammas(self, n_per_sample = 10, m = 20):
        new_gammas = []
        for i in range(self.nSamp):
            dmax   = self.samples.delta[i].max()
            njs    = cu.counter(self.samples.delta[i], dmax + 1 + m)
            ljs    = njs + (njs == 0) * self.samples.eta[i] / m
            # This part needs to be fixed.
            new_labs = self.generate_log_alphabeta_new(self.samples.mu[i], self.samples.Sigma[i], m)
            prob   = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            labs   = np.vstack((self.samples.log_alphabeta[i], new_labs))[deltas]
            alpha = np.exp(labs.T[:self.nCol].T)
            beta  = np.hstack((np.ones(labs.shape[0]).reshape(-1,1), np.exp(labs.T[self.nCol:].T)))
            new_gammas.append(gamma.rvs(alpha, scale = 1 / beta))
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
        conn   = sql.connect(path)
        mu     = pd.read_sql('select * from mu;', conn).values
        Sigma  = pd.read_sql('select * from Sigma;', conn).values
        deltas = pd.read_sql('select * from deltas;', conn).values.astype(int)
        eta    = pd.read_sql('select * from eta;', conn).values.reshape(-1)
        labs   = pd.read_sql('select * from log_alphabetas', conn)

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = ceil(mu.shape[1] / 2)
        mu_cols = mu.shape[1]

        self.samples = DPMPG_Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.mu = mu
        self.samples.Sigma = Sigma.reshape(self.nSamp, mu_cols, mu_cols)
        self.samples.delta = deltas
        self.samples.eta = eta
        self.samples.log_alphabeta = [
            labs.values[np.where(labs.iter == i)[0], 1:]
            for i in range(self.nSamp)
            ]
        return

    def __init__(self, path):
        self.load_data(path)
        mu_cols = 2 * self.nCol - 1
        prior_mu = NormalPrior(np.zeros(mu_cols), (np.sqrt(2) * np.eye(mu_cols),), 0.5 * np.eye(mu_cols))
        prior_Sigma = InvWishartPrior(mu_cols + 10, np.eye(mu_cols) * 0.5)
        prior_beta = GammaPrior(2.,2.)
        prior_eta = GammaPrior(2.,1.)
        self.priors = DPMPG_Prior(prior_mu, prior_Sigma, prior_beta, prior_eta)
        return

DPMPG_State = namedtuple('DPMPG_State','logalphabeta delta eta mu Sigma temp')

class DPMPG_Chain(pt.PTChain):
    @property
    def curr_log_alphabeta(self):
        return self.samples.log_alphabeta[self.curr_iter].copy()

    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()

    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()

    @property
    def curr_mu(self):
        return self.samples.mu[self.curr_iter].copy()

    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter].copy()

    @property
    def curr_labstack(self):
        return self.samples.labstack

    def update_stack(self):
        self.samples.labstack = np.append(self.samples.labstack, self.curr_log_alphabeta, axis = 0)
        return

    def get_state(self):
        state = DPMPG_State(self.curr_log_alphabeta, self.curr_delta, self.curr_eta,
                            self.curr_mu, self.curr_Sigma, self.temper_temp)
        return state

    def set_state(self, state):
        self.samples.log_alphabeta[self.curr_iter] = state.logalphabeta
        self.samples.delta[self.curr_iter]         = state.delta
        self.samples.eta[self.curr_iter]           = state.eta
        self.samples.mu[self.curr_iter]            = state.mu
        self.samples.Sigma[self.curr_iter]         = state.Sigma
        return

    def initialize_sampler(self, nSamp):
        self.samples = DPMPG_Samples(nSamp, self.nDat, self.nCol)
        self.curr_iter = 0
        self.samples.mu[0] = self.priors.mu.mu + self.priors.mu.SCho @ normal.rvs(size = 2 * self.nCol - 1)
        self.samples.Sigma[0] = invwishart(df = self.priors.Sigma.nu, scale = self.priors.Sigma.psi).rvs()
        self.samples.log_alphabeta[0] = self.sample_log_alphabeta_new(
                self.samples.mu[0], cho_factor(self.samples.Sigma[0]), self.nDat,
                )
        self.samples.delta[0] = range(self.nDat)
        if self.fixed_eta:
            self.samples.eta[0] = self.fixed_eta
        else:
            self.samples.eta[0] = 5.
        return

    def log_posterior_state(self, state):
        Sigma_chol = cho_factor(state.Sigma)
        Sigma_inv = cho_solve(Sigma_chol, np.eye(2 * self.nCol - 1))
        args = zip(self.data.lcoss, self.data.lsins, self.data.Yl, state.logalphabeta[state.delta])
        llik = np.array(list(map(log_density_gamma_i, args))).sum()
        args = zip(state.logalphabeta, repeat(state.mu), repeat(Sigma_chol), repeat(Sigma_inv))
        lp_lab = np.array(list(self.pool.map(log_density_mvnormal_wrapper, args))).sum()
        lp_mu = log_density_mvnormal(state.mu, self.priors.mu.mu, self.priors.mu.SCho, self.priors.mu.SInv)
        lp_Sigma = invwishart(df = self.priors.Sigma.nu, scale = self.priors.Sigma.psi).logpdf(state.Sigma)
        lp_eta = gamma(self.priors.eta.a, scale = 1 / self.priors.eta.b).logpdf(state.eta)
        return (llik + lp_lab + lp_mu + lp_Sigma + lp_eta) * self.inv_temper_temp

    def clean_delta(self, delta, log_alphabeta, i):
        assert(delta.max() + 1 == log_alphabeta.shape[0])
        _delta = np.delete(delta, i)
        nj = cu.counter(_delta, _delta.max() + 2)
        fz = cu.first_zero(nj)
        _log_alphabeta = log_alphabeta[np.where(nj > 0)[0]]
        if (fz == delta[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _log_alphabeta

    def sample_delta_i(self, delta, log_alphabeta, eta, mu, Sigma_chol, i):
        _delta, _log_alphabeta = self.clean_delta(delta, log_alphabeta, i)
        # yi = r[i] * self.data.Yl[i]
        _dmax = _delta.max()
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * (eta / self.m)
        log_alphabeta_new = self.sample_log_alphabeta_new(mu, Sigma_chol, self.m)
        lab_stack = np.vstack((_log_alphabeta, log_alphabeta_new))
        assert (lab_stack.shape[0] == ljs.shape[0])
        args = zip(
            repeat(self.data.lcoss[i]),
            repeat(self.data.lsins[i]),
            repeat(self.data.Yl[i]),
            lab_stack
            )
        # res = self.pool.map(log_density_gamma_i, args, chunksize = ceil(ljs.shape[0]/8))
        res = map(log_density_gamma_i, args)
        lps = np.array(list(res)) * self.inv_temper_temp
        lps[np.where(np.isnan(lps))] = - np.inf
        lps -= lps.max()
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + self.m + 1), 1, p = normalized)
        if dnew > _dmax:
            _log_alphabeta_ = np.vstack((_log_alphabeta, lab_stack[dnew]))
            _delta_ = np.insert(_delta, i, _dmax + 1)
        else:
            _delta_ = np.insert(_delta, i, dnew)
            _log_alphabeta_ = _log_alphabeta.copy()
        return _delta_, _log_alphabeta_

    def sample_log_alphabeta_new(self, mu, Sigma_chol, m):
        deviations = (np.triu(Sigma_chol[0]) @ normal.rvs(size = (2 * self.nCol - 1, m))).T
        return mu.reshape(1,-1) + deviations

    def sample_log_alphabeta(self, curr_log_alphabeta, delta, Sigma_cho, Sigma_inv, mu):
        nClust = delta.max() + 1
        assert (nClust == curr_log_alphabeta.shape[0])
        djs = [np.where(delta == j)[0] for j in range(nClust)]
        prop_log_alphabeta = np.empty(curr_log_alphabeta.shape)
        for j in range(nClust):
            prop_log_alphabeta[j] = self.sample_log_alphabeta_j(curr_log_alphabeta[j], djs[j],
                                                                    Sigma_cho, Sigma_inv, mu)
        return prop_log_alphabeta

    def sample_log_alphabeta_j(self, curr_log_alphabeta_j, delta_j, Sigma_cho, Sigma_inv, mu):
        mu_cols = 2 * self.nCol - 1
        curr_cov = self.localcov(curr_log_alphabeta_j)
        curr_cov_chol = cho_factor(curr_cov)
        curr_cov_inv = cho_solve(curr_cov_chol, np.eye(mu_cols))
        prop_log_alphabeta_j = curr_log_alphabeta_j + np.triu(curr_cov_chol[0]) @ normal.rvs(size = mu_cols)
        prop_cov = self.localcov(prop_log_alphabeta_j)
        prop_cov_chol = cho_factor(prop_cov)
        prop_cov_inv = cho_solve(prop_cov_chol, np.eye(mu_cols))

        curr_lp = log_density_log_alphabeta_j(
                curr_log_alphabeta_j, Sigma_cho, Sigma_inv, mu,
                self.data.lcoss[delta_j], self.data.lsins[delta_j], self.data.Yl[delta_j],
                ) * self.inv_temper_temp
        prop_lp = log_density_log_alphabeta_j(
                prop_log_alphabeta_j, Sigma_cho, Sigma_inv, mu,
                self.data.lcoss[delta_j], self.data.lsins[delta_j], self.data.Yl[delta_j],
                ) * self.inv_temper_temp
        pc_ld = log_density_mvnormal(curr_log_alphabeta_j, prop_log_alphabeta_j, prop_cov_chol, prop_cov_inv)
        cp_ld = log_density_mvnormal(prop_log_alphabeta_j, curr_log_alphabeta_j, curr_cov_chol, curr_cov_inv)

        if uniform.rvs() < prop_lp + pc_ld - curr_lp - cp_ld:
            return prop_log_alphabeta_j

        return curr_log_alphabeta_j

    def sample_eta(self, curr_eta, delta):
        nClust = delta.max() + 1
        g = beta.rvs(curr_eta + 1, nClust)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma.rvs(aaa, scale = 1/bb)

    def sample_mu(self, Sigma_inv, log_alphabeta):
        mu_cols = 2 * self.nCol - 1
        n = log_alphabeta.shape[0]
        lab_bar = log_alphabeta.mean(axis = 0)
        _Sigma = cho_solve(
            cho_factor(n * Sigma_inv * self.inv_temper_temp + self.priors.mu.SInv),
            np.eye(mu_cols),
            )
        _mu = _Sigma @ (
            + n * lab_bar @ Sigma_inv * self.inv_temper_temp
            + self.priors.mu.mu @ self.priors.mu.SInv
            ).reshape(-1)
        return _mu + cholesky(_Sigma) @ normal.rvs(size = mu_cols)

    def sample_Sigma(self, mu, log_alphabeta):
        n = log_alphabeta.shape[0]
        diff = log_alphabeta - mu
        C = sum([np.outer(diff[i], diff[i]) for i in range(log_alphabeta.shape[0])])
        _psi = self.priors.Sigma.psi + C * self.inv_temper_temp
        _nu  = self.priors.Sigma.nu + n * self.inv_temper_temp
        return invwishart.rvs(df = _nu, scale = _psi)

    def iter_sample(self):
        ''''''
        # Update the alpha stack
        self.update_stack()
        Sigma = self.curr_Sigma
        Sigma_chol = cho_factor(Sigma)
        Sigma_inv  = cho_solve(Sigma_chol, np.eye(2 * self.nCol - 1))
        log_alphabeta = self.curr_log_alphabeta
        mu    = self.curr_mu
        eta   = self.curr_eta
        delta = self.curr_delta

        self.curr_iter += 1

        for i in range(self.nDat):
            delta, log_alphabeta = self.sample_delta_i(
                    delta, log_alphabeta, eta, mu, Sigma_chol, i,
                    )
        self.samples.delta[self.curr_iter] = delta
        self.samples.log_alphabeta[self.curr_iter] = self.sample_log_alphabeta(
                log_alphabeta, self.curr_delta, Sigma_chol, Sigma_inv, mu,
                )
        self.samples.mu[self.curr_iter] = self.sample_mu(Sigma_inv, self.curr_log_alphabeta)
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_mu, self.curr_log_alphabeta)
        if self.fixed_eta:
            self.samples.eta[self.curr_iter] = self.fixed_eta
        else:
            self.samples.eta[self.curr_iter] = self.sample_eta(eta, self.curr_delta)
        return

    def localcov(self, target):
        return localcov(self.curr_labstack, target, self.radius, self.nu, self.psi0)

    def set_temperature(self, temperature):
        self.temper_temp = temperature
        self.inv_temper_temp = 1. / temperature
        self.radius = self.r0 * log(temperature + 1, 10)
        return

    def complete(self):
        self.pool.close()
        self.pool.join()
        return

    def write_to_disk(self, path, nburn, thin = 1):
        """ Write output to disk """
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # Assemble output arrays
        log_alphabetas = np.vstack([
            np.vstack((np.ones(lab.shape[0]) * i, lab.T)).T
            for i, lab in enumerate(self.samples.log_alphabeta[nburn::thin])
            ])
        deltas = self.samples.delta[nburn::thin]
        eta    = self.samples.eta[nburn::thin]
        mu     = self.samples.mu[nburn::thin]
        Sigma  = self.samples.Sigma[nburn::thin]
        # Assemble output DataFrames

        mu_cols = 2 * self.nCol - 1

        df_mu = pd.DataFrame(
            mu,
            columns = ['mu_{}'.format(i) for i in range(mu_cols)]
            )
        df_Sigma = pd.DataFrame(
            Sigma.reshape(-1, mu_cols * mu_cols),
            columns = ['Sigma_{}_{}'.format(i,j) for i in range(mu_cols) for j in range(mu_cols)]
            )
        df_log_alphabetas = pd.DataFrame(
            log_alphabetas,
            columns = (
                  ['iter']
                + ['log_alpha_{}'.format(i) for i in range(self.nCol)]
                + ['log_beta_{}'.format(i) for i in range(1, self.nCol)]
                )
            )
        df_deltas = pd.DataFrame(
            deltas,
            columns = ['delta_{}'.format(i) for i in range(self.nDat)]
            )
        df_eta = pd.DataFrame({'eta' : eta})
        # Write Dataframes to SQL Connection
        df_mu.to_sql('mu',         conn, index = False)
        df_Sigma.to_sql('Sigma',   conn, index = False)
        df_log_alphabetas.to_sql('log_alphabetas', conn, index = False)
        df_deltas.to_sql('deltas', conn, index = False)
        df_eta.to_sql('eta',       conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        return

    def __init__(
            self,
            data,
            prior_beta = GammaPrior(2.,2.),
            prior_eta = GammaPrior(2.,1.),
            m = 20,
            temperature = 1.,
            fixed_eta = False,
            ):
        self.m = m
        self.data = data
        self.nCol = data.nCol
        self.nDat = data.nDat
        if fixed_eta:
            self.fixed_eta = fixed_eta
        else:
            self.fixed_eta = False
        mu_cols = 2 * self.nCol - 1
        prior_mu = NormalPrior(
                np.zeros(mu_cols),
                (np.sqrt(2) * np.eye(mu_cols),),
                0.5 * np.eye(mu_cols),
                )
        prior_Sigma = InvWishartPrior(mu_cols + 10, np.eye(mu_cols) * 0.5)
        self.priors = DPMPG_Prior(prior_mu, prior_Sigma, prior_beta, prior_eta)
        self.r0   = 0.5
        self.psi0 = 1e-3
        self.nu   = 50
        self.set_temperature(temperature)
        self.pool = mp.Pool(processes = POOL_SIZE)
        np.seterr(under = 'ignore', over = 'raise')
        return
    pass

# EOF
