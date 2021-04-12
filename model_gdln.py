from cProjgamma import sample_beta_fc, logdgamma
from scipy.stats import invwishart
from numpy.random import gamma, uniform, normal, beta
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import loggamma
from numpy.random import choice
from collections import namedtuple
from itertools import repeat
from math import ceil, log, lgamma, exp
import numpy as np
np.seterr(under = 'ignore', over = 'raise')
# import multiprocessing as mp
import sqlite3 as sql
import pandas as pd
import data as dm
import cUtility as cu
import os
import mpi4py as mpi

import pt
from pointcloud import localcov
from data import Data

GammaPrior      = namedtuple('GammaPrior',  'a b')
DirichletPrior  = namedtuple('DirichletPrior', 'a')
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

def log_density_delta_i(args):
    return logdgamma(*args)

def sample_delta_i(args):
    Yi, alpha, beta, pi = args
    argset = zip(repeat(Yi), alpha, beta)
    lps = np.array(list(map(log_density_delta_i, argset)))
    lps[np.isnan(lps)] = -np.inf
    lps -= lps.max()
    unnormalized = np.exp(lps) * pi
    normalized = unnormalized / unnormalized.sum()
    return choice(pi.shape[0], 1, p = normalized)

def log_density_gamma_single(x, a, b):
    return a * log(b) - loggamma(a) + (a - 1) * log(x) - b * x

def logdgamma_wrapper(args):
    return logdgamma(*args)

def log_density_log_alpha_j(log_alpha_j, Y_j, Sigma_cho, Sigma_inv, mu, prior_beta):
    alpha_j = np.exp(log_alpha_j)
    sum_y, n = Y_j.sum(axis = 0), Y_j.shape[0]
    lp = (
        + ((alpha_j - 1) * np.log(Y_j).sum(axis = 0)).sum()
        - (n * loggamma(alpha_j)).sum()
        + (loggamma(n * alpha_j[1:] + prior_beta.a)).sum()
        - ((n * alpha_j[1:] + prior_beta.a) * np.log(Y_j.T[1:].sum(axis = 1) + prior_beta.b)).sum()
        + log_density_mvnormal(log_alpha_j, mu, np.triu(Sigma_cho[0]), Sigma_inv)
        )
    return lp

def log_density_mvnormal(x, mu, cov_chol, cov_inv):
    lp = (
        - 0.5 * 2 * np.log(np.diag(cov_chol)).sum()
        - 0.5 * ((x - mu).T @ cov_inv @ (x - mu)).sum()
        )
    return lp

def log_density_mvnormal_wrapper(args):
    return log_density_mvnormal(*args)

def update_beta_j_wrapper(args):
    Y_j, alpha_j, beta_prior = args
    prop_beta_j = np.empty(alpha_j.shape[0])
    prop_beta_j[0] = 1.
    for i in range(1, alpha_j.shape[0]):
        prop_beta_j[i] = sample_beta_fc(alpha_j[i], Y_j.T[i], beta_prior.a, beta_prior.b)
    return prop_beta_j

class MGDLN_Samples(object):
    alpha = None
    beta  = None
    delta = None
    pi    = None
    r     = None
    mu    = None
    Sigma = None
    log_alpha = None

    def __init__(self, nSamp, nDat, nCol, nMix):
        self.alpha = np.empty((nSamp + 1, nMix, nCol))
        self.beta  = np.empty((nSamp + 1, nMix, nCol))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.pi    = np.empty((nSamp + 1, nMix))
        self.r     = np.empty((nSamp + 1, nDat))
        self.mu    = np.empty((nSamp + 1, nCol))
        self.Sigma = np.empty((nSamp + 1, nCol, nCol))
        self.log_alpha = np.empty((0, nCol))
        return

class MGDLN_Result(object):
    samples = None
    nSamp = None
    nDat = None
    nCol = None
    nMix = None

    def generate_posterior_predictive(self, n_per_sample = 2):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample)
        return dm.euclidean_to_angular(hyp)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 2):
        gnew = self.generate_posterior_predictive_gammas(n_per_sample)
        return (gnew.T / gnew.max(axis = 1)).T

    def generate_posterior_predictive_gammas(self, n_per_sample = 2):
        new_gammas = np.empty((self.nSamp, n_per_sample, self.nCol))
        for i in range(self.nSamp):
            dnew = cu.generate_indices(self.samples.pi[i], n_per_sample)
            alpha = self.samples.alpha[i][dnew]
            beta  = self.samples.beta[i][dnew]
            new_gammas[i] = gamma(shape = alpha, scale = 1/beta)
        return new_gammas.reshape(self.nSamp * n_per_sample, self.nCol)

    def write_posterior_predictive(self, path):
        thetas = pd.DataFrame(
            self.generate_posterior_predictive(),
            columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
            )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)
        mu      = pd.read_sql('select * from mus;', conn).values
        Sigma   = pd.read_sql('select * from Sigmas;', conn).values
        alphas  = pd.read_sql('select * from alphas;', conn).values
        betas   = pd.read_sql('select * from betas;', conn).values
        deltas  = pd.read_sql('select * from deltas;', conn).values.astype(int)
        rs      = pd.read_sql('select * from rs', conn).values
        pis     = pd.read_sql('select * from pis;', conn).values

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = mu.shape[1]
        self.nMix  = pis.shape[1]

        self.samples = MGDLN_Samples(self.nSamp, self.nDat, self.nCol, self.nMix)
        self.samples.mu = mu
        self.samples.Sigma = Sigma.reshape(self.nSamp, self.nCol, self.nCol)
        self.samples.delta = deltas
        self.samples.pi = pis
        self.samples.r = rs
        self.samples.alpha = alphas.reshape(self.nSamp, self.nMix, self.nCol)
        self.samples.beta = betas.reshape(self.nSamp, self.nMix, self.nCol)
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

MGDLN_State = namedtuple('MGDLN_State', 'alpha beta delta pi r mu Sigma temp')
MGDLN_Prior = namedtuple('MGDLN_Prior', 'pi beta mu Sigma')

class MGDLN_Chain(pt.PTChain):
    samples = None
    priors = None

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
    def curr_pi(self):
        return self.samples.pi[self.curr_iter].copy()
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()
    @property
    def curr_mu(self):
        return self.samples.mu[self.curr_iter].copy()
    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter].copy()
    @property
    def curr_log_alpha_stack(self):
        return self.samples.log_alpha

    def update_log_alpha_stack(self):
        self.samples.log_alpha = np.append(
                self.samples.log_alpha, np.log(self.curr_alpha), axis = 0,
                )
        return

    def get_state(self):
        state = MGDLN_State(self.curr_alpha, self.curr_beta, self.curr_delta, self.curr_pi,
                            self.curr_r, self.curr_mu, self.curr_Sigma, self.temper_temp)
        return state

    def set_state(self, state):
        self.samples.alpha[self.curr_iter] = state.alpha
        self.samples.beta[self.curr_iter]  = state.beta
        self.samples.delta[self.curr_iter] = state.delta
        self.samples.pi[self.curr_iter]    = state.pi
        self.samples.r[self.curr_iter]     = state.r
        self.samples.mu[self.curr_iter]    = state.mu
        self.samples.Sigma[self.curr_iter] = state.Sigma
        return

    def initialize_sampler(self, nSamp):
        self.samples = MGDLN_Samples(nSamp, self.nDat, self.nCol, self.nMix)
        self.curr_iter = 0
        self.samples.mu[0] = self.priors.mu.mu + self.priors.mu.SCho @ normal(size = self.nCol)
        self.samples.Sigma[0] = invwishart.rvs(df = self.priors.Sigma.nu, scale = self.priors.Sigma.psi)
        self.samples.pi[0] = 1 / self.nMix
        alpha_new, beta_new = self.sample_alpha_beta_new(self.curr_mu, cho_factor(self.curr_Sigma), self.nMix)
        self.samples.alpha[0] = alpha_new
        self.samples.beta[0] = beta_new
        self.samples.r[0] = 1.
        self.samples.delta[0] = self.sample_delta(self.curr_alpha, self.curr_beta, self.curr_r, self.curr_pi)
        return

    def log_posterior_state(self, state):
        alphas = state.alpha[state.delta]
        betas  = state.beta[state.delta]
        Sigma_chol = cho_factor(state.Sigma)
        Sigma_inv = cho_solve(Sigma_chol, np.eye(self.nCol))
        Y = (self.data.S.T * state.r).T
        args = zip(Y, alphas, betas)
        llik = np.array(list(map(log_density_delta_i, args))).sum()
        args = zip(
                np.log(state.alpha),
                repeat(state.mu),
                repeat(np.triu(Sigma_chol[0])),
                repeat(Sigma_inv),
                )
        lp_alpha = np.array(list(map(log_density_mvnormal_wrapper, args))).sum()
        args = zip(
            betas.T[1:].T,
            repeat(np.ones(self.nCol - 1) * self.priors.beta.a),
            repeat(np.ones(self.nCol - 1) * self.priors.beta.b),
            )
        lp_beta = np.array(list(map(logdgamma_wrapper, args))).sum()
        lp_mu = log_density_mvnormal(state.mu, self.priors.mu.mu, self.priors.mu.SCho, self.priors.mu.SInv)
        lp_Sigma = invwishart(df = self.priors.Sigma.nu, scale = self.priors.Sigma.psi).logpdf(state.Sigma)
        # lp_pi
        return (llik + lp_alpha + lp_beta + lp_mu + lp_Sigma) * self.inv_temper_temp

    def sample_delta(self, alpha, beta, r, pi):
        Y = (self.data.S.T * r).T
        args = zip(Y, repeat(alpha), repeat(beta), repeat(pi))
        res = map(sample_delta_i, args)
        return np.array(list(res), dtype = int).reshape(-1)

    def sample_alpha_j(self, curr_alpha_j, Yj, Sigma_cho, Sigma_inv, mu):
        curr_log_alpha_j = np.log(curr_alpha_j)
        curr_cov      = self.localcov(curr_log_alpha_j)
        curr_cov_chol = cho_factor(curr_cov)
        curr_cov_inv  = cho_solve(curr_cov_chol, np.eye(self.nCol))

        prop_log_alpha_j = curr_log_alpha_j + np.triu(curr_cov_chol[0]) @ normal(size = self.nCol)
        prop_cov      = self.localcov(prop_log_alpha_j)
        prop_cov_chol = cho_factor(prop_cov)
        prop_cov_inv  = cho_solve(prop_cov_chol, np.eye(self.nCol))

        curr_lp = log_density_log_alpha_j(
                curr_log_alpha_j, Yj, Sigma_cho, Sigma_inv, mu, self.priors.beta,
                ) * self.inv_temper_temp
        prop_lp = log_density_log_alpha_j(
                prop_log_alpha_j, Yj, Sigma_cho, Sigma_inv, mu, self.priors.beta,
                ) * self.inv_temper_temp
        pc_ld = log_density_mvnormal(
                curr_log_alpha_j, prop_log_alpha_j, np.triu(prop_cov_chol[0]), prop_cov_inv,
                )
        cp_ld = log_density_mvnormal(
                prop_log_alpha_j, curr_log_alpha_j, np.triu(curr_cov_chol[0]), curr_cov_inv,
                )

        if log(uniform()) < prop_lp + pc_ld - curr_lp - cp_ld:
            return np.exp(prop_log_alpha_j)

        return curr_alpha_j

    def sample_alpha(self, curr_alpha, delta, r, Sigma_cho, Sigma_inv, mu):
        Y = (self.data.S.T * r).T
        Yjs = [Y[np.where(delta == j)[0]] for j in range(self.nMix)]
        prop_alpha = np.empty(curr_alpha.shape)
        for j in range(self.nMix):
            prop_alpha[j] = self.sample_alpha_j(
                curr_alpha[j], Yjs[j], Sigma_cho, Sigma_inv, mu
                )
        return prop_alpha

    def sample_alpha_beta_new(self, mu, Sigma_chol, n):
        log_alpha = mu.reshape(1,-1) + (np.triu(Sigma_chol[0]) @ normal(size = (self.nCol, n))).T
        alpha = np.exp(log_alpha)
        beta = np.hstack((
            np.ones((n,1)),
            gamma(self.priors.beta.a, scale = 1/self.priors.beta.b, size = (n, self.nCol - 1)),
            ))
        return alpha, beta

    def sample_beta(self, alpha, delta, r):
        Y = (self.data.S.T * r).T
        Yjs = [Y[np.where(delta == j)[0]] for j in range(self.nMix)]
        args = zip(
                Yjs,
                alpha,
                repeat(self.priors.beta),
                )
        # res = self.pool.map(update_beta_j_wrapper, args)
        res = map(update_beta_j_wrapper, args)
        return np.array(list(res))

    def sample_r(self, alphas, betas, delta):
        alpha = alphas[delta]
        beta = betas[delta]
        As = alpha.sum(axis = 1)
        Bs = (self.data.S * beta).sum(axis = 1)
        return gamma(shape = As, scale = 1 / Bs)

    def sample_pi(self, delta):
        shapes = cu.counter(delta, self.nMix) * self.inv_temper_temp + self.priors.pi.a
        unnormalized = gamma(shape = shapes)
        return unnormalized / unnormalized.sum()

    def sample_mu(self, Sigma_inv, alphas):
        la_bar = np.log(alphas).mean(axis = 0)
        _Sigma = cho_solve(
            cho_factor(self.nMix * Sigma_inv * self.inv_temper_temp + self.priors.mu.SInv),
            np.eye(self.nCol),
            )
        _mu = _Sigma @ (
            + self.nMix * la_bar @ Sigma_inv * self.inv_temper_temp
            + self.priors.mu.mu @ self.priors.mu.SInv
            ).reshape(-1)
        return _mu + cholesky(_Sigma) @ normal(size = self.nCol)

    def sample_Sigma(self, mu, alphas):
        diff = np.log(alphas) - mu
        C = sum([np.outer(diff[i], diff[i]) for i in range(self.nMix)])
        _psi = self.priors.Sigma.psi + C * self.inv_temper_temp
        _nu  = self.priors.Sigma.nu + self.nMix * self.inv_temper_temp
        return invwishart.rvs(df = _nu, scale = _psi)

    def iter_sample(self):
        self.update_log_alpha_stack()
        alpha = self.curr_alpha
        beta  = self.curr_beta
        Sigma = self.curr_Sigma
        Sigma_cho = cho_factor(Sigma)
        Sigma_inv  = cho_solve(Sigma_cho, np.eye(self.nCol))
        mu    = self.curr_mu
        pi    = self.curr_pi
        delta = self.curr_delta
        r     = self.curr_r

        self.curr_iter += 1

        self.samples.delta[self.curr_iter] = self.sample_delta(alpha, beta, r, pi)
        self.samples.r[self.curr_iter] = self.sample_r(alpha, beta, self.curr_delta)
        self.samples.alpha[self.curr_iter] = self.sample_alpha(
                alpha, self.curr_delta, self.curr_r, Sigma_cho, Sigma_inv, mu,
                )
        self.samples.beta[self.curr_iter] = self.sample_beta(
                self.curr_alpha, self.curr_delta, self.curr_r,
                )
        self.samples.pi[self.curr_iter] = self.sample_pi(self.curr_delta)
        self.samples.mu[self.curr_iter] = self.sample_mu(Sigma_inv, self.curr_alpha)
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_mu, self.curr_alpha)
        return

    def localcov(self, target):
        return localcov(self.curr_log_alpha_stack, target, self.radius, self.nu, self.psi0)

    def set_temperature(self, temperature):
        self.temper_temp = temperature
        self.inv_temper_temp = 1. / temperature
        self.radius = self.r0 * log(temperature + 1, 10)
        return

    def complete(self):
        # self.pool.close()
        # self.pool.join()
        return

    def write_to_disk(self, path, nburn, thin = 1):
        ntail = self.samples.alpha.shape[0] - nburn - 1
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        alpha = self.samples.alpha[-ntail :: thin]
        beta  = self.samples.beta[-ntail :: thin]
        delta = self.samples.delta[-ntail :: thin]
        pi    = self.samples.pi[-ntail :: thin]
        r     = self.samples.r[-ntail :: thin]
        mu    = self.samples.mu[-ntail :: thin]
        Sigma = self.samples.Sigma[-ntail :: thin]

        nsamp = delta.shape[0]

        alpha_df = pd.DataFrame(
                alpha.reshape(nsamp * self.nMix, self.nCol),
                columns = ['alpha_{}'.format(i) for i in range(self.nCol)],
                )
        beta_df  = pd.DataFrame(
                beta.reshape(nsamp * self.nMix, self.nCol),
                columns = ['beta_{}'.format(i) for i in range(self.nCol)],
                )
        delta_df = pd.DataFrame(delta, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        pi_df    = pd.DataFrame(pi, columns = ['pi_{}'.format(i) for i in range(self.nMix)])
        r_df     = pd.DataFrame(r, columns = ['r_{}'.format(i) for i in range(self.nDat)])
        mu_df    = pd.DataFrame(mu, columns = ['mu_{}'.format(i) for i in range(self.nCol)])
        Sigma_df = pd.DataFrame(
                Sigma.reshape(nsamp, self.nCol * self.nCol),
                columns = ['Sigma_{}_{}'.format(i,j) for i in range(self.nCol) for j in range(self.nCol)],
                )

        alpha_df.to_sql('alphas', conn, index = False)
        beta_df.to_sql('betas', conn, index = False)
        delta_df.to_sql('deltas', conn, index = False)
        pi_df.to_sql('pis', conn, index = False)
        r_df.to_sql('rs', conn, index = False)
        mu_df.to_sql('mus', conn, index = False)
        Sigma_df.to_sql('Sigmas', conn, index = False)
        conn.commit()
        conn.close()
        return

    def __init__(
            self,
            data,
            nMix        = 10,
            prior_pi    = DirichletPrior(0.5),
            prior_beta  = GammaPrior(2., 2.),
            prior_mu    = (0., 4.),
            prior_Sigma = (10., 0.5),
            m           = 20,
            temperature = 1.,
            r0          = 0.5,
            psi0        = 1e-2,
            nu          = 20,
            ):
        self.m = m
        self.data = data
        self.nCol = data.nCol
        self.nDat = data.nDat
        self.nMix = nMix
        prior_mu_actual = NormalPrior(
                np.ones(self.nCol) * prior_mu[0],
                np.eye(self.nCol) * np.sqrt(prior_mu[1]),
                np.eye(self.nCol) / np.sqrt(prior_mu[1]),
                )
        prior_Sigma_actual = InvWishartPrior(
                self.nCol + prior_Sigma[0],
                np.eye(self.nCol) * prior_Sigma[1],
                )
        self.priors = MGDLN_Prior(prior_pi, prior_beta, prior_mu_actual, prior_Sigma_actual)
        self.r0 = r0
        self.psi0 = psi0
        self.nu = nu
        self.set_temperature(temperature)
        # self.pool = mp.Pool(processes = POOL_SIZE)
        np.seterr(under = 'ignore', over = 'raise')
        return
    pass

class DPGDLN_Samples(object):
    alpha = None # list, each entry is np.array; each row of array pertains to a cluster
    beta  = None # same as alpha
    delta = None # numpy array; int; indicates cluster membership
    eta   = None # Dispersion variable for DP algorithm
    r     = None # (latent) observation radius
    mu    = None # Hierarchical Mean of shapes
    Sigma = None # Covariance Matrix for Cluster Dispersion from Hierarchial Mean
    log_alpha = None
    accepted   = None

    def __init__(self, nSamp, nDat, nCol):
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        self.alpha = [None] * (nSamp + 1)
        self.beta  = [None] * (nSamp + 1)
        self.log_alpha = np.empty((0, nCol))
        self.mu    = np.empty((nSamp + 1, nCol))
        self.Sigma = np.empty((nSamp + 1, nCol, nCol))
        self.accepted = None
        return

class DPGDLN_Result(object):
    samples = None
    nSamp   = None
    nDat    = None
    nCol    = None

    def generate_posterior_predictive(self, n_per_sample = 10):
        """ Generates posterior prediction, projects to hypercube, then
        casts to angular space """
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample)
        return dm.euclidean_to_angular(hyp)

    def generate_posterior_predictive_hypercube(self, n_per_sample):
        """ Generates a posterior prediction, and projects it to the hypercube """
        gnew = self.generate_posterior_predictive_gammas(n_per_sample)
        return (gnew.T / gnew.max(axis = 1)).T

    def generate_alpha_beta_new(self, mu, Sigma, m):
        log_alpha = mu.reshape(1,-1) + (cholesky(Sigma) @ normal(size = (self.nCol, m))).T
        alpha = np.exp(log_alpha)
        beta  = np.hstack((
            np.ones((m, 1)),
            gamma(shape = self.priors.beta.a, scale = 1/self.priors.beta.b, size = (m, self.nCol - 1))
            ))
        return alpha, beta

    def generate_posterior_predictive_gammas(self, n_per_sample = 10, m = 20):
        new_gammas = []
        for i in range(self.nSamp):
            dmax   = self.samples.delta[i].max()
            njs    = cu.counter(self.samples.delta[i], dmax + 1 + m)
            ljs    = njs + (njs == 0) * self.samples.eta[i] / m
            # This part needs to be fixed.
            new_alphas, new_betas = self.generate_alpha_beta_new(self.samples.mu[i], self.samples.Sigma[i], m)
            prob   = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            alpha  = np.vstack((self.samples.alpha[i], new_alphas))[deltas]
            beta   = np.vstack((self.samples.beta[i], new_betas))[deltas]
            new_gammas.append(gamma(shape = alpha, scale = 1 / beta))
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
        mu      = pd.read_sql('select * from mu;', conn).values
        Sigma   = pd.read_sql('select * from Sigma;', conn).values
        alphas  = pd.read_sql('select * from alphas;', conn).values
        betas   = pd.read_sql('select * from betas;', conn).values
        deltas  = pd.read_sql('select * from deltas;', conn).values.astype(int)
        rs      = pd.read_sql('select * from rs', conn).values
        eta     = pd.read_sql('select * from eta;', conn).values.T[0]

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = mu.shape[1]

        self.samples = DPGDLN_Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.mu = mu
        self.samples.Sigma = Sigma.reshape(self.nSamp, self.nCol, self.nCol)
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
        prior_mu = NormalPrior(np.zeros(self.nCol), (np.sqrt(2) * np.eye(self.nCol),), 0.5 * np.eye(self.nCol))
        prior_Sigma = InvWishartPrior(self.nCol + 10, np.eye(self.nCol) * 0.5)
        prior_beta = GammaPrior(2.,2.)
        prior_eta = GammaPrior(2.,1.)
        self.priors = DPGDLN_Prior(prior_mu, prior_Sigma, prior_beta, prior_eta)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

DPGDLN_State = namedtuple('DPGDLN_State', 'alphas betas delta eta r mu Sigma temp')
DPGDLN_Prior = namedtuple('DPGDLN_Prior', 'mu Sigma beta eta')

class DPGDLN_Chain(pt.PTChain):
    @property
    def curr_alphas(self):
        return self.samples.alpha[self.curr_iter].copy()

    @property
    def curr_betas(self):
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

    @property
    def curr_mu(self):
        return self.samples.mu[self.curr_iter].copy()

    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter].copy()

    @property
    def curr_log_alpha_stack(self):
        return self.samples.log_alpha

    def update_log_alpha_stack(self):
        self.samples.log_alpha = np.append(
                self.samples.log_alpha, np.log(self.curr_alphas), axis = 0,
                )
        return

    def get_state(self):
        state = DPGDLN_State(self.curr_alphas, self.curr_betas, self.curr_delta, self.curr_eta,
                            self.curr_r, self.curr_mu, self.curr_Sigma, self.temper_temp)
        return state

    def set_state(self, state):
        self.samples.alpha[self.curr_iter] = state.alphas
        self.samples.beta[self.curr_iter]  = state.betas
        self.samples.delta[self.curr_iter] = state.delta
        self.samples.eta[self.curr_iter]   = state.eta
        self.samples.r[self.curr_iter]     = state.r
        self.samples.mu[self.curr_iter]    = state.mu
        self.samples.Sigma[self.curr_iter] = state.Sigma
        return

    def initialize_sampler(self, nSamp):
        self.samples = DPGDLN_Samples(nSamp, self.nDat, self.nCol)
        self.curr_iter = 0
        self.samples.mu[0] = self.priors.mu.mu + self.priors.mu.SCho @ normal(size = self.nCol)
        self.samples.Sigma[0] = invwishart.rvs(df = self.priors.Sigma.nu, scale = self.priors.Sigma.psi)
        alpha_new, beta_new = self.sample_alpha_beta_new(self.curr_mu, cho_factor(self.curr_Sigma), self.nDat)
        self.samples.alpha[0] = alpha_new
        self.samples.beta[0] = beta_new
        self.samples.delta[0] = range(self.nDat)
        self.samples.r[0] = self.sample_r(self.curr_alphas, self.curr_betas, self.curr_delta)
        self.samples.eta[0] = 5.
        return

    def log_posterior_state(self, state):
        alphas = state.alphas[state.delta]
        betas  = state.betas[state.delta]
        Sigma_chol = cho_factor(state.Sigma)
        Sigma_inv = cho_solve(Sigma_chol, np.eye(self.nCol))
        Y = (self.data.S.T * state.r).T
        args = zip(Y, alphas, betas)
        llik = np.array(list(map(log_density_delta_i, args))).sum()
        args = zip(
                np.log(state.alphas),
                repeat(state.mu),
                repeat(np.triu(Sigma_chol[0])),
                repeat(Sigma_inv),
                )
        lp_alpha = np.array(list(map(log_density_mvnormal_wrapper, args))).sum()
        args = zip(
            betas.T[1:].T,
            repeat(np.ones(self.nCol - 1) * self.priors.beta.a),
            repeat(np.ones(self.nCol - 1) * self.priors.beta.b),
            )
        lp_beta = np.array(list(map(logdgamma_wrapper, args))).sum()
        lp_mu = log_density_mvnormal(state.mu, self.priors.mu.mu, self.priors.mu.SCho, self.priors.mu.SInv)
        lp_Sigma = invwishart(df = self.priors.Sigma.nu, scale = self.priors.Sigma.psi).logpdf(state.Sigma)
        # lp_eta = logdgamma(state.eta, np.array([self.priors.eta.a]), np.array([self.priors.beta.b]))
        lp_eta = log_density_gamma_single(state.eta, self.priors.eta.a, self.priors.eta.b)
        return (llik + lp_alpha + lp_beta + lp_mu + lp_Sigma + lp_eta) * self.inv_temper_temp

    def clean_delta(self, delta, alpha, beta, i):
        assert(delta.max() + 1 == alpha.shape[0])
        _delta = np.delete(delta, i)
        nj = cu.counter(_delta, _delta.max() + 2)
        fz = cu.first_zero(nj)
        _alpha = alpha[np.where(nj > 0)[0]]
        _beta  = beta[np.where(nj > 0)[0]]
        if (fz == delta[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _alpha, _beta

    def sample_delta_i(self, delta, r, alpha, beta, eta, mu, Sigma_chol, i):
        _delta, _alpha, _beta = self.clean_delta(delta, alpha, beta, i)
        _dmax = _delta.max()
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * (eta / self.m)
        alpha_new, beta_new = self.sample_alpha_beta_new(mu, Sigma_chol, self.m)
        alpha_stack = np.vstack((_alpha, alpha_new))
        beta_stack = np.vstack((_beta, beta_new))
        assert (alpha_stack.shape[0] == ljs.shape[0])
        Y = self.data.S[i] * r[i]
        args = zip(repeat(Y), alpha_stack, beta_stack)
        lps = np.array(list(map(log_density_delta_i, args))) * self.inv_temper_temp
        lps[np.where(np.isnan(lps))] = - np.inf
        lps -= lps.max()
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + self.m + 1), 1, p = normalized)
        if dnew > _dmax:
            _alpha_ = np.vstack((_alpha, alpha_stack[dnew]))
            _beta_  = np.vstack((_beta, beta_stack[dnew]))
            _delta_ = np.insert(_delta, i, _dmax + 1)
        else:
            _delta_ = np.insert(_delta, i, dnew)
            _alpha_ = _alpha.copy()
            _beta_  = _beta.copy()
        return _delta_, _alpha_, _beta_

    def sample_alpha_beta_new(self, mu, Sigma_chol, n):
        log_alpha = mu.reshape(1,-1) + (np.triu(Sigma_chol[0]) @ normal(size = (self.nCol, n))).T
        alpha = np.exp(log_alpha)
        beta = np.hstack((
            np.ones((n,1)),
            gamma(self.priors.beta.a, scale = 1/self.priors.beta.b, size = (n, self.nCol - 1)),
            ))
        return alpha, beta

    def sample_alpha_j(self, curr_alpha_j, Yj, Sigma_cho, Sigma_inv, mu):
        curr_log_alpha_j = np.log(curr_alpha_j)
        curr_cov      = self.localcov(curr_log_alpha_j)
        curr_cov_chol = cho_factor(curr_cov)
        curr_cov_inv  = cho_solve(curr_cov_chol, np.eye(self.nCol))

        prop_log_alpha_j = curr_log_alpha_j + np.triu(curr_cov_chol[0]) @ normal(size = self.nCol)
        prop_cov      = self.localcov(prop_log_alpha_j)
        prop_cov_chol = cho_factor(prop_cov)
        prop_cov_inv  = cho_solve(prop_cov_chol, np.eye(self.nCol))

        curr_lp = log_density_log_alpha_j(
                curr_log_alpha_j, Yj, Sigma_cho, Sigma_inv, mu, self.priors.beta,
                ) * self.inv_temper_temp
        prop_lp = log_density_log_alpha_j(
                prop_log_alpha_j, Yj, Sigma_cho, Sigma_inv, mu, self.priors.beta,
                ) * self.inv_temper_temp
        pc_ld = log_density_mvnormal(
                curr_log_alpha_j, prop_log_alpha_j, np.triu(prop_cov_chol[0]), prop_cov_inv,
                )
        cp_ld = log_density_mvnormal(
                prop_log_alpha_j, curr_log_alpha_j, np.triu(curr_cov_chol[0]), curr_cov_inv,
                )

        if log(uniform()) < prop_lp + pc_ld - curr_lp - cp_ld:
            return np.exp(prop_log_alpha_j)

        return curr_alpha_j

    def sample_alpha(self, curr_alpha, delta, r, Sigma_cho, Sigma_inv, mu):
        Y = (self.data.S.T * r).T
        nClust = delta.max() + 1
        assert (nClust == curr_alpha.shape[0])
        Yjs = [Y[np.where(delta == j)[0]] for j in range(nClust)]
        prop_alpha = np.empty(curr_alpha.shape)
        for j in range(nClust):
            prop_alpha[j] = self.sample_alpha_j(
                curr_alpha[j], Yjs[j], Sigma_cho, Sigma_inv, mu
                )
        # args = zip(
        #     curr_alpha,
        #     Yjs,
        #     repeat(Sigma_cho),
        #     repeat(Sigma_inv),
        #     repeat(mu),
        #     repeat(self.priors.beta),
        #     repeat(self.inv_temper_temp),
        #     )
        # res = self.pool.map(update_alpha_j_wrapper, args)
        # res = map(update_alpha_j_wrapper, args)
        # return np.array(list(res))
        return prop_alpha

    def sample_beta(self, alpha, delta, r):
        Y = (self.data.S.T * r).T
        nClust = delta.max() + 1
        assert (nClust == alpha.shape[0])
        Yjs = [Y[np.where(delta == j)[0]] for j in range(nClust)]
        args = zip(
                Yjs,
                alpha,
                repeat(self.priors.beta),
                )
        # res = self.pool.map(update_beta_j_wrapper, args)
        res = map(update_beta_j_wrapper, args)
        return np.array(list(res))

    def sample_eta(self, curr_eta, delta):
        nClust = delta.max() + 1
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1/bb)

    def sample_r(self, alphas, betas, delta):
        alpha = alphas[delta]
        beta = betas[delta]
        As = alpha.sum(axis = 1)
        Bs = (self.data.S * beta).sum(axis = 1)
        return gamma(shape = As, scale = 1 / Bs)

    def sample_mu(self, Sigma_inv, alphas):
        n = alphas.shape[0]
        la_bar = np.log(alphas).mean(axis = 0)
        _Sigma = cho_solve(
            cho_factor(n * Sigma_inv * self.inv_temper_temp + self.priors.mu.SInv),
            np.eye(self.nCol),
            )
        _mu = _Sigma @ (
            + n * la_bar @ Sigma_inv * self.inv_temper_temp
            + self.priors.mu.mu @ self.priors.mu.SInv
            ).reshape(-1)
        return _mu + cholesky(_Sigma) @ normal(size = self.nCol)

    def sample_Sigma(self, mu, alphas):
        n = alphas.shape[0]
        diff = np.log(alphas) - mu
        C = sum([np.outer(diff[i], diff[i]) for i in range(alphas.shape[0])])
        _psi = self.priors.Sigma.psi + C * self.inv_temper_temp
        _nu  = self.priors.Sigma.nu + n * self.inv_temper_temp
        return invwishart.rvs(df = _nu, scale = _psi)

    def iter_sample(self):
        ''''''
        # Update the alpha stack
        self.update_log_alpha_stack()
        alphas = self.curr_alphas
        betas = self.curr_betas
        Sigma = self.curr_Sigma
        Sigma_chol = cho_factor(Sigma)
        Sigma_inv  = cho_solve(Sigma_chol, np.eye(self.nCol))
        mu = self.curr_mu
        eta = self.curr_eta
        delta = self.curr_delta
        r = self.curr_r

        self.curr_iter += 1

        for i in range(self.nDat):
            delta, alphas, betas = self.sample_delta_i(
                    delta, r, alphas, betas, eta, mu, Sigma_chol, i,
                    )

        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter] = self.sample_r(alphas, betas, self.curr_delta)
        self.samples.alpha[self.curr_iter] = self.sample_alpha(
                alphas, self.curr_delta, self.curr_r, Sigma_chol, Sigma_inv, mu,
                )
        self.samples.beta[self.curr_iter] = self.sample_beta(
                self.curr_alphas, self.curr_delta, self.curr_r,
                )
        self.samples.mu[self.curr_iter] = self.sample_mu(Sigma_inv, self.curr_alphas)
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_mu, self.curr_alphas)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, self.curr_delta)
        return

    def localcov(self, target):
        return localcov(self.curr_log_alpha_stack, target, self.radius, self.nu, self.psi0)

    def set_temperature(self, temperature):
        self.temper_temp = temperature
        self.inv_temper_temp = 1. / temperature
        self.radius = self.r0 * log(temperature + 1, 10)
        return

    def complete(self):
        # self.pool.close()
        # self.pool.join()
        return

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
        mu     = self.samples.mu[nburn::thin]
        Sigma  = self.samples.Sigma[nburn::thin]
        # Assemble output DataFrames
        df_mu     = pd.DataFrame(
            mu,
            columns = ['mu_{}'.format(i) for i in range(self.nCol)]
            )
        df_Sigma  = pd.DataFrame(
            Sigma.reshape(-1, self.nCol * self.nCol),
            columns = ['Sigma_{}_{}'.format(i,j) for i in range(self.nCol) for j in range(self.nCol)]
            )
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
        df_rs     = pd.DataFrame(
            rs,
            columns = ['r_{}'.format(i) for i in range(self.nDat)]
            )
        df_eta    = pd.DataFrame({'eta' : eta})
        # Write Dataframes to SQL Connection
        df_mu.to_sql('mu',         conn, index = False)
        df_Sigma.to_sql('Sigma',   conn, index = False)
        df_alphas.to_sql('alphas', conn, index = False)
        df_betas.to_sql('betas',   conn, index = False)
        df_deltas.to_sql('deltas', conn, index = False)
        df_rs.to_sql('rs',         conn, index = False)
        df_eta.to_sql('eta',       conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        return

    def __init__(
            self,
            data,
            prior_beta = GammaPrior(2., 2.),
            prior_eta = GammaPrior(2., 0.5),
            prior_mu = (0., 4), # mean, sd
            prior_Sigma = (10, 0.5), # extra df, psi
            m = 20,
            temperature = 1.,
            r0   = 0.5,
            psi0 = 1e-2,
            nu   = 20,
            ):
        self.m = m
        self.data = data
        self.nCol = data.nCol
        self.nDat = data.nDat
        prior_mu_actual = NormalPrior(
                np.ones(self.nCol) * prior_mu[0],
                np.eye(self.nCol) * np.sqrt(prior_mu[1]),
                np.eye(self.nCol) / np.sqrt(prior_mu[1]),
                )
        prior_Sigma_actual = InvWishartPrior(
                self.nCol + prior_Sigma[0],
                np.eye(self.nCol) * prior_Sigma[1],
                )
        self.priors = DPGDLN_Prior(prior_mu_actual, prior_Sigma_actual, prior_beta, prior_eta)
        self.r0     = r0
        self.psi0   = psi0
        self.nu     = nu
        self.set_temperature(temperature)
        # self.pool = mp.Pool(processes = POOL_SIZE)
        np.seterr(under = 'ignore', over = 'raise')
        return
    pass

# EOF
