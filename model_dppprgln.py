from cProjgamma import sample_beta_fc, logdgamma_restricted
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

GammaPrior      = namedtuple('GammaPrior', 'a b')
DirichletPrior  = namedtuple('DirichletPrior', 'a')
MPRGLN_Prior    = namedtuple('MPRGLN_Prior', 'pi mu Sigma')
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
    return logdgamma_restricted(*args)

def sample_delta_i(args):
    Yi, alpha, pi = args
    argset = zip(repeat(Yi), alpha)
    lps = np.array(list(map(log_density_delta_i, argset)))
    lps[np.isnan(lps)] = - np.inf
    lps = lps - lps.max()
    unnormalized = np.exp(lps) * pi
    normalized = unnormalized / unnormalized.sum()
    return choice(pi.shape[0], 1, p = normalized)

def log_density_gamma_single(x, a, b):
    return a * log(b) - loggamma(a) + (a - 1) * log(x) - b * x

def logdgamma_wrapper(args):
    return logdgamma(*args)

def log_density_log_alpha_j(log_alpha_j, Y_j, Sigma_cho, Sigma_inv, mu):
    alpha_j = np.exp(log_alpha_j)
    sum_y, n = Y_j.sum(axis = 0), Y_j.shape[0]
    lp = (
        + ((alpha_j - 1) * np.log(Y_j).T.sum(axis = 1)).sum()
        - (n * loggamma(alpha_j)).sum()
        - (Y_j.sum(axis = 0)).sum()
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

class Samples(object):
    alpha = None # list, each entry is np.array; each row of array pertains to a cluster
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
        self.log_alpha = np.empty((0, nCol))
        self.mu    = np.empty((nSamp + 1, nCol))
        self.Sigma = np.empty((nSamp + 1, nCol, nCol))
        self.accepted = None
        return

class Result(object):
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

    def generate_alpha_new(self, mu, Sigma, m):
        log_alpha = mu.reshape(1,-1) + (cholesky(Sigma) @ normal(size = (self.nCol, m))).T
        return np.exp(log_alpha)

    def generate_posterior_predictive_gammas(self, n_per_sample = 10, m = 20):
        new_gammas = []
        for i in range(self.nSamp):
            dmax   = self.samples.delta[i].max()
            njs    = cu.counter(self.samples.delta[i], dmax + 1 + m)
            ljs    = njs + (njs == 0) * self.samples.eta[i] / m
            # This part needs to be fixed.
            new_alphas = self.generate_alpha_new(self.samples.mu[i], self.samples.Sigma[i], m)
            prob   = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            alpha  = np.vstack((self.samples.alpha[i], new_alphas))[deltas]
            new_gammas.append(gamma(shape = alpha))
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
        deltas  = pd.read_sql('select * from deltas;', conn).values.astype(int)
        rs      = pd.read_sql('select * from rs', conn).values
        eta     = pd.read_sql('select * from eta;', conn).values.T[0]

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = mu.shape[1]

        self.samples = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.mu = mu
        self.samples.Sigma = Sigma.reshape(self.nSamp, self.nCol, self.nCol)
        self.samples.delta = deltas
        self.samples.eta = eta
        self.samples.r = rs
        self.samples.alpha = [
            alphas[np.where(alphas.T[0] == i)[0], 1:]
            for i in range(self.nSamp)
            ]
        return

    def __init__(self, path):
        self.load_data(path)
        prior_mu = NormalPrior(np.zeros(self.nCol), (np.sqrt(2) * np.eye(self.nCol),), 0.5 * np.eye(self.nCol))
        prior_Sigma = InvWishartPrior(self.nCol + 10, np.eye(self.nCol) * 0.5)
        prior_eta = GammaPrior(2.,1.)
        self.priors = Prior(prior_mu, prior_Sigma, prior_eta)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

Prior = namedtuple('Prior', 'mu Sigma eta')
State = namedtuple('State', 'alphas delta eta r mu Sigma temp')

class Chain(pt.PTChain):
    @property
    def curr_alphas(self):
        return self.samples.alpha[self.curr_iter].copy()

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
        # self.samples.log_alpha = np.append(
        #         self.samples.log_alpha, np.log(self.curr_alphas), axis = 0,
        #         )
        pass
        return

    def get_state(self):
        state = State(
                self.curr_alphas, self.curr_delta, self.curr_eta,
                self.curr_r, self.curr_mu, self.curr_Sigma, self.temper_temp,
                )
        return state

    def set_state(self, state):
        self.samples.alpha[self.curr_iter] = state.alphas
        self.samples.delta[self.curr_iter] = state.delta
        self.samples.eta[self.curr_iter]   = state.eta
        self.samples.r[self.curr_iter]     = state.r
        self.samples.mu[self.curr_iter]    = state.mu
        self.samples.Sigma[self.curr_iter] = state.Sigma
        return

    def initialize_sampler(self, nSamp):
        self.samples = Samples(nSamp, self.nDat, self.nCol)
        self.curr_iter = 0
        self.samples.mu[0] = self.priors.mu.mu + self.priors.mu.SCho @ normal(size = self.nCol)
        self.samples.Sigma[0] = invwishart.rvs(
                df = self.priors.Sigma.nu, scale = self.priors.Sigma.psi,
                )
        self.samples.alpha[0] = self.sample_alpha_new(
                self.curr_mu, cho_factor(self.curr_Sigma), self.nDat,
                )
        self.samples.delta[0] = range(self.nDat)
        self.samples.r[0] = self.sample_r(self.curr_alphas, self.curr_delta)
        self.samples.eta[0] = 5.
        return

    def log_posterior_state(self, state):
        alphas = state.alphas[state.delta]
        Sigma_chol = cho_factor(state.Sigma)
        Sigma_inv = cho_solve(Sigma_chol, np.eye(self.nCol))
        Y = (self.data.Yp.T * state.r).T
        args = zip(Y, alphas)
        llik = np.array(list(map(log_density_delta_i, args))).sum()
        args = zip(
                np.log(state.alphas),
                repeat(state.mu),
                repeat(np.triu(Sigma_chol[0])),
                repeat(Sigma_inv),
                )
        lp_alpha = np.array(list(map(log_density_mvnormal_wrapper, args))).sum()
        lp_mu = log_density_mvnormal(state.mu, self.priors.mu.mu, self.priors.mu.SCho, self.priors.mu.SInv)
        lp_Sigma = invwishart(df = self.priors.Sigma.nu, scale = self.priors.Sigma.psi).logpdf(state.Sigma)
        lp_eta = log_density_gamma_single(state.eta, self.priors.eta.a, self.priors.eta.b)
        return (llik + lp_alpha + lp_mu + lp_Sigma + lp_eta) * self.inv_temper_temp

    def clean_delta(self, delta, alpha, i):
        assert(delta.max() + 1 == alpha.shape[0])
        _delta = np.delete(delta, i)
        nj = cu.counter(_delta, _delta.max() + 2)
        fz = cu.first_zero(nj)
        _alpha = alpha[np.where(nj > 0)[0]]
        if (fz == delta[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _alpha

    def sample_delta_i(self, delta, r, alpha, eta, mu, Sigma_chol, i):
        _delta, _alpha = self.clean_delta(delta, alpha, i)
        _dmax = _delta.max()
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * (eta / self.m)
        alpha_new = self.sample_alpha_new(mu, Sigma_chol, self.m)
        alpha_stack = np.vstack((_alpha, alpha_new))
        assert (alpha_stack.shape[0] == ljs.shape[0])
        Y = self.data.Yp[i] * r[i]
        args = zip(repeat(Y), alpha_stack)
        lps = np.array(list(map(log_density_delta_i, args))) * self.inv_temper_temp
        lps[np.where(np.isnan(lps))] = - np.inf
        lps -= lps.max()
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + self.m + 1), 1, p = normalized)
        if dnew > _dmax:
            _alpha_ = np.vstack((_alpha, alpha_stack[dnew]))
            _delta_ = np.insert(_delta, i, _dmax + 1)
        else:
            _delta_ = np.insert(_delta, i, dnew)
            _alpha_ = _alpha.copy()
        return _delta_, _alpha_

    def sample_alpha_new(self, mu, Sigma_chol, n):
        log_alpha = mu.reshape(1,-1) + (np.triu(Sigma_chol[0]) @ normal(size = (self.nCol, n))).T
        return np.exp(log_alpha)

    def sample_alpha_j(self, curr_alpha_j, Yj, Sigma_cho, Sigma_inv, mu):
        curr_log_alpha_j = np.log(curr_alpha_j)
        # curr_cov      = self.localcov(curr_log_alpha_j)
        curr_cov =
        curr_cov_chol = cho_factor(curr_cov)
        curr_cov_inv  = cho_solve(curr_cov_chol, np.eye(self.nCol))

        prop_log_alpha_j = curr_log_alpha_j + np.triu(curr_cov_chol[0]) @ normal(size = self.nCol)
        prop_cov      = self.localcov(prop_log_alpha_j)
        prop_cov_chol = cho_factor(prop_cov)
        prop_cov_inv  = cho_solve(prop_cov_chol, np.eye(self.nCol))

        curr_lp = log_density_log_alpha_j(
                curr_log_alpha_j, Yj, Sigma_cho, Sigma_inv, mu,
                ) * self.inv_temper_temp
        prop_lp = log_density_log_alpha_j(
                prop_log_alpha_j, Yj, Sigma_cho, Sigma_inv, mu,
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
        Y = (self.data.Yp.T * r).T
        nClust = delta.max() + 1
        assert (nClust == curr_alpha.shape[0])
        Yjs = [Y[np.where(delta == j)[0]] for j in range(nClust)]
        prop_alpha = np.empty(curr_alpha.shape)
        for j in range(nClust):
            prop_alpha[j] = self.sample_alpha_j(curr_alpha[j], Yjs[j], Sigma_cho, Sigma_inv, mu)
        return prop_alpha

    def sample_eta(self, curr_eta, delta):
        nClust = delta.max() + 1
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1/bb)

    def sample_r(self, alphas, delta):
        alpha = alphas[delta]
        As = alpha.sum(axis = 1)
        Bs = self.data.Yp.sum(axis = 1)
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
        Sigma = self.curr_Sigma
        Sigma_chol = cho_factor(Sigma)
        Sigma_inv  = cho_solve(Sigma_chol, np.eye(self.nCol))
        mu = self.curr_mu
        eta = self.curr_eta
        delta = self.curr_delta
        r = self.curr_r

        self.curr_iter += 1

        for i in range(self.nDat):
            delta, alphas = self.sample_delta_i(delta, r, alphas, eta, mu, Sigma_chol, i)

        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter] = self.sample_r(alphas, self.curr_delta)
        self.samples.alpha[self.curr_iter] = self.sample_alpha(
                alphas, self.curr_delta, self.curr_r, Sigma_chol, Sigma_inv, mu,
                )
        self.samples.mu[self.curr_iter] = self.sample_mu(Sigma_inv, self.curr_alphas)
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_mu, self.curr_alphas)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, self.curr_delta)
        return

    def localcov(self, target):
        # return localcov(self.curr_log_alpha_stack, target, self.radius, self.nu, self.psi0)
        return np.eye(self.ncol) * (0.1 / self.inv_temper_temp)**2

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
        deltas = self.samples.delta[nburn::thin]
        rs     = self.samples.r[nburn::thin]
        eta    = self.samples.eta[nburn::thin]
        mu     = self.samples.mu[nburn::thin]
        Sigma  = self.samples.Sigma[nburn::thin]

        nSamp  = deltas.shape[0]
        # Assemble output DataFrames
        df_mu     = pd.DataFrame(mu, columns = ['mu_{}'.format(i) for i in range(self.nCol)])
        df_Sigma  = pd.DataFrame(
            Sigma.reshape(nSamp * self.nCol, self.nCol),
            columns = ['Sigma_{}'.format(i) for i in range(self.nCol)]
            )
        df_alphas = pd.DataFrame(
            alphas,
            columns = ['iter'] + ['alpha_{}'.format(i) for i in range(self.nCol)],
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
        df_deltas.to_sql('deltas', conn, index = False)
        df_rs.to_sql('rs',         conn, index = False)
        df_eta.to_sql('eta',       conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        return

    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_mu    = (0., 4), # mean, sd
            prior_Sigma = (10, 0.5), # extra df, psi
            m           = 20,
            temperature = 1.,
            r0          = 0.5,
            psi0        = 1e-2,
            nu          = 20,
            p           = 10,
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
        self.priors = Prior(prior_mu_actual, prior_Sigma_actual, prior_eta)
        self.r0     = r0
        self.psi0   = psi0
        self.nu     = nu
        self.p      = p
        self.set_projection()
        self.set_temperature(temperature)
        # self.pool = mp.Pool(processes = POOL_SIZE)
        np.seterr(under = 'ignore', over = 'raise')
        return
    pass

# EOF
