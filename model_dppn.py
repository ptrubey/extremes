from scipy.stats import invwishart
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import erf, erfinv
from numpy.random import choice, normal, beta, gamma
from collections import namedtuple
from itertools import repeat
from math import ceil
import numpy as np
np.seterr(under = 'ignore')
import multiprocessing as mp
import sqlite3 as sql
import pandas as pd
import functools as ft
import os
from math import sqrt, log, exp, pi

from projgamma import GammaPrior
from data import Data_From_Raw as Data_From_Raw_Base, Data
import cUtility as cu

def log_density_mvnormal(args):
    Yi = args[0]
    mu = args[1]
    Sigma = args[2]
    ldSigma = cholesky_log_det(tuple(map(tuple, Sigma)))
    invSigma = cholesky_inversion(tuple(map(tuple, Sigma)))
    lp = (
        - 0.5 * (Yi - mu) @ invSigma @ (Yi - mu)
        - 0.5 * ldSigma
        )
    return lp

@ft.lru_cache(maxsize = 2048)
def cholesky_factorization(Sigma_as_tuple):
    Sigma = np.array(Sigma_as_tuple)
    return cho_factor(Sigma)

@ft.lru_cache(maxsize = 1024)
def cholesky_log_det(Sigma_as_tuple):
    cf = cholesky_factorization(Sigma_as_tuple)
    return 2 * np.log(np.diag(cf[0])).sum()

@ft.lru_cache(maxsize = 1024)
def cholesky_inversion(Sigma_as_tuple):
    cf = cholesky_factorization(Sigma_as_tuple)
    return cho_solve(cf, np.eye(cf[0].shape[0]))

class Transformer(object):
    @staticmethod
    def probit(x):
        return sqrt(2.) * erfinv(2 * x - 1)

    @staticmethod
    def invprobit(y):
        return 0.5 * (1 + erf(y / sqrt(2.)))

    @staticmethod
    def invprobitlogjac(y):
        return (- 0.5 * np.log(2 * pi) - y * y / 2.).sum()

class Data_From_Raw(Data_From_Raw_Base, Transformer):
    def cast_to_cube(self, A, eps = 1e-6):
        V = A / (pi / 2.)
        V[V > (1 - eps)] = 1 - eps
        V[V < eps] = eps
        return V

    def __init__(self, *args):
        super().__init__(*args)
        self.Vi  = self.cast_to_cube(self.A)
        self.pVi = self.probit(self.Vi)
        return

NormalPrior = namedtuple('NormalPrior', 'mu SInv')
InvWishartPrior = namedtuple('InvWishartPrior', 'nu psi')

class Samples(object):
    mu = None
    mu0 = None
    Sigma = None
    Sigma0 = None
    delta = None
    eta = None

    def __init__(self, nSamp, nDat, nCol):
        self.mu = [None] * (nSamp + 1)
        self.mu0 = np.empty((nSamp + 1, nCol - 1))
        self.Sigma = [None] * (nSamp + 1)
        self.Sigma0 = np.empty((nSamp + 1, nCol - 1, nCol - 1))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.eta = np.empty(nSamp + 1)
        return

Prior = namedtuple('Prior', 'mu Sigma mu0 Sigma0 eta')

class Chain(Transformer):
    samples = None

    @staticmethod
    def pd_matrix_inversion(mat):
        return cholesky_inversion(tuple(map(tuple, mat)))
    @staticmethod
    def pd_cholesky_decomposition(mat):
        return choleskyu_factorization(tuple(map(tuple, mat)))
    @property
    def curr_mus(self):
        return self.samples.mu[self.curr_iter].copy()
    @property
    def curr_Sigmas(self):
        return self.samples.Sigma[self.curr_iter].copy()
    @property
    def curr_mu0(self):
        return self.samples.mu0[self.curr_iter].copy()
    @property
    def curr_Sigma0(self):
        return self.samples.Sigma0[self.curr_iter].copy()
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()
    @property
    def curr_deltas(self):
        return self.samples.delta[self.curr_iter].copy()
    @property
    def curr_nClust(self):
        return self.curr_deltas.max() + 1

    def write_to_disk(self, path, nburn, thin = 1):
        if os.path.exists(path):
            os.remove(path)

        conn = sql.connect(path)

        mus_np    = np.vstack([
            np.vstack((np.ones(mus.shape[0]) * i, mus.T)).T
            for i, mus in enumerate(self.samples.mu[nburn::thin])
            ])
        Sigmas_np = np.vstack([
            np.vstack((np.ones(Sigmas.shape[0]) * i, Sigmas.reshape((Sigmas.shape[0],-1)).T)).T
            for i, Sigmas in enumerate(self.samples.Sigma[nburn::thin])
            ])
        mu0_np    = self.samples.mu0[nburn::thin]
        Sigma0_np = self.samples.Sigma0.reshape((self.samples.Sigma0.shape[0], -1))[nburn::thin]
        eta_np    = self.samples.eta[nburn::thin]
        deltas_np = self.samples.delta[nburn::thin]

        mu_cols   = ['iter'] + ['mu_{}'.format(i) for i in range(1, self.nCol)]
        mus_df    = pd.DataFrame(mus_np, columns = mu_cols)
        Sigma_cols = ['iter'] + [
            'Sigma_{}_{}'.format(i,j)
            for i in range(1, self.nCol)
            for j in range(1, self.nCol)
            ]
        Sigmas_df = pd.DataFrame(Sigmas_np, columns = Sigma_cols)

        mu0_cols = ['mu0_{}'.format(i) for i in range(1, self.nCol)]
        mu0_df   = pd.DataFrame(mu0_np, columns = mu0_cols)
        Sigma0_cols = [
            'Sigma0_{}_{}'.format(i,j)
            for i in range(1, self.nCol)
            for j in range(1, self.nCol)
            ]
        Sigma0_df = pd.DataFrame(Sigma0_np, columns = Sigma0_cols)

        delta_cols = ['delta_{}'.format(i) for i in range(1, self.nDat + 1)]
        deltas_df = pd.DataFrame(deltas_np, columns = delta_cols)
        eta_df = pd.DataFrame({'eta' : eta_np})

        # write to disk!
        mus_df.to_sql('mus', conn, index = False)
        Sigmas_df.to_sql('Sigmas', conn, index = False)
        mu0_df.to_sql('mu0', conn, index = False)
        Sigma0_df.to_sql('Sigma0', conn, index = False)
        eta_df.to_sql('eta', conn, index = False)
        deltas_df.to_sql('delta', conn, index = False)
        conn.commit()
        conn.close()
        return

    def clean_delta(self, deltas, mus, Sigmas, i):
        assert (deltas.max() + 1 == mus.shape[0])
        _deltas = np.delete(deltas, i)
        nj      = cu.counter(_deltas, _deltas.max() + 2)
        fz      = cu.first_zero(nj)
        _mus    = mus[np.where(nj > 0)[0]]
        _Sigmas = Sigmas[np.where(nj > 0)[0]]
        if (fz == deltas[i]) and (fz <= _deltas.max()):
            _deltas[_deltas > fz] = _deltas[_deltas > fz] - 1
        return _deltas, _mus, _Sigmas

    def sample_delta_i(self, deltas, mus, Sigmas, mu0, Sigma0, eta, i):
        _deltas, _mus, _Sigmas = self.clean_delta(deltas, mus, Sigmas, i)
        _dmax = _deltas.max()
        njs = cu.counter(_deltas, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * eta / self.m
        mus_new, Sigmas_new = self.sample_mu_Sigma_new(mu0, Sigma0, self.m)
        mu_stack = np.vstack((_mus, mus_new))
        Sigma_stack = np.vstack((_Sigmas, Sigmas_new))
        assert (mu_stack.shape[0] == ljs.shape[0])
        args = zip(repeat(self.data.pVi[i]), mu_stack, Sigma_stack)
        res = self.pool.map(log_density_mvnormal, args, chunksize = ceil(ljs.shape[0] / 8))
        lps = np.array(list(res))
        lps[np.where(np.isnan(lps))] = -np.inf
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + self.m + 1), 1, p = normalized)
        if dnew > _dmax:
            mu = np.vstack((_mus, mu_stack[dnew]))
            Sigma = np.vstack((_Sigmas, Sigma_stack[dnew]))
            delta = np.insert(_deltas, i, _dmax + 1)
        else:
            delta = np.insert(_deltas, i, dnew)
            mu    = _mus.copy()
            Sigma = _Sigmas.copy()
        return delta, mu, Sigma

    def sample_mu_Sigma_new(self, mu0, Sigma0, size):
        sig0_chol = cholesky(Sigma0)
        mu_new = mu0 + normal(size = (size, mu0.shape[0])) @ sig0_chol
        Sigma_new = invwishart.rvs(
                df = self.priors.Sigma.nu,
                scale = self.priors.Sigma.psi,
                size = size
                )
        return mu_new, Sigma_new

    def sample_mu_i(self, pVi, mu0, Sigma):
        pVi_bar = pVi.mean(axis = 0)
        n = pVi.shape[0]
        SigInv = self.pd_matrix_inversion(Sigma)
        _Sigma = self.pd_matrix_inversion(n * SigInv + self.priors.mu.SInv)
        _mu = _Sigma @ (n * pVi_bar @ SigInv + mu0 @ self.priors.mu.SInv)
        _Sigma_chol = cholesky(_Sigma)
        return _mu + _Sigma_chol @ normal(size = Sigma.shape[1])

    def sample_Sigma_i(self, pVi, mu):
        n = pVi.shape[0]
        _nu = self.priors.Sigma.nu + n
        _psi = self.priors.Sigma.psi + sum(np.outer(x,x) for x in (pVi - mu))
        return invwishart.rvs(df = _nu, scale = _psi)

    def sample_mu0(self, mus, Sigma0):
        mu_bar = mus.mean(axis = 0)
        n = mus.shape[0]
        Sig0Inv = self.pd_matrix_inversion(Sigma0)
        _Sigma = self.pd_matrix_inversion(n * Sig0Inv + self.priors.mu0.SInv)
        _mu    = _Sigma @ (n * mu_bar @ Sig0Inv + self.priors.mu0.mu @ self.priors.mu0.SInv)
        _Sigma_chol = cholesky(_Sigma)
        return _mu + _Sigma_chol @ normal(size = Sigma0.shape[1])

    def sample_Sigma0(self, mus, mu0):
        n = mus.shape[0]
        _nu = self.priors.Sigma0.nu + n
        _psi = self.priors.Sigma0.psi + sum(np.outer(x,x) for x in (mus - mu0))
        return invwishart.rvs(df = _nu, scale = _psi)

    def sample_eta(self, curr_eta, nClust):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def iter_sample(self):
        mus    = self.curr_mus
        Sigmas = self.curr_Sigmas
        mu0    = self.curr_mu0
        Sigma0 = self.curr_Sigma0
        eta    = self.curr_eta
        deltas = self.curr_deltas

        self.curr_iter += 1
        assert (deltas.max() + 1 == mus.shape[0])

        for i in range(self.nDat):
            deltas, mus, Sigmas = self.sample_delta_i(deltas, mus, Sigmas, mu0, Sigma0, eta, i)
            ds = np.where(deltas == deltas[i])[0]
            mus[deltas[i]] = self.sample_mu_i(self.data.pVi[ds], mu0, Sigmas[deltas[i]])
            Sigmas[deltas[i]] = self.sample_Sigma_i(self.data.pVi[ds], mus[deltas[i]])

        self.samples.delta[self.curr_iter] = deltas

        for j in range(deltas.max() + 1):
            djs = np.where(deltas == j)[0]
            mus[j] = self.sample_mu_i(self.data.pVi[djs], mu0, Sigmas[j])
            Sigmas[j] = self.sample_Sigma_i(self.data.pVi[djs], mus[j])

        self.samples.mu[self.curr_iter] = mus
        self.samples.Sigma[self.curr_iter] = Sigmas

        self.samples.mu0[self.curr_iter] = self.sample_mu0(mus, Sigma0)
        self.samples.Sigma0[self.curr_iter] = self.sample_Sigma0(mus, self.curr_mu0)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, deltas.max() + 1)
        return

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.delta[0] = np.array(list(range(self.nDat)), dtype = int)
        self.samples.eta[0] = 5
        self.samples.mu0[0] = 0.
        self.samples.Sigma0[0] = invwishart.rvs(
            df = self.priors.Sigma0.nu, scale = self.priors.Sigma0.psi,
            )
        mus, Sigmas = self.sample_mu_Sigma_new(
            self.samples.mu0[0], self.samples.Sigma0[0], self.nDat,
            )
        self.samples.mu[0] = mus
        self.samples.Sigma[0] = Sigmas
        self.curr_iter = 0
        return

    def sample(self, nsamp):
        self.initialize_sampler(nsamp)
        print_string = '\rSampling {:.1%} Completed, {} Clusters     '
        print(print_string.format(self.curr_iter / nsamp, self.curr_nClust), end = '')
        for i in range(nsamp):
            if ((i % 10) == 0):
                print(print_string.format(self.curr_iter / nsamp, self.curr_nClust), end = '')
            self.iter_sample()
        print('\rSampling 100% Completed                    ')
        return

    def __init__(
            self,
            data,
            prior_mu = (0., 1.), # NormalPrior(np.zeros(7), np.eye(7) * 1.),
            prior_mu0 = (0., 1.), # NormalPrior(np.zeros(7), np.eye(7) * 0.125),
            prior_Sigma = (5, 1.), # InvWishartPrior(10, np.eye(7) * 1),
            prior_Sigma0 = (5, 1.), #InvWishartPrior(10, np.eye(7) * 1),
            prior_eta = GammaPrior(2, 10),
            m = 20,
            ):
        self.m = m
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        mu_actual = NormalPrior(
            np.ones(self.nCol - 1) * prior_mu[0],
            np.eye(self.nCol - 1) * prior_mu[1],
            )
        mu0_actual = NormalPrior(
            np.ones(self.nCol - 1) * prior_mu0[0],
            np.eye(self.nCol - 1) * prior_mu[1],
            )
        Sigma_actual  = InvWishartPrior(
            self.nCol + prior_Sigma[0],
            np.eye(self.nCol - 1) * prior_Sigma[1],
            )
        Sigma0_actual = InvWishartPrior(
            self.nCol + prior_Sigma0[0],
            np.eye(self.nCol - 1) * prior_Sigma0[1],
            )
        self.priors = Prior(mu_actual, Sigma_actual, mu0_actual, Sigma0_actual, prior_eta)
        self.pool = mp.Pool(8)
        return

class Result(Transformer):
    samples = None
    nSamp = None
    nDat = None
    nCol = None

    def generate_posterior_predictive(self, n_per_sample):
        # hyp = self.generate_posterior_predictive_hypercube(n_per_sample)
        hyp = self.generate_posterior_predictive_probit(n_per_sample)
        return hyp * pi / 2.

    def generate_posterior_predictive_hypercube(self, n_per_sample):
        pp = self.generate_posterior_predictive_probit(n_per_sample)
        return (pp.T / pp.max(axis = 1)).T

    def generate_posterior_predictive_probit(self, n_per_sample, m = 20):
        new_pVis = []
        for i in range(self.nSamp):
            dmax = self.samples.delta[i].max()
            # njs = cu.counter(self.samples.delta[i], dmax + 1 + m)
            # ljs = njs + (njs == 0) * self.samples.eta[i] / m
            njs = cu.counter(self.samples.delta[i], dmax + 1)
            ljs = njs
            # new_mus = self.samples.mu0[i] + \
            #    self.samples.Sigma0[i] @ normal.rvs(size = (m, self.nCol))
            # new_Sigmas =
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            mus = self.samples.mu[i][deltas]
            Sigmas = self.samples.Sigma[i][deltas]
            for j in range(n_per_sample):
                new_pVis.append(mus[j] + Sigmas[j] @ normal(size = self.nCol))
        return self.invprobit(np.vstack(new_pVis))

    def write_posterior_predictive(self, path, n_per_sample = 10):
        thetas = pd.DataFrame(
            self.generate_posterior_predictive(n_per_sample),
            columns = ['theta_{}'.format(i) for i in range(1, self.nCol + 1)],
            )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)
        mus = pd.read_sql('select * from mus;', conn).values
        Sigmas = pd.read_sql('select * from Sigmas;', conn).values
        mu0 = pd.read_sql('select * from mu0;', conn).values
        Sigma0 = pd.read_sql('select * from Sigma0;', conn).values
        delta = pd.read_sql('select * from delta;', conn).values.astype(int)
        eta = pd.read_sql('select * from eta;', conn).values.reshape(-1)

        self.nSamp = eta.shape[0]
        self.nDat = delta.shape[1]
        self.nCol = mu0.shape[1]

        self.samples = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.mu = [
            mus[np.where(mus.T[0] == i)[0], 1:]
            for i in range(self.nSamp)
            ]
        self.samples.Sigma = [
            Sigs.reshape((Sigs.shape[0], self.nCol, self.nCol))
            for Sigs in [Sigmas[np.where(Sigmas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
            ]
        self.samples.mu0 = mu0
        self.samples.Sigma0 = Sigma0
        self.samples.delta = delta
        self.samples.eta = eta
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
        return

# EOF
