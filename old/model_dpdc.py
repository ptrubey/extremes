from numpy.random import choice, gamma, beta, binomial, normal, uniform
from scipy.special import loggamma
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
from projgamma import GammaPrior, DirichletPrior, BetaPrior

def dirichlet_logdensity_wrapper(args):
    return logddirichlet(*args)

def sample_delta_i(args):
    Xi = args[0]
    zeta = args[1]
    pi = args[2]
    argset = zip(repeat(Xi), zeta)
    # lps = np.array(list(map(lambda arg: ddirichlet_single(*arg), argset)))
    lps = np.array(list(map(dirichlet_logdensity_wrapper, argset)))
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

def update_zeta_j(curr_zeta_j, Yj, alpha, beta, gamma):
    """
    Wrapper for projgamma.sample_alpha_1_mh
    sample_alpha_1_mh assumes a Gamma likelihood with rate parameter = 1, and a gamma
    prior for the shape parameter.
    """
    # priors = [GammaPrior(a,b) for a,b in zip(alpha,beta)]
    anew = alpha[0] * (1 - gamma) + alpha[1] * gamma
    bnew = beta[0] * (1 - gamma) + beta[1] * gamma
    args = zip(curr_zeta_j, Yj.T, anew, bnew)
    res = map(update_zeta_jl_wrapper, args)
    return np.array(list(res))

def update_zeta_j_wrapper(args):
    return update_zeta_j(*args)

# Functions and definitions relating to DP Mixture

def log_density_delta_i(args):
    return logdgamma_restricted(args[0], args[1])

def log_density_gamma_j(args):
    # zeta_j, alpha, beta
    # zeta_j = (self.nCol)
    # alpha, beta = (2, self.nCol)
    zeta_j, alpha, beta = args # parse arguments
    lps = (
        + alpha * np.log(beta)
        - loggamma(alpha)
        + (alpha - 1) * np.log(zeta_j)
        - beta * zeta_j
        ) # calculate log-density
    lps[np.isnan(lps)] = - np.inf # if failed to calculate, set density to 0
    lps -= lps.max()              # remove possible overflows
    return lps


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
        self.alpha = np.empty((nSamp + 1, 2, nCol))
        self.beta  = np.empty((nSamp + 1, 2, nCol))
        self.rho   = np.empty((nSamp + 1, nCol))
        self.zeta  = [None] * (nSamp + 1) # shape parameters (cluster)
        self.gamma = [None] * (nSamp + 1) # cone identifier (cluster)
        return

Prior = namedtuple('Prior', 'alpha beta eta rho')

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 10, m = 20):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = cu.counter(self.samples.delta[s], int(dmax + 1 + m))
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            gnew = binomial(size = (m, self.nCol), n = 1, p = self.samples.rho[s])
            anew = self.samples.alpha[s][0] * (1 - gnew) + self.samples.alpha[s][1] * gnew
            bnew = self.samples.beta[s][0] * (1 - gnew) + self.samples.beta[s][1] * gnew
            new_zetas = gamma(shape = anew, scale = 1. / bnew, size = (m, self.nCol))
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
        rs     = pd.read_sql('select * from rs;', conn).values
        etas   = pd.read_sql('select * from etas;', conn).values.reshape(-1)
        rhos   = pd.read_sql('select * from rhos;', conn).values
        gammas = pd.read_sql('select * from gammas;', conn).values

        self.nDat  = deltas.shape[1]
        self.nSamp = deltas.shape[0]
        self.nCol  = rhos.shape[1]

        alphas = alphas.reshape(self.nSamp, 2, self.nCol)
        betas  = betas.reshape(self.nSamp, 2, self.nCol)

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.r     = rs
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.rho   = rhos
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.gamma = [gammas[np.where(gammas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
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
    @property
    def curr_gamma(self):
        return self.samples.gamma[self.curr_iter].copy()
    @property
    def curr_rho(self):
        return self.samples.rho[self.curr_iter].copy()

    def clean_delta(self, delta, zeta, gamma, i):
        assert (delta.max() + 1 == zeta.shape[0])
        _delta = np.delete(delta, i)
        nj = cu.counter(_delta, _delta.max() + 2)
        fz = cu.first_zero(nj)
        _zeta = zeta[np.where(nj > 0)[0]]
        _gamma = gamma[np.where(nj > 0)[0]]
        if (fz == delta[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _zeta, _gamma

    def sample_zeta_gamma_new(self, m, alpha, beta, rho):
        gnew = binomial(size = (m, self.nCol), n = 1, p = rho)
        anew = alpha[0] * (1 - gnew) + alpha[1] * gnew
        bnew = beta[0] * (1 - gnew) + beta[1] * gnew
        return gamma(shape = anew, scale = 1. / bnew, size = (m, self.nCol)), gnew

    def sample_zeta(self, curr_zeta, delta, r, alpha, beta, gamma):
        Y = (self.data.S.T * r).T
        djs = [np.where(delta == j)[0] for j in range(delta.max() + 1)]
        args = zip(
            curr_zeta,
            [Y[djs[j]] for j in range(delta.max() + 1)],
            repeat(alpha),
            repeat(beta),
            gamma,
            )
        res = map(update_zeta_j_wrapper, args)
        # res = self.pool.map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def sample_rho(self, gamma):
        shape1 = (1 - gamma).sum(axis = 0) + self.priors.rho.a
        shape2 = gamma.sum(axis = 0) + self.priors.rho.b
        return beta(shape1, shape2, size = self.nCol)

    def sample_alpha(self, zeta, curr_alpha, gamma):
        # zeta  = (m, self.nCol)
        # gamma = (m, self.nCol)
        zgt   = (zeta * gamma).T
        zngt  = (zeta * (1 - gamma)).T
        args1 = zip(
            curr_alpha[0],
            [zngt[i][np.where(zngt[i] > 0)[0]] for i in range(self.nCol)],
            repeat(self.priors.alpha.a[0]),
            repeat(self.priors.alpha.b[0]),
            repeat(self.priors.beta.a[0]),
            repeat(self.priors.beta.b[0]),
            )
        res1 = map(update_alpha_l_wrapper, args1)
        args2 = zip(
            curr_alpha[1],
            [zgt[i][np.where(zgt[i] > 0)[0]] for i in range(self.nCol)],
            repeat(self.priors.alpha.a[1]),
            repeat(self.priors.alpha.b[1]),
            repeat(self.priors.beta.a[1]),
            repeat(self.priors.beta.b[1]),
            )
        res2  = map(update_alpha_l_wrapper, args2)
        return np.vstack((np.array(list(res1)), np.array(list(res2))))

    def sample_beta(self, zeta, alpha, gamma):
        zgt   = (zeta * gamma).T
        zngt  = (zeta * (1 - gamma)).T
        args1 = zip(
            alpha[0].T,
            [zngt[i][np.where(zngt[i] > 0)[0]] for i in range(self.nCol)],
            repeat(self.priors.beta.a[0]),
            repeat(self.priors.beta.b[0]),
            )
        res1  = map(update_beta_l_wrapper, args1)
        args2 = zip(
            alpha[1].T,
            [zgt[i][np.where(zgt[i] > 0)[0]] for i in range(self.nCol)],
            repeat(self.priors.beta.a[1]),
            repeat(self.priors.beta.b[1]),
            )
        res2  = map(update_beta_l_wrapper, args2)
        return np.vstack((np.array(list(res1)), np.array(list(res2))))

    def sample_r(self, delta, zeta):
        shapes = zeta[delta].sum(axis = 1)
        return gamma(shape = shapes)

    def sample_eta(self, curr_eta, delta):
        nClust = delta.max() + 1
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_gamma(self, zeta, rho, alpha, beta):
        args = zip(zeta, repeat(alpha), repeat(beta))
        lps = np.array(list(map(log_density_gamma_j, args))) # should be (m x 2 x self.nCol)
        unnormalized = lps * np.vstack((1 - rho, rho)).reshape(1, 2, self.nCol)
        normalized   = unnormalized[:,1] / (unnormalized[:,0] + unnormalized[:,1])
        return binomial(size = (zeta.shape[0], self.nCol), n = 1, p = normalized)

    def sample_delta_i(self, delta, zeta, gamma, r, alpha, beta, rho, eta, i):
        _delta, _zeta, _gamma = self.clean_delta(delta, zeta, gamma, i)
        _dmax = _delta.max()
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * eta / self.m
        _zeta_new, _gamma_new = self.sample_zeta_gamma_new(self.m, alpha, beta, rho)
        zeta_stack = np.vstack((_zeta, _zeta_new))
        gamma_stack = np.vstack((_gamma, _gamma_new))
        assert (zeta_stack.shape[0] == ljs.shape[0])
        args = zip(
            repeat(r[i] * self.data.S[i]),
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
            gamma = np.vstack((_gamma, gamma_stack[dnew]))
            delta = np.insert(_delta, i, _dmax + 1)
        else:
            delta = np.insert(_delta, i, dnew)
            zeta = _zeta.copy()
            gamma = _gamma.copy()
        return delta, zeta, gamma

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.delta[0] = np.array(list(range(self.nDat)), dtype = int)
        self.samples.r[0]     = 1.
        self.samples.eta[0]   = 40.
        self.samples.alpha[0][0] = 0.1
        self.samples.alpha[0][1] =  1.
        self.samples.beta[0]  = 1.
        self.samples.rho[0]   = 0.5
        self.samples.zeta[0], self.samples.gamma[0] = self.sample_zeta_gamma_new(
                self.nDat, self.samples.alpha[0], self.samples.beta[0], self.samples.rho[0],
                )
        self.curr_iter = 0
        return

    def iter_sample(self):
        """ Advance the sampler one iteration """
        # Parse the current values
        zeta  = self.curr_zeta
        gamma = self.curr_gamma
        rho   = self.curr_rho
        delta = self.curr_delta
        eta   = self.curr_eta
        r     = self.curr_r
        alpha = self.curr_alpha
        beta  = self.curr_beta
        # Advance the iterator
        self.curr_iter += 1
        # Compute new cluster assignments based on current values of zeta (and new as needed)
        for i in range(self.nDat):
            delta, zeta, gamma = self.sample_delta_i(delta, zeta, gamma, r, alpha, beta, rho, eta, i)
        # Update sampler with new values
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter] = self.sample_r(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter] = self.sample_zeta(
                zeta, self.curr_delta, self.curr_r, alpha, beta, gamma,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha, gamma)
        self.samples.beta[self.curr_iter]  = self.sample_beta(self.curr_zeta, self.curr_alpha, gamma)
        self.samples.eta[self.curr_iter]   = self.sample_eta(eta, self.curr_delta)
        self.samples.gamma[self.curr_iter] = self.sample_gamma(
                self.curr_zeta, rho, self.curr_alpha, self.curr_beta,
                )
        self.samples.rho[self.curr_iter]   = self.sample_rho(self.curr_gamma)
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
        gammas = np.vstack([
            np.hstack((np.ones((gamma.shape[0], 1)) * i, gamma))
            for i, gamma in enumerate(self.samples.gamma[nburn::thin])
        ])
        rhos   = self.samples.rho[nburn::thin]
        alphas = self.samples.alpha[nburn::thin].reshape(-1,self.nCol)
        betas  = self.samples.beta[nburn::thin].reshape(-1,self.nCol)
        etas   = self.samples.eta[nburn::thin]
        deltas = self.samples.delta[nburn::thin]
        rs     = self.samples.r[nburn::thin]
        # Assemble Output DataFrames
        df_zeta  = pd.DataFrame(
                zetas, columns = ['iter'] + ['zeta_{}'.format(i) for i in range(self.nCol)],
                )
        df_gamma = pd.DataFrame(
                gammas, columns = ['iter'] + ['gamma_{}'.format(i) for i in range(self.nCol)],
                )
        df_alpha = pd.DataFrame(alphas, columns = ['alpha_{}'.format(i) for i in range(self.nCol)])
        df_beta  = pd.DataFrame(betas, columns = ['beta_{}'.format(i) for i in range(self.nCol)])
        df_eta   = pd.DataFrame({'eta' : etas})
        df_r     = pd.DataFrame(rs, columns = ['r_{}'.format(i) for i in range(self.nDat)])
        df_delta = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        df_rho   = pd.DataFrame(rhos, columns = ['rho_{}'.format(i) for i in range(self.nCol)])
        # Write DataFrames to SQL Connection
        df_zeta.to_sql('zetas', conn, index = False)
        df_alpha.to_sql('alphas', conn, index = False)
        df_beta.to_sql('betas', conn, index = False)
        df_eta.to_sql('etas', conn, index = False)
        df_r.to_sql('rs', conn, index = False)
        df_delta.to_sql('deltas', conn, index = False)
        df_rho.to_sql('rhos', conn, index = False)
        df_gamma.to_sql('gammas', conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        pass

    def __init__(
            self,
            data,
            prior_alpha = GammaPrior((0.1, 1.), (1., 1.)),
            prior_beta = GammaPrior((1., 1.), (1.,1.)),
            prior_eta = GammaPrior(2.,0.5),
            prior_rho = BetaPrior(0.5,0.5),
            m = 20,
            ):
        self.m = m
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_alpha, prior_beta, prior_eta, prior_rho)
        # self.pool = Pool(8)
        return

# EOF
