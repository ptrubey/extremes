from scipy.stats import gamma, beta, gmean, norm as normal, invwishart, uniform, dirichlet
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import loggamma
from numpy.random import choice, gamma, beta, normal, uniform
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
from numpy import log, exp
from scipy.special import loggamma
from data import *
# from multiprocessing import Pool
import os
import sqlite3 as sql
from math import ceil

import cUtility as cu
from projgamma import sample_alpha_1_mh as sample_eta_jl
from projgamma import sample_alpha_1_mh, sample_alpha_k_mh, sample_beta_fc

epsilon = 1e-20

GammaPrior = namedtuple('GammaPrior','a b')
DirichletPrior = namedtuple('DirichletPrior','a')

def ddirichlet_single(X, alpha, log = True):
    if log:
        return dirichlet(alpha).logpdf(X)
    else:
        return dirichlet(alpha).pdf(X)

def sample_delta_i(args):
    Xi = args[0]
    zeta = args[1]
    pi = args[2]
    argset = zip(repeat(Xi), zeta)
    lps = np.array(list(map(lambda arg: ddirichlet_single(*arg), argset)))
    lps[np.isnan(lps)] = - np.inf
    lps = lps - lps.max()
    unnormalized = np.exp(lps) * pi
    normalized = unnormalized / unnormalized.sum()
    return choice(pi.shape[0], 1, p = normalized)

def to_simplex(data):
    return ((data + epsilon).T / (data + epsilon).sum(axis = 1)).T

def to_hypercube(data):
    return (data.T / data.max(axis = 1)).T

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
    priors = [GammaPrior(a,b) for a,b in zip(alpha,beta)]
    args = zip(curr_zeta_j, Yj.T, priors)
    res = map(update_zeta_jl_wrapper, args)
    return np.array(list(res))

def update_zeta_j_wrapper(args):
    return update_zeta_j(*args)

class FMIX_Samples(object):
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

class FMIX_Prior(object):
    alpha = None
    beta  = None
    pi    = None

    def __init__(self, alpha_prior, beta_prior, pi_prior):
        self.alpha = alpha_prior
        self.beta  = beta_prior
        self.pi    = pi_prior
        return

class FMIX_Chain(object):
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
        args = zip(curr_alpha, zeta.T, repeat(self.priors.alpha), repeat(self.priors.beta))
        res = map(update_alpha_l_wrapper, args)
        # res self.pool.map(update_alpha_l_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        args = zip(alpha, zeta.T, repeat(self.priors.beta))
        res = map(update_beta_l_wrapper, args)
        # res = self.pool.map(update_beta_l_wrapper, args)
        return np.array(list(res))

    def sample_pi(self, delta):
        shapes = cu.counter(delta, self.nMix) + self.priors.pi.a
        unnormalized = gamma(shape = shapes)
        return unnormalized / unnormalized.sum()

    def sample_r(self, zeta, delta):
        shapes = zeta[delta].sum(axis = 1)
        return gamma(shape = shapes)

    def sample_zeta(self, curr_zeta, r, delta, alpha, beta):
        Y = (self.data.S.T * r).T
        djs = [np.where(delta == j)[0] for j in range(self.nMix)]
        args = zip(
            curr_zeta,
            [Y[djs[j]] for j in range(self.nMix)],
            repeat(alpha),
            repeat(beta),
            )
        res = map(update_zeta_j_wrapper, args)
        # res = self.pool.map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def sample_delta(self, zeta, pi):
        args = zip(self.data.S, repeat(zeta), repeat(pi))
        res = map(sample_delta_i, args)
        # res = self.pool.map(sample_delta_i, args)
        return np.array(list(res)).reshape(-1)

    def initialize_sampler(self, ns):
        self.samples = FMIX_Samples(ns, self.nDat, self.nCol, self.nMix)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.zeta[0] = gamma(shape = 1., size = (self.nMix, self.nCol))
        self.samples.pi[0] = 1. / self.nMix
        self.samples.delta[0] = self.sample_delta(self.samples.zeta[0], self.samples.pi[0])
        self.samples.r[0] = self.sample_r(self.samples.zeta[0], self.samples.delta[0])
        self.curr_iter = 0
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
        self.samples.delta[self.curr_iter] = self.sample_delta(zeta, pi)
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
        deltas_df.to_sql('deltas', conn, index = False)
        pis_df.to_sql('pis', conn, index = False)
        rs_df.to_sql('rs', conn, index = False)
        alphas_df.to_sql('alphas', conn, index = False)
        betas_df.to_sql('betas', conn, index = False)
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
        self.data.S = to_simplex(self.data.V)
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.nMix = nMix
        self.priors = FMIX_Prior(prior_alpha, prior_beta, prior_pi)
        # self.pool = Pool(processes = 8)
        return

class FMIX_Result(object):
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
            postpred[n] = to_simplex(gamma(shape = zeta_new, size = (n_per_sample, self.nCol)))
        simplex = postpred.reshape(self.nSamp * n_per_sample, self.nCol)
        return to_hypercube(simplex)

    def generate_posterior_predictive_angular(self, n_per_sample = 1):
        hypercube = self.generate_posterior_predictive_hypercube(n_per_sample)
        return to_angular(hypercube)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        thetas = pd.DataFrame(
                self.generate_posterior_predictive_angular(n_per_sample),
                columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
                )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)

        deltas = pd.read_sql('select * from deltas;', conn).values
        pis    = pd.read_sql('select * from pis;', conn).values
        zetas  = pd.read_sql('select * from zetas;', conn).values
        alphas = pd.read_sql('select * from alphas;', conn).values
        betas  = pd.read_sql('select * from betas;', conn).values

        self.nSamp = deltas.shape[0]
        self.nDat = deltas.shape[1]
        self.nMix = pis.shape[1]
        self.nCol = zetas.shape[1]

        self.samples = FMIX_Samples(self.nSamp, self.nDat, self.nCol, self.nMix)
        self.samples.delta = deltas
        self.samples.pi = pis
        self.samples.zeta = zetas.reshape((self.nSamp, self.nMix, self.nCol))
        self.samples.alpha = alphas
        self.samples.beta = betas
        return

    def __init__(self, path):
        self.load_data(path)
        return

# Functions and definitions relating to DP Mixture

def log_density_delta_i(args):
    return gamma.logpdf(args[0], a = args[1]).sum()

class DPSimplex_Samples(object):
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
        self.alpha = np.empty((nSamp + 1, nCol))
        self.beta  = np.empty((nSamp + 1, nCol))
        self.zeta  = [None] * (nSamp + 1)
        return

DPSimplex_Prior = namedtuple('DPSimplex_Prior', 'alpha beta eta')

class DPSimplex_Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 10, m = 20):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = cu.counter(self.samples.delta[s], int(dmax + 1 + m))
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            new_zetas = gamma.rvs(
                    a = self.samples.alpha[s],
                    scale = 1. / self.samples.beta[s],
                    size = (m, self.nCol),
                    )
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            new_gammas.append(gamma.rvs(a = zeta))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 10):
        gnew = self.generate_posterior_predictive_gammas(n_per_sample) + epsilon
        return (gnew.T / gnew.max(axis = 1)).T

    def generate_posterior_predictive(self, n_per_sample = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample) + epsilon
        return to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 10):
        thetas = pd.DataFrame(
            self.generate_posterior_predictive(n_per_sample),
            columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
            )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)

        zetas = pd.read_sql('select * from zetas;', conn).values
        deltas = pd.read_sql('select * from deltas;', conn).values.astype(int)
        alphas = pd.read_sql('select * from alphas;', conn).values
        betas = pd.read_sql('select * from betas;', conn).values
        rs = pd.read_sql('select * from rs', conn).values
        etas = pd.read_sql('select * from etas', conn).values.reshape(-1)

        self.nDat = deltas.shape[1]
        self.nSamp = deltas.shape[0]
        self.nCol = alphas.shape[1]

        self.samples = DPSimplex_Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta = etas
        self.samples.r = rs
        self.samples.alpha = alphas
        self.samples.beta = betas
        self.samples.zeta = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        return

    def __init__(self, path):
        self.load_data(path)
        return

class DPSimplex_Chain(object):
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

    def clean_delta(self, delta, zeta, i):
        assert (delta.max() + 1 == zeta.shape[0])
        _delta = np.delete(delta, i)
        nj = cu.counter(_delta, _delta.max() + 2)
        fz = cu.first_zero(nj)
        _zeta = zeta[np.where(nj > 0)[0]]
        if (fz == delta[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _zeta

    def sample_zeta_new(self, m, alpha, beta):
        return gamma(shape = alpha, scale = 1/beta, size = (m, self.nCol))

    def sample_zeta(self, curr_zeta, delta, r, alpha, beta):
        Y = ((self.data.S).T * r).T
        djs = [np.where(delta == j)[0] for j in range(delta.max() + 1)]
        args = zip(
            curr_zeta,
            [Y[djs[j]] for j in range(delta.max() + 1)],
            repeat(alpha),
            repeat(beta),
            )
        res = map(update_zeta_j_wrapper, args)
        # res = self.pool.map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def sample_alpha(self, zeta, curr_alpha):
        args = zip(curr_alpha, zeta.T, repeat(self.priors.alpha), repeat(self.priors.beta))
        # res = self.pool.map(update_alpha_l_wrapper, args)
        res = map(update_alpha_l_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        args = zip(alpha, zeta.T, repeat(self.priors.beta))
        # res = self.pool.map(update_beta_l_wrapper, args)
        res = map(update_beta_l_wrapper, args)
        return np.array(list(res))

    def sample_r(self, delta, zeta):
        shapes = zeta[delta].sum(axis = 1)
        return gamma(shape = shapes)

    def sample_eta(self, curr_eta, delta):
        nClust = delta.max() + 1
        g = beta.rvs(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_delta_i(self, delta, zeta, r, alpha, beta, eta, i):
        _delta, _zeta = self.clean_delta(delta, zeta, i)
        _dmax = _delta.max()
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        ljs = njs + (njs == 0) * eta / self.m
        _zeta_new = self.sample_zeta_new(self.m, alpha, beta)
        zeta_stack = np.vstack((_zeta, _zeta_new))
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
            delta = np.insert(_delta, i, _dmax + 1)
        else:
            delta = np.insert(_delta, i, dnew)
            zeta = _zeta.copy()
        return delta, zeta

    def initialize_sampler(self, ns):
        self.samples = DPSimplex_Samples(ns, self.nDat, self.nCol)
        self.samples.delta[0] = np.array(list(range(self.nDat)), dtype = int)
        self.samples.r[0]     = 1.
        self.samples.eta[0]   = 5.
        self.samples.alpha[0] = 1.
        self.samples.beta[0]  = 1.
        self.samples.zeta[0]  = self.sample_zeta_new(
            self.nDat, self.samples.alpha[0], self.samples.beta[0],
            )
        self.curr_iter = 0
        return

    def iter_sample(self):
        """ Advance the sampler one iteration """
        # Parse the current values
        zeta  = self.curr_zeta
        delta = self.curr_delta
        eta   = self.curr_eta
        r     = self.curr_r
        alpha = self.curr_alpha
        beta  = self.curr_beta
        # Advance the iterator
        self.curr_iter += 1
        # Compute new cluster assignments based on current values of zeta (and new as needed)
        for i in range(self.nDat):
            delta, zeta = self.sample_delta_i(delta, zeta, r, alpha, beta, eta, i)
        # Update sampler with new values
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter] = self.sample_r(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter] = self.sample_zeta(
                zeta, self.curr_delta, self.curr_r, alpha, beta,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter] = self.sample_beta(self.curr_zeta, self.curr_alpha)
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

    def write_to_disk(self, path, nburn, thin = 1):
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # Gather output data
        zetas = np.vstack([
            np.vstack((np.ones(zeta.shape[0]) * i, zeta.T)).T
            for i, zeta in enumerate(self.samples.zeta[nburn::thin])
            ])
        alphas = self.samples.alpha[nburn::thin]
        betas  = self.samples.beta[nburn::thin]
        etas   = self.samples.eta[nburn::thin]
        deltas = self.samples.delta[nburn::thin]
        rs     = self.samples.r[nburn::thin]
        # Assemble Output DataFrames
        df_zeta  = pd.DataFrame(
                zetas, columns = ['iter'] + ['zeta_{}'.format(i) for i in range(self.nCol)],
                )
        df_alpha = pd.DataFrame(alphas, columns = ['alpha_{}'.format(i) for i in range(self.nCol)])
        df_beta  = pd.DataFrame(betas, columns = ['beta_{}'.format(i) for i in range(self.nCol)])
        df_eta   = pd.DataFrame({'eta' : etas})
        df_r     = pd.DataFrame(rs, columns = ['r_{}'.format(i) for i in range(self.nDat)])
        df_delta = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        # Write DataFrames to SQL Connection
        df_zeta.to_sql('zetas', conn, index = False)
        df_alpha.to_sql('alphas', conn, index = False)
        df_beta.to_sql('betas', conn, index = False)
        df_eta.to_sql('etas', conn, index = False)
        df_r.to_sql('rs', conn, index = False)
        df_delta.to_sql('deltas', conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        pass

    def __init__(
            self,
            data,
            prior_alpha = GammaPrior(1.,1.),
            prior_beta = GammaPrior(1.,1.),
            prior_eta = GammaPrior(2.,0.5),
            m = 20,
            ):
        self.m = m
        self.data = data
        self.data.S = to_simplex(self.data.V)
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = DPSimplex_Prior(prior_alpha, prior_beta, prior_eta)
        # self.pool = Pool(8)
        return

# EOF
