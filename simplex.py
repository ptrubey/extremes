from scipy.stats import gamma, beta, gmean, norm as normal, invwishart, uniform, dirichlet
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import loggamma
from numpy.random import choice
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
from numpy import log, exp
from scipy.special import loggamma
from data import *
from multiprocessing import Pool
import os
import sqlite3 as sql

import cUtility as cu
from projgamma import sample_alpha_1_mh as sample_eta_jl

epsilon = 1e-5

GammaPrior = namedtuple('GammaPrior','a b')
DirichletPrior = namedtuple('DirichletPrior','a')

def dmvnormal(x, mu, cov_chol, cov_inv, log = True):
    ld = (
        - 0.5 * 2 * np.log(np.diag(cov_chol[0])).sum()
        - 0.5 * ((x - mu).T @ cov_inv @ (x - mu)).sum()
        )
    if log:
        return ld
    return exp(lp)

def ddirichlet_single(X, alpha, log = True):
    # ld = (
    #    + loggamma(alpha.sum())
    #    - loggamma(alpha).sum()
    #    + ((alpha - 1) * log(X)).sum() #something's wrong here...
    #    )
    #if log:
    #    return ld
    #return exp(ld)
    if log:
        return dirichlet(alpha).logpdf(X)
    else:
        return dirichlet(alpha).pdf(X)

def ddirichlet_multi(X, alpha, log = True):
    ld = (
        + loggamma(alpha.sum())
        - loggamma(alpha).sum()
        + ((alpha - 1) * log(X)).sum(axis = 1)
    )
    if log:
        return ld
    return exp(ld)

def log_posterior_log_eta_jl(log_eta, Xjl, l, prior):
    nj = Xjl.shape[0]
    eta = exp(log_eta)
    lp = (
        + nj * loggamma(eta.sum())
        - nj * loggamma(eta[l])
        + eta[l] * log(Xjl).sum()
        + prior.a * log_eta[l] # prior.a - 1 + 1 (jacobian of log xform)
        - prior.b * eta[l]
        )
    return lp

def sample_delta_i(args):
    Xi = args[0]
    eta = args[1]
    pi = args[2]
    argset = zip(repeat(Xi), eta)
    lps = np.array(list(map(lambda arg: ddirichlet_single(*arg), argset)))
    lps[np.isnan(lps)] = - np.inf
    lps = lps - lps.max()
    unnormalized = np.exp(lps) * pi
    normalized = unnormalized / unnormalized.sum()
    return choice(pi.shape[0], 1, p = normalized)

def sample_eta_j(args):
    curr_eta = args[0]
    Xj = args[1]
    prior = args[2]
    prop = np.empty(curr_eta.shape)
    for l in range(curr_eta.shape[0]):
        prop[l] = sample_eta_jl(curr_eta[l], Xj.T[l], prior)
    return prop

def to_simplex(data):
    return ((data + epsilon).T / (data + epsilon).sum(axis = 1)).T

def to_hypercube(data):
    return (data.T / data.max(axis = 1)).T

class FMIX_Samples(object):
    eta = None
    delta = None
    pi = None

    def __init__(self, nSamp, nDat, nCol, nMix):
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.eta = np.empty((nSamp + 1, nMix, nCol))
        self.pi = np.empty((nSamp + 1, nMix))
        self.r = np.empty((nSamp + 1, nDat))
        return

class FMIX_Prior(object):
    eta = None
    pi = None

    def __init__(self, eta_prior, pi_prior):
        self.eta = eta_prior
        self.pi = pi_prior
        return

class FMIX_Chain(object):
    samples = None
    priors = None

    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()

    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter].copy()

    @property
    def curr_pi(self):
        return self.samples.pi[self.curr_iter].copy()

    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()

    def sample_pi(self, delta):
        shapes = cu.counter(delta, self.nMix) + self.priors.pi.a
        unnormalized = gamma.rvs(a = shapes)
        return unnormalized / unnormalized.sum()

    def sample_r(self, eta, delta):
        eta_sum = eta[delta].sum(axis = 1)
        return gamma.rvs(a = eta_sum, scale = 1)

    def sample_eta(self, curr_eta, r, delta):
        Y = (self.data.S.T * r).T
        Yj = [Y[np.where(delta == j)[0]] for j in range(self.nMix)]
        args = zip(curr_eta, Yj, repeat(self.priors.eta))
        prop_eta = np.array(list(self.pool.map(sample_eta_j, args))).reshape(self.nMix, self.nCol)
        return prop_eta

    def sample_delta(self, eta, pi):
        args = zip(self.data.S, repeat(eta), repeat(pi))
        return np.array(list(self.pool.map(sample_delta_i, args))).reshape(-1)

    def initialize_sampler(self, ns):
        self.samples = FMIX_Samples(ns, self.nDat, self.nCol, self.nMix)
        self.samples.eta[0] = gamma.rvs(
                a = self.priors.eta.a, scale = 1. / self.priors.eta.b,
                size = (self.nMix, self.nCol),
                )
        self.samples.pi[0] = 1. / self.nMix
        self.samples.delta[0] = self.sample_delta(self.samples.eta[0], self.samples.pi[0])
        self.samples.r[0] = self.sample_r(self.samples.eta[0], self.samples.delta[0])
        self.curr_iter = 0
        return

    def iter_sample(self):
        # Pull current values
        eta   = self.curr_eta
        pi    = self.curr_pi
        delta = self.curr_delta
        r     = self.curr_r
        # Advance the iterator
        self.curr_iter += 1
        # Compute new values
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, r, delta)
        self.samples.delta[self.curr_iter] = self.sample_delta(self.curr_eta, pi)
        self.samples.pi[self.curr_iter] = self.sample_pi(self.curr_delta)
        self.samples.r[self.curr_iter] = self.sample_r(self.curr_eta, self.curr_delta)
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
        nTail = self.samples.eta.shape[0] - nBurn - 1
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # declare the resulting sample set
        etas   = self.samples.eta[-nTail :: nThin]
        deltas = self.samples.delta[-nTail :: nThin]
        pis    = self.samples.pi[-nTail :: nThin]
        rs     = self.samples.r[-nTail :: nThin]
        # get the output sample size
        nSamp = deltas.shape[0]
        # set up resulting data frames
        etas_df = pd.DataFrame(etas.reshape(nSamp * self.nMix, self.nCol),
                                columns = ['eta_{}'.format(i) for i in range(self.nCol)])
        deltas_df = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        pis_df = pd.DataFrame(pis, columns = ['pi_{}'.format(i) for i in range(self.nMix)])
        rs_df = pd.DataFrame(rs, columns = ['r_{}'.format(i) for i in range(self.nDat)])
        # write to disk
        etas_df.to_sql('etas', conn, index = False)
        deltas_df.to_sql('deltas', conn, index = False)
        pis_df.to_sql('pis', conn, index = False)
        rs_df.to_sql('rs', conn, index = False)
        return

    def __init__(self, data, nMix, prior_eta = GammaPrior(1.,1.), prior_pi = DirichletPrior(1.)):
        self.data = data
        self.data.S = to_simplex(self.data.V)
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.nMix = nMix
        self.priors = FMIX_Prior(prior_eta, prior_pi)
        self.pool = Pool(processes = 8)
        return

class FMIX_Result(object):
    samples = None
    nSamp = None
    nDat = None
    nCol = None
    nMix = None

    def generate_posterior_predictive_hypercube(self, n_per_sample = 5):
        postpred = np.empty((self.nSamp, n_per_sample, self.nCol))
        for n in range(self.nSamp):
            delta_new = choice(self.nMix, n_per_sample, p = self.samples.pi[n])
            eta_new = self.samples.eta[n,delta_new]
            postpred[n] = np.apply_along_axis(
                    lambda a: dirichlet.rvs(alpha = a), 1, eta_new,
                    ).reshape(n_per_sample, self.nCol)
        simplex = postpred.reshape(self.nSamp * n_per_sample, self.nCol)
        return to_hypercube(simplex)

    def generate_posterior_predictive_angular(self, n_per_sample = 5):
        hypercube = self.generate_posterior_predictive_hypercube(n_per_sample)
        return to_angular(hypercube)

    def write_posterior_predictive(self, path, n_per_sample = 5):
        thetas = pd.DataFrame(
                self.generate_posterior_predictive_angular(n_per_sample),
                columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
                )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        conn = sql.connect(path)

        deltas = pd.read_sql('select * from deltas;', conn).values
        pis = pd.read_sql('select * from pis;', conn).values
        etas = pd.read_sql('select * from etas;', conn).values

        self.nSamp = deltas.shape[0]
        self.nDat = deltas.shape[1]
        self.nMix = pis.shape[1]
        self.nCol = etas.shape[1]

        self.samples = FMIX_Samples(5, self.nDat, self.nCol, self.nMix)
        self.samples.delta = deltas
        self.samples.pi = pis
        self.samples.eta = etas.reshape((self.nSamp, self.nMix, self.nCol))
        return

    def __init__(self, path):
        self.load_data(path)
        return
