from projgamma import *
from scipy.stats import gamma, beta, gmean
from numpy.random import choice
from collections import namedtuple
from itertools import repeat
from math import ceil
import numpy as np
np.seterr(under = 'ignore', over = 'raise')
import multiprocessing as mp
import sqlite3 as sql
import pandas as pd
import data as dm
import cUtility as cu
import os

BNPPGPrior = namedtuple('BNPPGPrior', 'alpha beta eta')
Theta      = namedtuple('Theta','alpha beta')

# def log_density_gamma_i(args):
#     return logdprojgamma_pre_single(*args)
def update_alpha_wrapper(args):
    if args[0] == 0:
        return sample_alpha_1_mh(*args[1:4])
    elif args[0] > 0:
        return sample_alpha_k_mh(*args[1:])
    else:
        raise ValueError('Something other than col index was passed!')
def update_beta_wrapper(args):
    return sample_beta_fc(*args)

def log_density_gamma_i(args):
    # Y, alpha, beta
    return gamma(a = args[1], scale = 1/args[2]).logpdf(args[0]).sum()




class SamplesDPMPG(object):
    alpha = None # list, each entry is np.array; each row of array pertains to a cluster
    beta  = None # same as alpha
    delta = None # numpy array; int; indicates cluster membership
    eta   = None # Dispersion variable for DP algorithm
    r     = None # (latent) observation radius
    alpha_shape = None
    alpha_rate  = None
    beta_shape  = None
    beta_rate   = None

    def __init__(self, nSamp, nDat, nCol):
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        self.alpha = []
        self.beta  = []
        self.alpha_shape = np.empty((nSamp + 1, nCol))
        # self.alpha_rate  = np.empty((nSamp + 1, nCol))
        self.beta_shape  = np.empty((nSamp + 1, nCol - 1))
        # self.beta_rate   = np.empty((nSamp + 1, nCol - 1))
        return

class DPMPG(object):
    samples = None
    fixed_eta = False

    @property
    def curr_alpha_shape(self):
        return self.samples.alpha_shape[self.curr_iter].copy()

    # @property
    # def curr_alpha_rate(self):
    #     return self.samples.alpha_rate[self.curr_iter].copy()

    @property
    def curr_beta_shape(self):
        return self.samples.beta_shape[self.curr_iter].copy()

    # @property
    # def curr_beta_rate(self):
    #     return self.samples.beta_rate[self.curr_iter].copy()

    @property
    def curr_alphas(self):
        return self.samples.alpha[self.curr_iter].copy()

    @property
    def curr_betas(self):
        return self.samples.beta[self.curr_iter].copy()

    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter].copy()

    @property
    def curr_deltas(self):
        return self.samples.delta[self.curr_iter].copy()

    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter].copy()

    @property
    def curr_nClust(self):
        return self.curr_deltas.max() + 1

    def write_to_disk(self, path, nburn, thin = 1):
        """ Write output to disk """
        if os.path.exists(path):
            os.remove(path)
        conn = sql.connect(path)
        # Assemble output arrays
        ashapes = self.samples.alpha_shape[nburn::thin]
        bshapes = self.samples.beta_shape[nburn::thin]
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
        # Assemble output DataFrames
        df_ashapes = pd.DataFrame(
            ashapes,
            columns = ['alpha_shape_{}'.format(i) for i in range(self.nCol)]
            )
        df_bshapes = pd.DataFrame(
            bshapes,
            columns = ['beta_shape_{}'.format(i) for i in range(1, self.nCol)]
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
        df_rs = pd.DataFrame(
            rs,
            columns = ['r_{}'.format(i) for i in range(self.nDat)]
            )
        df_eta = pd.DataFrame({'eta' : eta})
        # Write Dataframes to SQL Connection
        df_ashapes.to_sql('alpha_shape', conn, index = False)
        df_bshapes.to_sql('beta_shape', conn, index = False)
        df_alphas.to_sql('alphas', conn, index = False)
        df_betas.to_sql('betas', conn, index = False)
        df_deltas.to_sql('deltas', conn, index = False)
        df_rs.to_sql('rs', conn, index = False)
        df_eta.to_sql('eta', conn, index = False)
        # Commit and Close
        conn.commit()
        conn.close()
        return

    def clean_delta(self, deltas, alphas, betas, i):
        assert (deltas.max() + 1 == alphas.shape[0])
        _delta = np.delete(deltas, i)
        nj     = cu.counter(_delta, _delta.max() + 2)
        fz     = cu.first_zero(nj)
        _alpha = alphas[np.where(nj > 0)[0]]
        _beta  = betas[np.where(nj > 0)[0]]
        if (fz == deltas[i]) and (fz <= _delta.max()):
            _delta[_delta > fz] = _delta[_delta > fz] - 1
        return _delta, _alpha, _beta

    def sample_shape_alpha(self, curr_shapes, alphas):
        """ Sample the shape parameter for the distribution for alpha """
        args = zip(repeat(0), curr_shapes, alphas.T,
                    repeat(self.priors.alpha), repeat(self.priors.beta))
        prop_shape = np.array(list(self.pool.map(update_alpha_wrapper, args))).reshape(-1)
        return prop_shape

    def sample_rate_alpha(self, shapes, alphas):
        """ Sample the rate parameter for the distribution for alpha | shape """
        args = zip(shapes, alphas.T, repeat(self.priors.beta))
        prop_rate = np.array(list(self.pool.map(update_beta_wrapper, args))).reshape(-1)
        return prop_rate

    def sample_shape_beta(self, curr_shapes, betas):
        """ Sample the shape parameter for the distribution for beta """
        args = zip(repeat(0), curr_shapes, betas.T[1:],
                    repeat(self.priors.alpha), repeat(self.priors.beta))
        prop_shape = np.array(list(self.pool.map(update_alpha_wrapper, args))).reshape(-1)
        return prop_shape

    def sample_rate_beta(self, shapes, betas):
        """ Sample the rate parameter for the distribution for beta | shape """
        args = zip(shapes, betas.T[1:], repeat(self.priors.beta))
        prop_rate = np.array(list(self.pool.map(update_beta_wrapper, args))).reshape(-1)
        return prop_rate

    def sample_delta_i(self, deltas, rs, alphas, betas, ashape, bshape, eta, i): # bshape, brate, eta, i):
        # Clean the deltas, alphas, and betas.  calculate the new max delta
        _delta, _alpha, _beta = self.clean_delta(deltas, alphas, betas, i)
        _dmax = _delta.max()
        # Compute the prior probabilities for the collapsed sampler.
        njs = cu.counter(_delta, _dmax + 1 + self.m)
        # njs = np.array([(_delta == j).sum() for j in range(_dmax + 1 + self.m)])
        ljs = njs + (njs == 0) * eta / self.m
        # Generate potential new clusters
        # alpha_new, beta_new = self.sample_alpha_beta_new(self.m, ashape, arate, bshape, brate)
        alpha_new, beta_new = self.sample_alpha_beta_new(self.m, ashape, bshape)
        alpha_stack = np.vstack((_alpha, alpha_new))
        beta_stack  = np.vstack((_beta, beta_new))
        assert (alpha_stack.shape[0] == ljs.shape[0])
        # Calculate log-posteriors under each cluster
        # args = zip(
        #     repeat(self.data.lcoss[i]),
        #     repeat(self.data.lsins[i]),
        #     repeat(self.data.Yl[i]),
        #     alpha_stack,
        #     beta_stack,
        #     )
        args = zip(repeat(rs[i] * self.data.Yl[i]), alpha_stack, beta_stack)
        res = self.pool.map(log_density_gamma_i, args, chunksize = ceil(ljs.shape[0]/8))
        lps = np.array(list(res))
        # lps = np.array(list(map(log_density_gamma_i, args)))
        lps[np.where(np.isnan(lps))] = - np.inf
        unnormalized = np.exp(lps) * ljs
        normalized = unnormalized / unnormalized.sum()
        dnew = choice(range(_dmax + self.m + 1), 1, p = normalized)
        if dnew > _dmax:
            alpha = np.vstack((_alpha, alpha_stack[dnew]))
            beta  = np.vstack((_beta,  beta_stack[dnew]))
            delta = np.insert(_delta, i, _dmax + 1)
        else:
            delta = np.insert(_delta, i, dnew)
            alpha = _alpha.copy()
            beta  = _beta.copy()
        return delta, alpha, beta

    def sample_alpha_beta_new(self, size, ashape, bshape): # arate, bshape, brate):
        # alphas = gamma.rvs(a = ashape, scale = 1/arate, size = (size, self.nCol))
        alphas = gamma.rvs(a = ashape, size = (size, self.nCol))
        betas  = np.hstack((
            np.ones((size, 1)),
            #gamma.rvs(a = bshape, scale = 1/brate, size = (size, self.nCol - 1)),
            gamma.rvs(a = bshape, size = (size, self.nCol - 1)),
            ))
        return alphas, betas

    def update_alpha_j(self, curr_alpha_j, Yj, ashapes, bshapes):
        priors_alpha = [GammaPrior(x,1) for x in ashapes]
        priors_beta = [None] + [GammaPrior(x,1) for x in bshapes]
        args = zip(list(range(self.nCol)), curr_alpha_j, Yj.T, priors_alpha, priors_beta)
        prop_alphas = np.array(list(self.pool.map(update_alpha_wrapper, args))).reshape(-1)
        return prop_alphas

    def update_beta_j(self, alpha_j, Yj, bshapes):
        priors_beta = [GammaPrior(x,1) for x in bshapes]
        args = zip(alpha_j[1:], Yj.T[1:], priors_beta)
        prop_betas = np.array([1.] + list(self.pool.map(update_beta_wrapper, args))).reshape(-1)
        return prop_betas

    def update_alpha_beta(self, curr_alphas, deltas, Y, ashapes, bshapes):#  arates, bshapes, brates):
        nClust = deltas.max() + 1
        djs  = [(deltas == j) for j in range(nClust)]
        Yjks = [Y[djs[j], k] for j in range(nClust) for k in range(self.nCol)]
        idxs = list(range(self.nCol)) * nClust
        # priors_alpha = [GammaPrior(x,y) for x,y in zip(ashapes, arates)] * nClust
        priors_alpha = [GammaPrior(x,1) for x in ashapes] * nClust
        # priors_beta_primitive = [GammaPrior(x,y) for x,y in zip(bshapes, brates)]
        priors_beta_primitive = [GammaPrior(x,1) for x in bshapes]
        priors_beta  = ([None] + priors_beta_primitive) * nClust
        alpha_args = zip(
            idxs,
            curr_alphas.reshape(-1),
            Yjks,
            priors_alpha,
            priors_beta,
            )
        prop_alphas = np.array(list(self.pool.map(update_alpha_wrapper, alpha_args))).reshape(curr_alphas.shape)
        # prop_alphas = np.array(list(map(update_alpha_wrapper, alpha_args))).reshape(curr_alphas.shape)
        Yjks = [Y[djs[j], k] for j in range(nClust) for k in range(1, self.nCol)]
        beta_args = zip(prop_alphas[:,1:].reshape(-1), Yjks, priors_beta_primitive * nClust)
        prop_betas = np.hstack((
            np.ones((nClust, 1)),
            np.array(list(self.pool.map(update_beta_wrapper, beta_args))).reshape(nClust, self.nCol - 1),
            # np.array(list(map(update_beta_wrapper, beta_args))).reshape(nClust, self.nCol - 1),
            ))
        return prop_alphas, prop_betas

    def sample_r(self, alphas, betas, deltas):
        alpha = alphas[deltas]
        beta  = betas[deltas]
        As = alpha.sum(axis = 1)
        Bs = (self.data.Yl * beta).sum(axis = 1)
        return gamma.rvs(As, scale = 1/Bs)

    def sample_eta(self, curr_eta, nClust):
        g  = beta.rvs(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + nClust
        bb = self.priors.eta.b - log(g)
        eps = (aa - 1) / (self.nDat * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma.rvs(aaa, bb)

    def set_prior_samplers(self):
        self.alpha_prior = gamma(self.priors.alpha.a, scale = 1. / self.priors.alpha.b)
        self.beta_prior  = gamma(self.priors.beta.a,  scale = 1. / self.priors.beta.b)
        return

    def initialize_sampler(self, ns):
        self.samples = SamplesDPMPG(ns, self.nDat, self.nCol)
        # Initial conditions
        self.samples.delta[0] = np.array(list(range(self.nDat)))
        self.samples.r[0]     = 1.
        self.samples.eta[0]   = 5.
        if self.fixed_eta:
            self.samples.eta[0] = self.fixed_eta
        self.samples.alpha_shape[0] = 1.
        # elf.samples.alpha_rate[0] = 1.
        self.samples.beta_shape[0] = 1.
        # self.samples.beta_rate[0] = 1.
        alpha_start, beta_start = self.sample_alpha_beta_new(
                self.nDat,
                self.samples.alpha_shape[0],
                # self.samples.alpha_rate[0],
                self.samples.beta_shape[0],
                # self.samples.beta_rate[0],
                )
        self.samples.alpha.append(alpha_start)
        self.samples.beta.append(beta_start)
        self.curr_iter = 0
        return

    def iter_sample(self):
        # Fix the current estimates
        alphas, betas = self.curr_alphas, self.curr_betas
        deltas, eta   = self.curr_deltas, self.curr_eta
        alpha_shapes  = self.curr_alpha_shape
        rs            = self.curr_r
        # alpha_rates  = self.curr_alpha_rate
        beta_shapes  = self.curr_beta_shape
        # beta_rate    = self.curr_beta_rate

        Y = (self.data.Yl.T * rs).T

        # advance the iterator
        self.curr_iter += 1

        # Generate new Hierarchical shapes and rates for the Gamma RV's.
        self.samples.alpha_shape[self.curr_iter] = \
                self.sample_shape_alpha(alpha_shapes, alphas)
        # self.samples.alpha_rate[self.curr_iter]  = \
        #         self.sample_rate_alpha(self.curr_alpha_shape, alphas)
        self.samples.beta_shape[self.curr_iter]  = \
                self.sample_shape_beta(beta_shapes, betas)
        # self.samples.beta_rate[self.curr_iter]   = \
        #         self.sample_rate_beta(self.curr_beta_shape, betas)

        # Sample cluster assignments
        for i in range(self.nDat):
            deltas, alphas, betas = self.sample_delta_i(
                    deltas, rs, alphas, betas,
                    self.curr_alpha_shape, self.curr_beta_shape,
                    # self.curr_alpha_shape, self.curr_alpha_rate,
                    # self.curr_beta_shape, self.curr_beta_rate,
                    eta, i,
                    )
            # Resampling alpha_j, beta_j as cluster_j changes
            di = deltas[i]
            dix = np.where(deltas == di)[0]
            alphas[di] = self.update_alpha_j(alphas[di], Y[dix], alpha_shapes, beta_shapes)
            betas[di]  = self.update_beta_j(alphas[di], Y[dix], beta_shapes)

        self.samples.delta[self.curr_iter] = deltas

        # Compute new latent radii
        self.samples.r[self.curr_iter] = self.sample_r(alphas, betas, deltas)

        # Sample new estimates based on new cluster assignments
        Y = (self.data.Yl.T * self.curr_r).T
        alphas, betas = self.update_alpha_beta(
                alphas, deltas, Y,
                self.curr_alpha_shape, self.curr_beta_shape,
                # self.curr_alpha_shape, self.curr_alpha_rate,
                # self.curr_beta_shape, self.curr_beta_rate,
                )
        self.samples.alpha.append(alphas)
        self.samples.beta.append(betas)
        if self.fixed_eta:
            self.samples.eta[self.curr_iter] = self.fixed_eta
        else:
            self.samples.eta[self.curr_iter] = self.sample_eta(eta, alphas.shape[0])
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        print_string = '\rSampling {:.1%} Completed, {} Clusters     '
        print(print_string.format(self.curr_iter / ns, self.nDat), end = '')
        while (self.curr_iter < ns):
            if (self.curr_iter % 10) == 0:
                print(print_string.format(self.curr_iter / ns, self.curr_nClust), end = '')
            self.iter_sample()
        print('\rSampling 100% Completed                    ')
        return

    def __init__(
            self,
            data,
            prior_alpha = GammaPrior(1.,0.5),
            prior_beta = GammaPrior(1.,0.5),
            prior_eta = GammaPrior(2.,.5),
            m = 30,
            fixed_eta = False,
            ):
        self.m = m
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        if fixed_eta:
            self.fixed_eta = fixed_eta
        self.priors = BNPPGPrior(prior_alpha, prior_beta, prior_eta)
        self.set_prior_samplers()
        self.pool = mp.Pool(8)
        return

class ResultDPMPG(object):
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

    def generate_posterior_predictive_gammas(self, n_per_sample = 10, m = 20):
        new_gammas = []
        for i in range(self.nSamp):
            dmax   = self.samples.delta[i].max()
            njs    = cu.counter(self.samples.delta[i], dmax + 1 + m)
            ljs    = njs + (njs == 0) * self.samples.eta[i] / m
            new_alphas = gamma.rvs(a = self.samples.alpha_shape[i], size = (m, self.nCol))
            new_betas  = np.hstack((
                np.ones((m, 1)),
                gamma.rvs(a = self.samples.beta_shape[i], size = (m, self.nCol - 1)),
                ))
            prob   = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            alpha  = np.vstack((self.samples.alpha[i], new_alphas))[deltas]
            beta   = np.vstack((self.samples.beta[i], new_betas))[deltas]
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
        conn    = sql.connect(path)
        ashapes = pd.read_sql('select * from alpha_shape;', conn).values
        bshapes = pd.read_sql('select * from beta_shape;', conn).values
        alphas  = pd.read_sql('select * from alphas;', conn).values
        betas   = pd.read_sql('select * from betas;', conn).values
        deltas  = pd.read_sql('select * from deltas;', conn).values.astype(int)
        rs      = pd.read_sql('select * from rs', conn).values
        eta     = pd.read_sql('select * from eta;', conn).values.T[0]

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = ashapes.shape[1]

        self.samples = SamplesDPMPG(self.nSamp, self.nDat, self.nCol)
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
        self.samples.alpha_shape = ashapes
        self.samples.beta_shape  = bshapes
        return

    def __init__(self, path):
        self.load_data(path)
        return

# EOF
