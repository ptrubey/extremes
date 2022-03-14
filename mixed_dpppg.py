from numpy.random import choice, gamma, beta, uniform
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import sqlite3 as sql
import pickle
from math import ceil, log
from scipy.special import gammaln

import cUtility as cu
from cProjgamma import sample_alpha_1_mh_summary, sample_alpha_k_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube, euclidean_to_simplex, MixedData
from projgamma import GammaPrior

# from multiprocessing import Pool
# from energy import limit_cpu

def dprodgamma_log_my_mt(aY, aAlpha, aBeta):
    """
    Product of Gammas log-density for multiple Y, multiple theta (not paired)
    ----
    aY     : array of Y     (n x d)
    aAlpha : array of alpha (J x d)
    aBeta  : array of beta  (J x d)
    ----
    return : array of ld    (n x J)
    """
    out = np.zeros((aY.shape[0], aAlpha.shape[0]))
    out += np.einsum('jd,jd->j', aAlpha, np.log(aBeta)).reshape(1,-1) # beta^alpha
    out -= np.einsum('jd->j', gammaln(aAlpha)).reshape(1,-1)          # gamma(alpha)
    out += np.einsum('jd,nd->nj', aAlpha - 1, np.log(aY))             # y^(alpha - 1)
    out -= np.einsum('jd,nd->nj', aBeta, aY)                          # e^(-beta y)
    return out

def update_zeta_j_wrapper(args):
    # parse arguments
    curr_zeta_j, n_j, Y_js, lY_js, alpha, beta, xi, tau, sigma_unity = args
    prop_zeta_j = np.empty(curr_zeta_j.shape)
    # prepare placeholders for xi, tau (with same indexing as zeta)
    xxi = np.ones(curr_zeta_j.shape)
    xta = np.ones(curr_zeta_j.shape)
    xxi[~sigma_unity] = xi
    xta[~sigma_unity] = tau
    # iterate through zeta sampling
    for i in range(curr_zeta_j.shape[0]):
        if sigma_unity[i]:
            prop_zeta_j[i] = sample_alpha_1_mh_summary(
                curr_zeta_j[i], n_j, Y_js[i], lY_js[i], alpha[i], beta[i],
                )
        else:
            prop_zeta_j[i] = sample_alpha_k_mh_summary(
                curr_zeta_j[i], n_j, Y_js[i], lY_js[i], 
                alpha[i], beta[i], xxi[i], xta[i],
                )
    return prop_zeta_j

def update_sigma_j_wrapper(args):
    zeta_j, n_j, Y_js, xi, tau, sigma_unity = args
    prop_sigma_j = np.ones(zeta_j.shape)
    As = n_j * zeta_j[~sigma_unity] + xi
    Bs = Y_js[~sigma_unity] + tau
    prop_sigma_j[~sigma_unity] = gamma(shape = As, scale = 1 / Bs)
    return prop_sigma_j

def sample_gamma_shape_wrapper(args):
    return sample_alpha_k_mh_summary(*args)
    # return sample_alpha_1_mh_summary(*args)

Prior = namedtuple('Prior', 'eta alpha beta xi tau')

class Samples(object):
    pi    = None
    zeta  = None
    sigma = None
    alpha = None
    beta  = None
    xi    = None
    tau   = None
    delta = None
    r     = None
    eta   = None

    def __init__(self, nSamp, nDat, nCol, nCat, nCats):
        """
        nCol: number of 
        nCat: number of categorical columns
        nCats: number of categorical variables        
        """
        self.zeta  = [None] * (nSamp + 1)
        self.sigma = [None] * (nSamp + 1)
        self.rho    = np.empty((nSamp + 1, nDat, nCat))
        self.alpha = np.empty((nSamp + 1, nCol + nCat))
        self.beta  = np.empty((nSamp + 1, nCol + nCat))
        self.xi    = np.empty((nSamp + 1, nCol + nCat - 1 - nCats))
        self.tau   = np.empty((nSamp + 1, nCol + nCat - 1 - nCats))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty(nSamp + 1)
        return

class Chain(object):
    @property
    def curr_rho(self):
        return self.samples.rho[self.curr_iter]
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_sigma(self):
        return self.samples.sigma[self.curr_iter]
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter]
    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter]
    @property
    def curr_xi(self):
        return self.samples.xi[self.curr_iter]
    @property
    def curr_tau(self):
        return self.samples.tau[self.curr_iter]
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter]
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter]
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter]

    sigma_unity = None

    def sample_delta_i(self, curr_cluster_state, cand_cluster_state, eta, 
                                        log_likelihood_i, delta_i, p, scratch):
        scratch[:] = 0
        curr_cluster_state[delta_i] -= 1
        scratch += curr_cluster_state
        scratch += cand_cluster_state * (eta / (cand_cluster_state.sum() + 1e-9))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            np.log(scratch, out = scratch)
        # scratch += np.log(curr_cluster_state + cand_cluster_state * eta / cand_cluster_state.sum())
        scratch += log_likelihood_i
        np.nan_to_num(scratch, False, -np.inf)
        scratch -= scratch.max()
        with np.errstate(under = 'ignore'):
            np.exp(scratch, out = scratch)
        np.cumsum(scratch, out = scratch)
        delta_i = np.searchsorted(scratch, p * scratch[-1])
        curr_cluster_state[delta_i] += 1
        cand_cluster_state[delta_i] = False
        return delta_i
    
    def clean_delta_zeta_sigma(self, delta, zeta, sigma):
        """
        delta : cluster indicator vector (n)
        zeta  : cluster parameter matrix (J* x d)
        sigma : cluster parameter matrix (J* x d)
        """
        # which clusters are populated
        # keep = np.bincounts(delta) > 0 
        # reindex those clusters
        keep, delta[:] = np.unique(delta, return_inverse = True)
        # return new indices, cluster parameters associated with populated clusters
        return delta, zeta[keep], sigma[keep]

    def sample_zeta_new(self, alpha, beta, m):
        return gamma(shape = alpha, scale = 1 / beta, size = (m, self.nCol + self.nCat))

    def sample_sigma_new(self, xi, tau, m):
        prop_sigma = np.empty((m, self.nCol + self.nCat))
        prop_sigma[:, np.where(self.sigma_unity)[0]] = 1
        prop_sigma[:, np.where(~self.sigma_unity)[0]] = gamma(
            shape = xi, scale = 1 / tau, size = (m, self.nSigma)
            )
        return prop_sigma

    def sample_alpha(self, zeta, curr_alpha):
        n = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        lzs = np.log(zeta).sum(axis = 0)
        args = zip(
            curr_alpha, repeat(n), zs, lzs,
            repeat(self.priors.alpha.a), repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a), repeat(self.priors.beta.b),
            )
        res = map(sample_gamma_shape_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        n  = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        return gamma(shape = As, scale = 1 / Bs)
        # return np.ones(As.shape)

    def sample_xi(self, sigma, curr_xi):
        n    = sigma.shape[0]
        ss   = sigma.sum(axis = 0)
        lss  = np.log(sigma).sum(axis = 0)
        args = zip(
            curr_xi, repeat(n), ss, lss,
            repeat(self.priors.xi.a), repeat(self.priors.xi.b),
            repeat(self.priors.tau.a), repeat(self.priors.tau.b),
            )
        res = map(sample_gamma_shape_wrapper, args)
        return np.array(list(res))

    def sample_tau(self, sigma, xi):
        n  = sigma.shape[0]
        ss = sigma[:,~self.sigma_unity].sum(axis = 0)
        As = n * xi + self.priors.tau.a
        Bs = ss + self.priors.tau.b
        return gamma(shape = As, scale = 1 / Bs)

    def sample_r(self, delta, zeta, sigma):
        # As = zeta[delta][:, :self.nCol].sum(axis = 1)
        # Bs = (self.data.Yp * sigma[delta][:, :self.nCol]).sum(axis = 1)
        As = np.einsum('il->i', zeta[:,:self.nCol][delta])
        Bs = np.einsum('il,il->i', self.data.Yp, sigma[:, :self.nCol][delta])
        return gamma(shape = As, scale = 1 / Bs)

    def sample_rho(self, delta, zeta, sigma):
        """ Sampling the PG_1 gammas for categorical variables

        Args:
            delta ([type]): [description]
            zeta ([type]): [description]
            sigma ([type]): [description]
        """
        As = zeta[:, self.nCol:][delta] + self.data.W
        Bs = sigma[:, self.nCol:][delta]
        rho = gamma(shape = As, scale = 1 / Bs)
        rho[rho < 1e-9] = 1e-9
        return rho

    def sample_eta(self, curr_eta, delta):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max() + 1
        bb = self.priors.beta.b - log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_zeta(self, curr_zeta, r, rho, delta, alpha, beta, xi, tau):
        dmat = delta[:,None] == np.arange(delta.max() + 1) # n x J
        Y    = np.hstack((r[:, None] * self.data.Yp, rho)) # n x D
        n    = dmat.sum(axis = 0)
        Ysv  = (Y.T @ dmat).T          # np.einsum('nd,nj->jd', Y, dmat) 
        lYsv = (np.log(Y).T @ dmat).T # np.einsum('nd,nj->jd', np.log(Y), dmat)
        args = zip(
            curr_zeta, n, Ysv, lYsv, 
            repeat(alpha), repeat(beta), 
            repeat(xi), repeat(tau), repeat(self.sigma_unity),
            )
        # curr_zeta_j, n_j, Y_js, lY_js, alpha, beta, xi, tau, sigma_unity = args
        res = map(update_zeta_j_wrapper, args)
        return np.array(list(res))

    def sample_sigma(self, zeta, r, rho, delta, xi, tau):
        dmat = delta[:, None] == np.arange(delta.max() + 1)
        Y = np.hstack((r[:, None] * self.data.Yp, rho))
        n = dmat.sum(axis = 0)
        # Y = r.reshape(-1, 1) * self.data.Yp
        Ysv = (Y.T @ dmat).T # (J x d)
        args = zip(
            zeta, n, Ysv,
            repeat(xi), repeat(tau), repeat(self.sigma_unity),
            )
        res = np.array(list(map(update_sigma_j_wrapper, args)))
        # zeta_j, n_j, Y_js, xi, tau, sigma_unity = args
        # res = self.pool.map(update_sigma_j_wrapper, args)
        res[res < 1e-12] = 1e-12
        return res

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol, self.nCat, self.nCats)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.xi[0] = 1.
        self.samples.tau[0] = 1.
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 30, self.nCol + self.nCat))
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 30, self.nCol + self.nCat))
        self.samples.sigma[0][:, self.sigma_unity] = 1.
        self.samples.eta[0] = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 30, size = self.nDat)
        self.samples.delta[0][-1] = np.arange(self.max_clust_count - 30)[-1]
        self.samples.r[0] = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0], self.samples.sigma[0],
                )
        self.samples.rho[0] = (self.CatMat / self.data.Cats).sum(axis = 1)
        self.curr_iter = 0
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy();  m = self.max_clust_count - (delta.max() + 1)
        curr_cluster_state = np.bincount(delta, minlength = self.max_clust_count)
        cand_cluster_state = np.hstack((np.zeros(delta.max() + 1, dtype = bool), np.ones(m, dtype = bool)))
        alpha = self.curr_alpha
        beta  = self.curr_beta
        xi    = self.curr_xi
        tau   = self.curr_tau
        zeta  = np.vstack((self.curr_zeta, self.sample_zeta_new(alpha, beta, m)))
        sigma = np.vstack((self.curr_sigma, self.sample_sigma_new(xi, tau, m)))
        eta   = self.curr_eta
        r     = self.curr_r
        rho   = self.curr_rho

        self.curr_iter += 1
        # projecting rho onto unit simplex (for each projection)
        #rho_normalized = rho * np.einsum('dc,nc->nd', self.CatMat, 1 / (rho @ self.CatMat))
        # Pre-Compute Log-likelihood under each (extant and candidate) cluster
        # log_likelihood = (
        #     + dprodgamma_log_my_mt(r[:,None] * self.data.Yp, zeta[:, :self.nCol], sigma[:, :self.nCol])
        #     + dprodgamma_log_my_mt(rho, zeta[:, self.nCol:], sigma[:, self.nCol:])
        #     )
        Y = np.hstack((r[:,None] * self.data.Yp, rho))
        log_likelihood = dprodgamma_log_my_mt(
            np.hstack((r[:,None] * self.data.Yp, rho)), zeta, sigma,
            )
        # pre-generate uniforms to inverse-cdf sample cluster indices
        unifs   = uniform(size = self.nDat)
        # provide a cluster index probability placeholder, so it's not being re-allocated for every sample
        scratch = np.empty(self.max_clust_count)
        for i in range(self.nDat):
            delta[i] = self.sample_delta_i(
                            curr_cluster_state, cand_cluster_state, eta,
                            log_likelihood[i], delta[i], unifs[i], scratch,
                            )
        # clean indices (clear out dropped clusters, unused candidate clusters, and re-index)
        delta, zeta, sigma = self.clean_delta_zeta_sigma(delta, zeta, sigma)
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta, sigma)
        self.samples.rho[self.curr_iter]   = self.sample_rho(self.curr_delta, zeta, sigma)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_r, self.curr_rho, self.curr_delta, alpha, beta, xi, tau,
                )
        self.samples.sigma[self.curr_iter] = self.sample_sigma(
                self.curr_zeta, self.curr_r, self.curr_rho, self.curr_delta, xi, tau,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter]  = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.xi[self.curr_iter]    = self.sample_xi(self.curr_sigma, xi)
        self.samples.tau[self.curr_iter]   = self.sample_tau(self.curr_sigma, self.curr_xi)
        self.samples.eta[self.curr_iter]   = self.sample_eta(eta, self.curr_delta)
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
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.mkdir(folder)
        if os.path.exists(path):
            os.remove(path)

        zetas  = np.vstack([
            np.hstack((np.ones((zeta.shape[0], 1)) * i, zeta))
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        sigmas = np.vstack([
            np.hstack((np.ones((sigma.shape[0], 1)) * i, sigma))
            for i, sigma in enumerate(self.samples.sigma[nBurn :: nThin])
            ])
        rhos   = self.samples.rho[nBurn::nThin].reshape(-1, self.nCat)
        alphas = self.samples.alpha[nBurn :: nThin]
        betas  = self.samples.beta[nBurn :: nThin]
        xis    = self.samples.xi[nBurn :: nThin]
        taus   = self.samples.tau[nBurn :: nThin]
        deltas = self.samples.delta[nBurn :: nThin]
        rs     = self.samples.r[nBurn :: nThin]
        etas   = self.samples.eta[nBurn :: nThin]

        out = {
            'zetas'  : zetas,
            'sigmas' : sigmas,
            'alphas' : alphas,
            'betas'  : betas,
            'xis'    : xis,
            'taus'   : taus,
            'rhos'   : rhos,
            'rs'     : rs,
            'deltas' : deltas,
            'etas'   : etas,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'nCat'   : self.nCat,
            'nSigma' : self.nSigma,
            'sigmaunity' : self.sigma_unity,
            'cats'   : self.data.Cats,
            'V'      : self.data.V,
            'W'      : self.data.W,
            }
        
        try:
            out['Y'] = self.data.Y
        except AttributeError:
            pass
        
        with open(path, 'wb') as file:
            pickle.dump(out, file)

        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        self.data.Yp[self.data.Yp <= 1e-6] = 1e-6
        return
    
    def categorical_considerations(self):
        """ Builds the CatMat """
        cats = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(self.data.Cats)))
        self.CatMat = cats[:, None] == np.arange(len(self.data.Cats))
        return

    def build_sigma_unity(self):
        sigma_unity = np.zeros(self.nCol + self.nCat, dtype = bool)
        # declare sigma_0 = 1
        sigma_unity[0] = True
        # advance iterator to the start of the categorical columns
        iter = self.nCol
        # for each set of categorical columns (each projected gamma) set first sigma = 1
        for ncat in self.data.Cats:
            for x in range(ncat):
                if (x == 0):
                    sigma_unity[iter] = True
                iter += 1
        self.sigma_unity = sigma_unity
        self.nSigma = sum(~self.sigma_unity)
        return
    
    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_alpha = GammaPrior(0.5, 0.5),
            prior_beta  = GammaPrior(2., 2.),
            prior_xi    = GammaPrior(2., 2.),
            prior_tau   = GammaPrior(2., 2.),
            p           = 10,
            max_clust_count = 300,
            ):
        assert type(data) is MixedData
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCat = self.data.nCat
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.nCats = self.data.Cats.shape[0]
        self.priors = Prior(prior_eta, prior_alpha, prior_beta, prior_xi, prior_tau)
        self.set_projection()
        self.categorical_considerations()
        self.build_sigma_unity()
        # self.pool = Pool(processes = 8, initializer = limit_cpu())
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = np.bincount(self.samples.delta[s], minlength = int(dmax + 1 + m))
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            new_zetas = gamma(
                shape = self.samples.alpha[s],
                scale = 1. / self.samples.beta[s],
                size = (m, self.nCol + self.nCat),
                )
            new_sigmas = np.ones(new_zetas.shape)
            new_sigmas[:,np.where(~self.sigma_unity)[0]] = gamma(
                shape = self.samples.xi[s],
                scale = 1 / self.samples.tau[s],
                size = (m, self.nSigma)
                )
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            sigma = np.vstack((self.samples.sigma[s], new_sigmas))[deltas]
            new_gammas.append(gamma(shape = zeta, scale = 1 / sigma))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1, m = 10):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m)
        hypcube = euclidean_to_hypercube(gammas[:,:self.nCol])
        simplex = []
        cat_idx = np.where(self.sigma_unity)[0][1:]
        for i in range(cat_idx.shape[0]):
            cat_start = cat_idx[i]
            try:
                cat_end = cat_idx[i + 1]
            except IndexError:
                cat_end = self.sigma_unity.shape[0]
            simplex.append(euclidean_to_simplex(gammas[:,cat_start:cat_end]))
        return np.hstack([hypcube] + simplex)

    def generate_posterior_predictive_angular(self, n_per_sample = 1, m = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample, m)
        return euclidean_to_angular(hyp)

    def write_posterior_predictive(self, path, n_per_sample = 1):
        colnames_y = ['Y_{}'.format(i) for i in range(self.nCol)]
        colnames_p = [
            ['p_{}_{}'.format(i,j) for j in range(catlength)]
            for i, catlength in enumerate(self.cats)
            ]
        colnames_p = list(chain(*colnames_p))

        thetas = pd.DataFrame(
                self.generate_posterior_predictive_hypercube(n_per_sample),
                # self.generate_posterior_predictive_angular(n_per_sample),
                #columns = ['theta_{}'.format(i) for i in range(1, self.nCol)],
                columns = colnames_y + colnames_p
                )
        thetas.to_csv(path, index = False)
        return

    def load_data(self, path):
        with open(path, 'rb') as file:
            out = pickle.load(file)
        
        deltas = out['deltas']
        etas   = out['etas']
        zetas  = out['zetas']
        sigmas = out['sigmas']
        alphas = out['alphas']
        betas  = out['betas']
        xis    = out['xis']
        taus   = out['taus']
        rs     = out['rs']
        rhos   = out['rhos']
        su     = out['sigmaunity']
        cats   = out['cats']

        self.sigma_unity = su

        self.nSamp  = deltas.shape[0]
        self.nDat   = deltas.shape[1]
        self.nCat   = rhos.shape[1]
        self.nCol   = alphas.shape[1] - self.nCat
        self.nCats  = su.sum() - 1 # number of projections - 1.
        self.nSigma = self.nCol + self.nCat - self.nCats - 1
        self.cats   = cats

        self.V = out['V']
        self.W = out['W']
        try:
            self.Y = out['Y']
        except KeyError:
            pass

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol, self.nCat, self.nCats)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.xi    = xis
        self.samples.tau   = taus
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.sigma = [sigmas[np.where(sigmas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.r     = rs
        self.samples.rho   = rhos.reshape(self.nSamp, self.nDat, self.nCat)
        return

    def __init__(self, path):
        self.load_data(path)
        return

if __name__ == '__main__':
    from data import MixedData
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    raw = read_csv('./datasets/ad2_cover_x.csv')
    data = MixedData(raw, cat_vars = np.array([0,3], dtype = int), decluster = False, quantile = 0.999)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(4000)
    model.write_to_disk('./test/results.pickle', 2000, 2)
    res = Result('./test/results.pickle')
    res.write_posterior_predictive('./test/postpred.csv')

# EOF
