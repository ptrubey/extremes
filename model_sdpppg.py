from numpy.random import choice, gamma, beta, uniform
from collections import namedtuple
from itertools import repeat
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
import sqlite3 as sql
from math import ceil, log
from scipy.special import gammaln

import cUtility as cu
from samplers import DirichletProcessSampler
from cProjgamma import sample_alpha_k_mh_summary, sample_alpha_1_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube, Data
from projgamma import GammaPrior


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
    out -= np.einsum('jd,nd->nj', aBeta, aY)                          # e^(- beta y)
    return out

def update_zeta_j_wrapper(args):
    # parse arguments
    curr_zeta_j, n_j, Y_js, lY_js, alpha, beta, xi, tau = args
    prop_zeta_j = np.empty(curr_zeta_j.shape)
    prop_zeta_j[0] = sample_alpha_1_mh_summary(
        curr_zeta_j[0], n_j, Y_js[0], lY_js[0], alpha[0], beta[0]
        )
    for i in range(1, curr_zeta_j.shape[0]):
        prop_zeta_j[i] = sample_alpha_k_mh_summary(
            curr_zeta_j[i], n_j, Y_js[i], lY_js[i], 
            alpha[i], beta[i], xi[i-1], tau[i-1],
            )
    return prop_zeta_j

def update_sigma_j_wrapper(args):
    zeta_j, n_j, Y_js, xi, tau = args
    prop_sigma_j = np.ones(zeta_j.shape)
    As = n_j * zeta_j[1:] + xi
    Bs = Y_js[1:] + tau
    prop_sigma_j[1:] = gamma(shape = As, scale = 1 / Bs)
    return prop_sigma_j

def sample_gamma_shape_wrapper(args):
    return sample_alpha_k_mh_summary(*args)

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

class Chain(DirichletProcessSampler):
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

    def sample_delta_i(self, curr_cluster_state, cand_cluster_state, 
                            eta, log_likelihood_i, delta_i, p, scratch):
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
        return gamma(shape = alpha, scale = 1/beta, size = (m, self.nCol))

    def sample_sigma_new(self, xi, tau, m):
        return np.hstack((
                    np.ones((m, 1)), 
                    gamma(shape = xi, scale = 1/tau, size = (m, self.nCol - 1))
                    ))

    def sample_alpha(self, zeta, curr_alpha):
        n    = zeta.shape[0]
        zs   = zeta.sum(axis = 0)
        lzs  = np.log(zeta).sum(axis = 0)
        args = zip(
            curr_alpha, repeat(n), zs, lzs,
            repeat(self.priors.alpha.a), repeat(self.priors.alpha.b),
            repeat(self.priors.beta.a), repeat(self.priors.beta.b),
            )
        res = map(sample_gamma_shape_wrapper, args)
        return np.array(list(res))

    def sample_beta(self, zeta, alpha):
        n = zeta.shape[0]
        zs = zeta.sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        return gamma(shape = As, scale = 1 / Bs)

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
        ss = sigma[:,1:].sum(axis = 0)
        As = n * xi + self.priors.tau.a
        Bs = ss + self.priors.tau.b
        return gamma(shape = As, scale = 1 / Bs)

    def sample_r(self, delta, zeta, sigma):
        As = np.einsum('il->i', zeta[delta])
        Bs = np.einsum('il,il->i', self.data.Yp, sigma[delta])
        return gamma(shape = As, scale = 1 / Bs)
    
    def sample_eta(self, curr_eta, delta):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max() + 1
        bb = self.priors.beta.b - log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def sample_zeta(self, curr_zeta, r, delta, alpha, beta, xi, tau):
        dmat = delta[:,None] == np.arange(delta.max() + 1)
        Y    = r[:,None] * self.data.Yp
        n    = dmat.sum(axis = 0)
        Ysv  = (Y.T @ dmat).T
        lYsv = (np.log(Y).T @ dmat).T
        args = zip(
            curr_zeta, n, Ysv, lYsv, 
            repeat(alpha), repeat(beta), 
            repeat(xi), repeat(tau),
            )
        res = map(update_zeta_j_wrapper, args)
        return np.array(list(res))
    
    def sample_sigma(self, zeta, r, delta, xi, tau):
        dmat = delta[:, None] == np.arange(delta.max() + 1)
        Y = r[:,None] * self.data.Yp
        n = dmat.sum(axis = 0)
        Ysv = (Y.T @ dmat).T # (J x d)
        args = zip(zeta, n, Ysv, repeat(xi), repeat(tau))
        res = np.array(list(map(update_sigma_j_wrapper, args)))
        res[res < 1e-12] = 1e-12
        return res
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.xi[0] = 1.
        self.samples.tau[0] = 1.
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 30, self.nCol))
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 30, self.nCol))
        self.samples.eta[0] = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 30, size = self.nDat)
        self.samples.delta[0][-1] = np.arange(self.max_clust_count - 30)[-1]
        self.samples.r[0] = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0], self.samples.sigma[0],
                )
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

        self.curr_iter += 1
        # normalizing constant for product of Gammas
        log_likelihood = dprodgamma_log_my_mt(r.reshape(-1,1) * self.data.Yp, zeta, sigma)
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
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_r, self.curr_delta, alpha, beta, xi, tau,
                )
        self.samples.sigma[self.curr_iter] = self.sample_sigma(
                zeta, self.curr_r, self.curr_delta, xi, tau,
                )
        self.samples.alpha[self.curr_iter] = self.sample_alpha(self.curr_zeta, alpha)
        self.samples.beta[self.curr_iter]  = self.sample_beta(self.curr_zeta, self.curr_alpha)
        self.samples.xi[self.curr_iter]    = self.sample_xi(self.curr_sigma, xi)
        self.samples.tau[self.curr_iter]   = self.sample_tau(self.curr_sigma, self.curr_xi)
        self.samples.eta[self.curr_iter]   = self.sample_eta(eta, self.curr_delta)
        return

    def sample(self, ns):
        self.initialize_sampler(ns)
        print_string = '\rSampling {:.2%} Completed, {} Clusters     '
        print(print_string.format(self.curr_iter / ns, self.nDat), end = '')
        while (self.curr_iter < ns):
            if (self.curr_iter % 100) == 0:
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
            'rs'     : rs,
            'deltas' : deltas,
            'etas'   : etas,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'V'      : self.data.V,
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
        return

    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_alpha = GammaPrior(0.5, 0.5),
            prior_beta  = GammaPrior(2., 2.),
            prior_xi    = GammaPrior(0.5, 0.5),
            prior_tau   = GammaPrior(2., 2.),
            p           = 10,
            max_clust_count = 300,
            ):
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = Prior(prior_eta, prior_alpha, prior_beta, prior_xi, prior_tau)
        self.set_projection()
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

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = alphas.shape[1]

        self.V = out['V']
        try:
            self.Y = out['Y']
        except KeyError:
            pass
        self.samples       = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.xi    = xis
        self.samples.tau   = taus
        self.samples.zeta  = [
            zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)
            ]
        self.samples.sigma = [
            sigmas[np.where(sigmas.T[0] == i)[0], 1:] for i in range(self.nSamp)
            ]
        self.samples.r     = rs
        return

    def __init__(self, path):
        self.load_data(path)
        return

# EOF

if __name__ == '__main__':
    from data import Data_From_Raw
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    raw = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    data.write_empirical('./test/empirical.csv')
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(4000)
    model.write_to_disk('./test/results.pickle', 2000, 2)
    res = Result('./test/results.pickle')
    res.write_posterior_predictive('./test/postpred.csv')
    # EOL



# EOF 2
