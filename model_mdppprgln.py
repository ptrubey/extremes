from numpy.random import choice, gamma, beta, uniform, normal, multinomial
from scipy.stats import invwishart
from numpy.linalg import cholesky, inv
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
# np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from multiprocessing import Pool
from energy import limit_cpu

EPS = np.finfo(float).eps

import cUtility as cu
from samplers import DirichletProcessSampler, cumsoftmax2d, pt_dp_sample_cluster, bincount2D_vectorized
from data import Projection, MixedDataBase, MixedData, euclidean_to_angular,    \
    euclidean_to_hypercube, euclidean_to_simplex, euclidean_to_psphere,         \
    category_matrix, euclidean_to_catprob
from projgamma import GammaPrior, NormalPrior, InvWishartPrior,                 \
    pt_logd_cumdircategorical_mx_ma_inplace_unstable, pt_logd_mvnormal_mx_st,   \
    logd_gamma_my, logd_mvnormal_mx_st, logd_invwishart_ms,                     \
    pt_logd_cumdirmultinom_paired_yt, pt_logd_projgamma_my_mt_inplace_unstable, \
    pt_logd_projgamma_paired_yt       
from cov import PerObsTemperedOnlineCovariance

Prior = namedtuple('Prior', 'eta mu Sigma')

class Samples(object):
    zeta  = None
    mu    = None
    Sigma = None
    delta = None
    eta   = None
    lzhist = None

    def __init__(self, nSamp, nDat, nCol, nCat, nTemp):
        """
        nCol: number of 
        nCat: number of categorical columns
        nCats: number of categorical variables        
        """
        tCol = nCol + nCat
        self.zeta  = [None] * (nSamp + 1)
        self.mu    = np.empty((nSamp + 1, nTemp, tCol))
        self.Sigma = np.empty((nSamp + 1, nTemp, tCol, tCol))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.eta   = np.empty((nSamp + 1, nTemp))
        return

def Samples_(Samples):
    def __init__(self, nSamp, nDat, nCol, nCat):
        tCol = nCol + nCat 
        self.zeta  = [None] * (nSamp)
        self.mu    = np.empty((nSamp, tCol))
        self.Sigma = np.empty((nSamp, tCol, tCol))
        self.delta = np.empty((nSamp, nDat))
        self.eta   = np.empty((nSamp))
        return

class Chain(DirichletProcessSampler, Projection):
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_mu(self):
        return self.samples.mu[self.curr_iter]
    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter]
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter]
    @property
    def curr_eta(self):
        return self.samples.eta[self.curr_iter]
    @property
    def curr_cluster_count(self):
        return self.curr_delta[0].max() + 1
    def average_cluster_count(self, ns):
        acc = self.samples.delta[(ns//2):][:,0].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)

    # Adaptive Metropolis Placeholders
    am_cov_c  = None
    am_mean_c = None
    am_cov_i  = None
    am_mean_i = None
    am_Sigma  = None
    am_scale  = None
    max_clust_count = None
    swap_attempts = None
    swap_succeeds = None

    def sample_delta(self, delta, zeta, eta):
        log_likelihood = self.log_delta_likelihood(zeta)
        p = uniform(size = (self.nDat, self.nTemp))
        pt_dp_sample_cluster(delta, log_likelihood, p, eta)
        return

    def clean_delta_zeta(self, delta, zeta):
        """
        Find populated clusters, re-index them, 
        keep only parameters associated with extant clusters
        ---
        inputs:
            delta : cluster indicator vector (n)
            zeta  : cluster parameter matrix (J x d)
        outputs:
            delta : cluster indicator vector (n)
            zeta  : cluster parameter matrix (J* x d)
        """
        # Find populated clusters, re-index them
        for t in range(self.nTemp):
            keep, delta[t] = np.unique(delta[t], return_inverse = True)
            zeta[t][:keep.shape[0]] = zeta[t,keep]
        return

    def sample_zeta_new(self, mu, Sigma_chol):
        """ Sample new zetas as log-normal (sample normal, then exponentiate) """
        sizes = (self.nTemp, self.max_clust_count, self.nCat + self.nCol)
        out = np.empty(sizes)
        np.einsum('tzy,tjy->tjz', np.triu(Sigma_chol), normal(size = sizes), out = out)
        out += mu[:,None,:]
        np.exp(out, out=out)
        return out

    def am_covariance_matrices(self, delta, index):
        return self.am_Sigma.cluster_covariance(delta)[index]

    def sample_zeta(self, zeta, delta, mu, Sigma_chol, Sigma_inv):
        """
        zeta      : (t x J x D)
        delta     : (t x n)
        r         : (t x n)
        mu        : (t x D)
        Sigma_cho : (t x D x D)
        Sigma_inv : (t x D x D)
        """
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_cluster_state = (curr_cluster_state == 0)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)
        idx = np.where(~cand_cluster_state)
        covs = self.am_covariance_matrices(delta, idx)
        
        am_alpha = np.zeros((self.nTemp, self.max_clust_count))
        am_alpha[:] = -np.inf
        am_alpha[idx] = 0.

        zcurr = zeta.copy()
        with np.errstate(divide = 'ignore'):
            lzcurr = np.log(zeta)
        lzcand = lzcurr.copy()
        lzcand[idx] += np.einsum(
            'mpq,mq->mp', 
            cholesky(self.am_scale * covs), 
            normal(size = (idx[0].shape[0], self.tCol)),
            )
        zcand = np.exp(lzcand)
        
        am_alpha += self.log_zeta_likelihood(zcand, delta, delta_ind_mat)
        am_alpha -= self.log_zeta_likelihood(zcurr, delta, delta_ind_mat)
        with np.errstate(invalid = 'ignore'):
            am_alpha *= self.itl[:,None]
        am_alpha += self.log_logzeta_prior(lzcand, mu, Sigma_chol, Sigma_inv)
        am_alpha -= self.log_logzeta_prior(lzcurr, mu, Sigma_chol, Sigma_inv)
        
        keep = np.where(np.log(uniform(size = am_alpha.shape)) < am_alpha)
        zcurr[keep] = zcand[keep]
        return zcurr

    def sample_Sigma(self, zeta, mu, extant_clusters):
        n = extant_clusters.sum(axis = 1)
        diff = (np.log(zeta) - mu[:,None,:]) * extant_clusters[:,:,None]
        C = np.einsum('tjd,tje->tde', diff, diff)
        _psi = self.priors.Sigma.psi + C * self.itl[:,None,None]
        _nu  = self.priors.Sigma.nu + n * self.itl
        tCol = self.nCol + self.nCat
        out = np.empty((self.nTemp, tCol, tCol))
        for i in range(self.nTemp):
            out[i] = invwishart.rvs(df = _nu[i], scale = _psi[i])
        return out

    def sample_mu(self, zeta, Sigma_inv, extant_clusters):
        n = extant_clusters.sum(axis = 1)
        assert np.all(zeta[extant_clusters] > 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            lzbar = np.nansum(np.log(zeta) * extant_clusters[:,:,None], axis = 1) / n[:,None]
        _Sigma = inv(n[:,None,None] * Sigma_inv + self.priors.mu.SInv)
        _mu = np.einsum(
            'tjl,tl->tj', 
            _Sigma, 
            self.priors.mu.SInv @ self.priors.mu.mu + 
                np.einsum('tjl,tl->tj', Sigma_inv, n[:,None] * lzbar),
            )
        out = np.zeros((self.nTemp, self.tCol))
        np.einsum(
            'tkl,tl->tk', cholesky(_Sigma), 
            normal(size = (self.nTemp, self.tCol)),
            out = out,
            )
        out += _mu
        return out

    def sample_eta(self, curr_eta, delta):
        """
        curr_eta : (t)
        delta    : (t x n)
        """
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max(axis = 1) + 1
        bb = self.priors.eta.b - np.log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        id = uniform(self.nTemp) > eps
        aaa = aa * id + (aa - 1) * (1 - id)
        return gamma(shape = aaa, scale = 1 / bb)

    def log_delta_likelihood(self, zeta):
        out = np.zeros((self.nDat, self.nTemp, self.max_clust_count))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pt_logd_projgamma_my_mt_inplace_unstable(
                out, self.data.Yp , zeta[:,:,:self.nCol], self.sigma_ph1,
                )
            pt_logd_cumdircategorical_mx_ma_inplace_unstable(
                out, self.data.W, zeta[:,:,self.nCol:], self.CatMat,
                )
        np.nan_to_num(out, False, -np.inf)
        return out
    
    def log_zeta_likelihood(self, zeta, delta, delta_ind_mat):
        out = np.zeros((self.nTemp, self.max_clust_count))
        zetas = zeta[
            self.temp_unravel, delta.ravel(),
            ].reshape(self.nTemp, self.nDat, self.tCol)
        out += np.einsum(
            'tn,tnj->tj',
            pt_logd_projgamma_paired_yt(
                self.data.Yp, zetas[:,:,:self.nCol], self.sigma_ph2,
                ),
            delta_ind_mat,
            )
        out += np.einsum(
            'tn,tnj->tj',
            pt_logd_cumdirmultinom_paired_yt(
                self.data.W, zetas[:,:,self.nCol:], self.CatMat,
                ),
            delta_ind_mat,
            )
        return out
    
    def log_logzeta_prior(self, logzeta, mu, Sigma_chol, Sigma_inv):
        """
        logzeta   :  (t, j, d)
        mu        :  (t, d)
        Sigma_inv :  (t,d,d)
        """
        return pt_logd_mvnormal_mx_st(logzeta, mu, Sigma_chol, Sigma_inv)

    def log_tempering_likelihood(self):
        out = np.zeros(self.nTemp)
        out += pt_logd_projgamma_paired_yt(
            self.data.Yp, 
            self.curr_zeta[:,:,:self.nCol][
                self.temp_unravel, 
                self.curr_delta.ravel(),
                ].reshape(self.nTemp, self.nDat, self.nCol),
            self.sigma_ph2,
            ).sum(axis = 1)
        out += pt_logd_cumdirmultinom_paired_yt(
            self.data.W, 
            self.curr_zeta[:,:,self.nCol:][
                self.temp_unravel,
                self.curr_delta.ravel(),
                ].reshape(self.nTemp, self.nDat, self.nCat),
            self.CatMat,
            ).sum(axis = 1)
        return out

    def log_tempering_prior(self):
        out = np.zeros(self.nTemp)
        Sigma_cho = cholesky(self.curr_Sigma)
        Sigma_inv = inv(self.curr_Sigma)
        extant_clusters = (bincount2D_vectorized(self.curr_delta, self.max_clust_count) > 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out += np.nansum(
                extant_clusters * pt_logd_mvnormal_mx_st(
                    np.log(self.curr_zeta), self.curr_mu, Sigma_cho, Sigma_inv,
                    ),
                axis = 1,
                )
        out += logd_mvnormal_mx_st(self.curr_mu, *self.priors.mu)
        out += logd_invwishart_ms(self.curr_Sigma, *self.priors.Sigma)
        out += logd_gamma_my(self.curr_eta, *self.priors.eta)
        return out

    def initialize_sampler(self, ns):
        # Samples
        self.samples = Samples(ns, self.nDat, self.nCol, self.nCat, self.nTemp)
        self.samples.mu[0] = 0.
        self.samples.Sigma[0] = np.eye(self.tCol) * 2.
        self.samples.zeta[0] = gamma(
                shape = 2., scale = 2., 
                size = (self.nTemp, self.max_clust_count, self.tCol),
                )
        self.samples.eta[0] = 10.
        self.samples.delta[0] = choice(
            self.max_clust_count - 20, 
            size = (self.nTemp, self.nDat),
            )
        # Iterator
        self.curr_iter = 0
        # Adaptive Metropolis related        
        self.am_cov_i     = np.empty((self.nDat, self.nTemp, self.tCol, self.tCol))
        self.am_cov_i[:]  = np.eye(self.tCol, self.tCol)[None,None,:,:] * 1e-4
        self.am_mean_i    = np.empty((self.nDat, self.nTemp, self.tCol))
        self.am_mean_i[:] = 0.
        self.am_cov_c     = np.empty((self.nTemp, self.max_clust_count, self.tCol, self.tCol))
        self.am_mean_c    = np.empty((self.nTemp, self.max_clust_count, self.tCol))
        self.am_n_c       = np.zeros((self.nTemp, self.max_clust_count))
        self.am_alpha     = np.zeros((self.nTemp, self.max_clust_count))
        self.swap_attempts = np.zeros((self.nTemp, self.nTemp))
        self.swap_succeeds = np.zeros((self.nTemp, self.nTemp))
        # Placeholders
        self.sigma_ph1 = np.ones((self.nTemp, self.max_clust_count, self.nCol))
        self.sigma_ph2 = np.ones((self.nTemp, self.nDat, self.nCol))
        self.zeta_shape = (self.nTemp, self.nDat, self.nCol + self.nCat)
        return

    def update_am_cov(self):
        """ Online updating for Adaptive Metropolis Covariance per obsv. """
        lzeta = np.swapaxes(
            np.log(
                self.curr_zeta[
                    self.temp_unravel, self.curr_delta.ravel()
                    ].reshape(
                        self.nTemp, self.nDat, self.tCol
                        )
                ),
            0, 1,
            )
        self.am_Sigma.update(lzeta)
        return

    def try_tempering_swap(self):
        ci = self.curr_iter
        # declare log-likelihood, log-prior
        lpl = self.log_tempering_likelihood()
        lpp = self.log_tempering_prior()
        # declare swap choices
        sw  = choice(self.nTemp, 2 * self.nSwap_per, replace = False).reshape(-1, 2)
        for s in sw:
            # record attempted swap
            self.swap_attempts[s[0],s[1]] += 1
            self.swap_attempts[s[1],s[0]] += 1
        # compute swap log-probability
        sw_alpha = np.zeros(sw.shape[0])
        sw_alpha += lpl[sw.T[1]] - lpl[sw.T[0]]
        sw_alpha *= self.itl[sw.T[0]] - self.itl[sw.T[1]]
        sw_alpha += lpp[sw.T[1]] - lpp[sw.T[0]]
        logp = np.log(uniform(size = sw_alpha.shape))
        for tt in sw[np.where(logp < sw_alpha)[0]]:
            # report successful swap
            self.swap_succeeds[tt[0],tt[1]] += 1
            self.swap_succeeds[tt[1],tt[0]] += 1
            # do the swap
            self.samples.zeta[ci][tt[0]], self.samples.zeta[ci][tt[1]] =   \
                self.samples.zeta[ci][tt[1]].copy(), self.samples.zeta[ci][tt[0]].copy()
            self.samples.mu[ci][tt[0]], self.samples.mu[ci][tt[1]] = \
                self.samples.mu[ci][tt[1]].copy(), self.samples.mu[ci][tt[0]].copy()
            self.samples.Sigma[ci][tt[0]], self.samples.Sigma[ci][tt[1]] = \
                self.samples.Sigma[ci][tt[1]].copy(), self.samples.Sigma[ci][tt[0]].copy()
            self.samples.delta[ci][tt[0]], self.samples.delta[ci][tt[1]] = \
                self.samples.delta[ci][tt[1]].copy(), self.samples.delta[ci][tt[0]].copy()
            self.samples.eta[ci][tt[0]], self.samples.eta[ci][tt[1]] =     \
                self.samples.eta[ci][tt[1]].copy(), self.samples.eta[ci][tt[0]].copy()
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy()
        zeta  = self.curr_zeta.copy()
        mu    = self.curr_mu
        Sigma = self.curr_Sigma
        Sigma_cho = cholesky(self.curr_Sigma)
        Sigma_inv = inv(Sigma)
        eta   = self.curr_eta
        
        # Adaptive Metropolis Update
        self.update_am_cov()
        
        # Advance the iterator
        self.curr_iter += 1
        ci = self.curr_iter

        # Sample new candidate clusters
        cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_clusters = np.where(cluster_state == 0)
        zeta[cand_clusters] = self.sample_zeta_new(mu, Sigma_cho)[cand_clusters]
        
        # Update cluster assignments and re-index
        self.sample_delta(delta, zeta, eta)
        self.clean_delta_zeta(delta, zeta)
        self.samples.delta[ci] = delta
        
        # do rest of sampling
        extant_clusters = (cluster_state > 0)
        self.samples.zeta[ci] = self.sample_zeta(zeta, delta, mu, Sigma_cho, Sigma_inv)
        self.samples.mu[ci] = self.sample_mu(zeta, Sigma_inv, extant_clusters)
        self.samples.Sigma[ci] = self.sample_Sigma(zeta, mu, extant_clusters)
        self.samples.eta[ci] = self.sample_eta(eta, self.curr_delta)

        # Attempt Swap:
        if self.curr_iter >= self.swap_start:
           self.try_tempering_swap()
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        folder = os.path.split(path)[0]
        if not os.path.exists(folder):
            os.mkdir(folder)
        if os.path.exists(path):
            os.remove(path)
        # assemble data
        zetas  = np.vstack([
            np.hstack((np.ones((zeta[0].shape[0], 1)) * i, zeta[0]))
            for i, zeta in enumerate(self.samples.zeta[(nBurn+1) :: nThin])
            ])
        mus    = self.samples.mu[(nBurn+1) :: nThin, 0]
        Sigmas = self.samples.Sigma[(nBurn+1) :: nThin, 0]
        deltas = self.samples.delta[(nBurn+1) :: nThin, 0]
        etas   = self.samples.eta[(nBurn+1) :: nThin, 0]
        # make output dictionary
        out = {
            'zetas'  : zetas,
            'mus'    : mus,
            'Sigmas' : Sigmas,
            'deltas' : deltas,
            'etas'   : etas,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'nCat'   : self.nCat,
            'cats'   : self.data.Cats,
            'V'      : self.data.V,
            'W'      : self.data.W,
            'swap_y' : self.swap_succeeds,
            'swap_n' : self.swap_attempts - self.swap_succeeds,
            'swap_p' : self.swap_succeeds / (self.swap_attempts + 1e-9),
            }
        # try to add outcome / radius to dictionary
        for attr in ['Y','R','P']:
            if hasattr(self.data, attr):
                out[attr] = self.data.__dict__[attr]
        # write to disk
        with open(path, 'wb') as file:
            pickle.dump(out, file)
        return
    
    def categorical_considerations(self):
        """ Builds the CatMat """
        self.CatMat = category_matrix(self.data.Cats)
        return
    
    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_mu    = (0, 3.),
            prior_Sigma = (10, 0.5),
            p           = 10,
            max_clust_count = 300,
            ntemps      = 3,
            stepping    = 1.05,
            ):
        assert type(data) is MixedData
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCat = self.data.nCat
        self.nCol = self.data.nCol
        self.tCol = self.nCol + self.nCat
        self.nDat = self.data.nDat
        self.nCats = self.data.Cats.shape[0]

        # Setting Priors
        _prior_mu = NormalPrior(
            np.ones(self.tCol) * prior_mu[0], 
            np.eye(self.tCol) * np.sqrt(prior_mu[1]),
            np.eye(self.tCol) / np.sqrt(prior_mu[1]),            
            )
        _prior_Sigma = InvWishartPrior(
            self.tCol + prior_Sigma[0],
            np.eye(self.tCol) * prior_Sigma[1],
            )
        self.priors = Prior(prior_eta, _prior_mu, _prior_Sigma)
        self.set_projection()
        self.categorical_considerations()

        # Parallel Tempering
        self.nTemp = ntemps
        self.itl = 1 / stepping**np.arange(ntemps)
        self.temp_unravel = np.repeat(np.arange(self.nTemp), self.nDat)
        self.nSwap_per = self.nTemp // 2
        self.swap_start = 100

        # Adaptive Metropolis
        self.am_Sigma = PerObsTemperedOnlineCovariance(
            self.nTemp, self.nDat, self.tCol, self.max_clust_count
            )
        self.am_scale = 2.38**2 / self.tCol
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = np.bincount(self.samples.delta[s], minlength = int(dmax + 1 + m))
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            new_zetas = np.empty((m, self.nCol + self.nCat))
            np.einsum(
                'zy,jy->jz', 
                cholesky(self.samples.Sigma[s]), 
                normal(size = (m, self.nCol + self.nCat)),
                out = new_zetas,
                )
            new_zetas += self.samples.mu[s][None,:]
            np.exp(new_zetas, out = new_zetas)
            prob = ljs / ljs.sum()
            deltas = cu.generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
            new_gammas.append(gamma(shape = zeta))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1, m = 10):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample, m)
        # hypercube transformation for real variates
        hypcube = euclidean_to_hypercube(gammas[:,:self.nCol])
        # simplex transformation for categ variates
        simplex_reverse = []
        indices = list(np.arange(self.nCol + self.nCat))
        # Foe each category, last first
        for i in list(range(self.cats.shape[0]))[::-1]:
            # identify the ending index (+1 to include boundary)
            cat_length = self.cats[i]
            cat_end = indices.pop() + 1
            # identify starting index
            for _ in range(cat_length - 1):
                cat_start = indices.pop()
            # transform gamma variates to simplex
            simplex_reverse.append(euclidean_to_simplex(gammas[:,cat_start:cat_end]))
        # stack hypercube and categorical variables side by side.
        return np.hstack([hypcube] + simplex_reverse[::-1])

    def generate_posterior_predictive_angular(self, n_per_sample = 1, m = 10):
        hyp = self.generate_posterior_predictive_hypercube(n_per_sample, m)
        return euclidean_to_angular(hyp)

    def generate_posterior_predictive_spheres(self, n_per_sample):
        rhos = self.generate_posterior_predictive_gammas(n_per_sample)[:,self.nCol:] # (s,D)
        CatMat = category_matrix(self.data.Cats) # (C,d)
        return euclidean_to_catprob(rhos, CatMat)
        
    def generate_conditional_posterior_predictive_radii(self):
        """ r | zeta, V ~ Gamma(r | sum(zeta), sum(V)) """
        # As = np.einsum('il->i', zeta[delta].T[:self.nCol].T)
        # Bs = np.einsum('il->i', self.data.Yp)
        shapes = np.array([
            zeta[delta]
            for delta, zeta
            in zip(self.samples.delta, self.samples.zeta)
            ]).sum(axis = 2)
        rates = self.data.V.sum(axis = 1)[None,:]
        rs = gamma(shape = shapes, scale = 1 / rates)
        return rs

    def generate_conditional_posterior_predictive_gammas(self):
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]),0,1) # (n,s,d)
        W = np.hstack((np.zeros((self.nDat, self.nCol)), self.data.W)) # (n,d)
        return gamma(shape = zetas + W[:,None,:])

    def generate_conditional_posterior_predictive_spheres(self):
        """ pi | zeta, delta = normalized rho
        currently discarding generated Y's, keeping latent pis
        """
        rhos = self.generate_conditional_posterior_predictive_gammas()[:,:,self.nCol:]
        CatMat = category_matrix(self.data.Cats) # (C,d)
        shro = rhos @ CatMat.T # (s,n,C)
        nrho = np.einsum('snc,cd->snd', shro, CatMat) # (s,n,d)
        pis = rhos / nrho
        return pis

    def generate_new_conditional_posterior_predictive_spheres(self, Vnew, Wnew):
        rhos   = self.generate_new_conditional_posterior_predictive_gammas(
            Vnew, Wnew,
            )[:,:,self.nCol:]
        CatMat = category_matrix(self.data.Cats)
        shro   = rhos @ CatMat.T
        nrho   = np.einsum('snc,cd->snd', shro, CatMat) # (s,n,d)
        pis    = rhos / nrho
        return pis
    
    def generate_new_conditional_posterior_predictive_radii(self, Vnew, Wnew):
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew, Wnew)
        radii = znew[:,:,:self.nCol].sum(axis = 2)
        return gamma(radii)
    
    def generate_new_conditional_posterior_predictive_gammas(self, Vnew, Wnew):
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew, Wnew)
        return gamma(znew)

    def generate_new_conditional_posterior_predictive_hypercube(self, Vnew, Wnew):
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew, Wnew)
        Ypnew = euclidean_to_psphere(Vnew, 10)
        R = gamma(znew[:,:,:self.nCol].sum(axis = 2))
        G = gamma(znew[:,:,self.nCol:])
        return euclidean_to_hypercube(np.hstack((R[:,:,None] * Ypnew, G)))
    
    def generate_new_conditional_posterior_predictive_euclidean(self, Vnew, Wnew):
        znew = self.generate_new_conditional_posterior_predictive_zetas(Vnew, Wnew)
        Ypnew = euclidean_to_psphere(Vnew, 10)
        R = gamma(znew[:,:,:self.nCol].sum(axis = 2))
        G = gamma(znew[:,:,self.nCol:])
        return np.hstack((R[:,:,None] * Ypnew, G))

    def generate_new_conditional_posterior_predictive_zetas(self, Vnew, Wnew):
        n = Vnew.shape[0]
        Ypnew = euclidean_to_psphere(Vnew, 10)
        
        max_clust_count = self.samples.delta.max() + 20
        zetas = np.einsum(
            'sab,sjb->sja',
            cholesky(self.samples.Sigma),
            normal(size = (self.nSamp, max_clust_count, self.tCol)),
            )
        zetas += self.samples.mu[:,None,:]
        np.exp(zetas, out = zetas)
        weights = np.zeros((self.nSamp, max_clust_count))
        for s in range(self.nSamp):
            zetas[s][:self.samples.zeta[s].shape[0]] = self.samples.zeta[s]
            weights[s] = np.bincount(self.samples.delta, minlength=max_clust_count)
        weights += (weights == 0) * (self.samples.eta / ((weights == 0).sum(axis = 1) + EPS))
        np.log(weights, out = weights)
        loglik = np.zeros((n, self.nSamp, max_clust_count))
        sigma_ph = np.ones((1, max_clust_count, self.nCol))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pt_logd_projgamma_my_mt_inplace_unstable(
                loglik, Ypnew , zetas[None,:,:self.nCol], sigma_ph,
                )
            pt_logd_cumdircategorical_mx_ma_inplace_unstable(
                loglik, Wnew, zetas[None,:,self.nCol:], self.CatMat,
                )
        np.nan_to_num(out, False, -np.inf)
        # combine logprior weights and likelihood under cluster
        weights = weights[None] + loglik
        np.exp(weights, out = weights) # unnormalized cluster probability
        for s in range(self.nSamp):
            weights[s] = cumsoftmax2d(weights[s])
        p = uniform(size = (n, self.nSamp))
        dnew = np.empty((n, self.nSamp), dtype = int)
        znew = np.empty((n, self.nSamp, self.tCol))
        for i in range(n):
            for s in range(self.nSamp):
                dnew[i,s] = multinomial(1, pvals = weights[i,s])
                znew[i,s] = zetas[s,dnew[i,s]]
        return znew
    
    def load_data(self, path):        
        with open(path, 'rb') as file:
            out = pickle.load(file)
        
        deltas = out['deltas']
        etas   = out['etas']
        zetas  = out['zetas']
        mus    = out['mus']
        Sigmas = out['Sigmas']
        cats   = out['cats']
        
        self.data = MixedDataBase(out['V'], out['W'], out['cats'])
        self.nSamp  = deltas.shape[0]
        self.nDat   = deltas.shape[1]
        self.nCat   = self.data.nCat
        self.nCol   = self.data.nCol
        self.tCol   = self.nCol + self.nCat
        self.nCats  = cats.shape[0]
        self.cats   = cats
        self.CatMat = category_matrix(self.data.Cats)
        
        if 'Y' in out.keys():
            self.data.fill_outcome(out['Y'])        
        if 'R' in out.keys():
            self.data.R = out['R']
        if 'P' in out.keys():
            self.data.P = out['P']

        self.samples = Samples(
            self.nSamp, self.nDat, self.nCol, self.nCat, self.nCats
            )
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.mu    = mus
        self.samples.Sigma = Sigmas
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]

        if 'swap_y' in out.keys():
            self.swap_y = out['swap_y']
            self.swap_n = out['swap_n']
            self.swap_p = out['swap_p']
        return

    def __init__(self, path):
        self.load_data(path)
        return

def argparser():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('in_data_path')
    p.add_argument('out_path')
    p.add_argument('cat_vars')
    p.add_argument('--in_outcome_path', default = False)
    p.add_argument('--decluster', default = 'False')
    p.add_argument('--quantile', default = 0.95)
    p.add_argument('--nSamp', default = 20000)
    p.add_argument('--nKeep', default = 10000)
    p.add_argument('--nThin', default = 10)
    p.add_argument('--eta_alpha', default = 2.)
    p.add_argument('--eta_beta', default = 1.)
    return p.parse_args()

class Heap(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        return

if __name__ == '__main__':
    from data import MixedData
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    # p = argparser()
    d = {
        'in_data_path'    : './ad/mammography/data.csv',
        'in_outcome_path' : './ad/mammography/outcome.csv',
        'out_path' : './ad/mammography/results_mdppprgln_test.pkl',
        'cat_vars' : '[5,6,7,8]',
        'decluster' : 'False',
        'quantile' : 0.95,
        'nSamp' : 5000,
        'nKeep' : 2000,
        'nThin' : 3,
        'eta_alpha' : 2.,
        'eta_beta' : 1.,
        }
    p = Heap(**d)

    raw = read_csv(p.in_data_path).values
    out = read_csv(p.in_outcome_path).values
    data = MixedData(
        raw, 
        cat_vars = np.array(eval(p.cat_vars), dtype = int), 
        decluster = eval(p.decluster), 
        quantile = float(p.quantile),
        outcome = out,
        )
    data.fill_outcome(out)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10, ntemps = 3)
    model.sample(p.nSamp)
    model.write_to_disk(p.out_path, p.nKeep, p.nThin)
    res = Result(p.out_path)
    # res.write_posterior_predictive('./test/postpred.csv')

# EOF
