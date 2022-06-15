from numpy.random import choice, gamma, beta, uniform, normal
from numpy.linalg import cholesky
from collections import namedtuple
from itertools import repeat, chain
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log
from scipy.special import gammaln, betaln

import cUtility as cu
from samplers import DirichletProcessSampler
from cProjgamma import sample_alpha_1_mh_summary, sample_alpha_k_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube,              \
    euclidean_to_simplex, MixedDataBase, MixedData
from model_sdpppgln import bincount2D_vectorized, cluster_covariance_mat
from projgamma import logd_loggamma_paired, pt_logd_prodgamma_my_st,        \
    logd_gamma_my,  pt_logd_loggamma_mx_st,                                 \
    pt_logd_cumdirmultinom_mx_ma, pt_logd_cumdirmultinom_paired_yt,         \
    pt_logd_projgamma_my_mt, pt_logd_projgamma_paired_yt, GammaPrior
from multiprocessing import Pool
from energy import limit_cpu

def update_zeta_j_cat(curr_zeta, Ws, alpha, beta, catmat):
    """ Update routine for zeta on categorical/multinomial data """
    curr_log_zeta = np.log(curr_zeta)
    prop_log_zeta = curr_log_zeta.copy()
    offset = normal(scale = 0.3, size = curr_zeta.shape)
    lunifs = np.log(uniform(size = curr_zeta.shape))
    logp = np.zeros(2)
    for i in range(curr_zeta.shape[0]):
        prop_log_zeta[i] += offset[i]
        logp += pt_logd_cumdirmultinom_mx_ma(
            Ws, 
            np.exp(np.vstack((curr_log_zeta, prop_log_zeta))), 
            catmat,
            ).sum(axis = 0)
        logp += pt_logd_loggamma_mx_st(
            np.vstack((curr_log_zeta[i], prop_log_zeta[i])), 
            alpha[i], beta[i],
            ).ravel()
        if lunifs[i] < logp[1] - logp[0]:
            curr_log_zeta[i] = prop_log_zeta[i]
        else:
            prop_log_zeta[i] = curr_log_zeta[i]
        logp[:] = 0.
    return np.exp(curr_log_zeta)

def update_zeta_j_sph(curr_zeta, n, sY, slY, alpha, beta):
    """ Update routine for zeta on spherical data """
    prop_zeta = np.empty(curr_zeta.shape)
    for l in range(curr_zeta.shape[0]):
        prop_zeta[l] = sample_alpha_1_mh_summary(
            curr_zeta[l], n, sY[l], slY[l], alpha[l], beta[l]
            )
    return prop_zeta

def update_zeta_j_wrapper(args):
    # parse arguments
    curr_zeta_j, nj, sYj, slYj, Ws, alpha, beta, ncol, catmat = args
    prop_zeta_j = np.empty(curr_zeta_j.shape)
    prop_zeta_j[:ncol] = update_zeta_j_sph(
        curr_zeta_j[:ncol], nj, sYj, slYj, alpha[:ncol], beta[:ncol]
        )
    prop_zeta_j[ncol:] = update_zeta_j_cat(
        curr_zeta_j[ncol:], Ws, alpha[ncol:], beta[ncol:], catmat
        )
    return prop_zeta_j

def sample_gamma_shape_wrapper(args):
    # return sample_alpha_k_mh_summary(*args)
    return sample_alpha_1_mh_summary(*args)

def category_matrix(cats):
    catvec = np.hstack(list(np.ones(ncat) * i for i, ncat in enumerate(cats)))
    CatMat = (catvec[:, None] == np.arange(len(cats))).T
    return CatMat

Prior = namedtuple('Prior', 'eta alpha beta')

class Samples(object):
    zeta  = None
    alpha = None
    beta  = None
    delta = None
    eta   = None
    lzhist = None

    def __init__(self, nSamp, nDat, nCol, nCat, nTemp):
        """
        nCol: number of 
        nCat: number of categorical columns
        nCats: number of categorical variables        
        """
        self.zeta  = [None] * (nSamp + 1)
        self.alpha = np.empty((nSamp + 1, nTemp, nCol + nCat))
        self.beta  = np.empty((nSamp + 1, nTemp, nCol + nCat))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.eta   = np.empty((nSamp + 1, nTemp))
        self.lzhist = np.empty((nSamp + 1, nDat, nTemp, nCol + nCat))
        return

def Samples_(object):
    zeta  = None
    alpha = None
    beta  = None
    delta = None
    eta   = None

    def __init__(self, nSamp, nDat, nCol, nCat):
        self.zeta = [None] * (nSamp)
        self.alpha = np.empty((nSamp, nCol + nCat))
        self.beta  = np.empty((nSamp, nCol + nCat))
        self.delta = np.empty((nSamp, nDat))
        self.eta   = np.empty((nSamp))
        return

class Chain(DirichletProcessSampler):
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter]
    @property
    def curr_beta(self):
        return self.samples.beta[self.curr_iter]
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
    am_cov_i  = None
    am_mean_c = None
    am_mean_i = None
    am_n_c    = None
    max_clust_count = None
    swap_attempts = None
    swap_succeeds = None

    def sample_delta(self, delta, zeta, eta):
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_cluster_state = (curr_cluster_state == 0)
        log_likelihood = self.log_delta_likelihood(zeta)
        # log_likelihood = np.zeros((self.nDat, self.nTemp, self.max_clust_count))
        # log_likelihood += pt_logd_projgamma_my_mt(self.data.Yp, zeta[:,:,:self.nCol], self.sigma_ph1)
        # log_likelihood += pt_logd_cumdirmultinom_mx_ma(self.data.W, zeta[:,:,self.nCol:], self.CatMat)
        tidx = np.arange(self.nTemp)
        p = uniform(size = (self.nDat, self.nTemp))
        p += tidx[None,:]
        scratch = np.empty(curr_cluster_state.shape)
        for i in range(self.nDat):
            curr_cluster_state[tidx, delta.T[i]] -= 1
            scratch[:] = 0
            scratch += curr_cluster_state
            scratch += cand_cluster_state * (eta / (cand_cluster_state.sum(axis = 1) + 1e-9))[:,None]
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                np.log(scratch, out = scratch)
            scratch += log_likelihood[i]
            np.nan_to_num(scratch, False, -np.inf)
            scratch -= scratch.max(axis = 1)[:,None]
            with np.errstate(under = 'ignore'):
                np.exp(scratch, out = scratch)
            np.cumsum(scratch, axis = 1, out = scratch)
            scratch /= scratch.T[-1][:,None]
            scratch += tidx[:,None]
            delta.T[i] = np.searchsorted(scratch.ravel(), p[i]) % self.max_clust_count
            curr_cluster_state[tidx, delta.T[i]] += 1
            cand_cluster_state[tidx, delta.T[i]] = False
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

    def sample_zeta_new(self, alpha, beta):
        out = gamma(
            shape = alpha[:,None,:], 
            scale = 1 / beta[:,None,:], 
            size = (self.nTemp, self.max_clust_count, self.tCol),
            )
        return out

    def am_covariance_matrices(self, delta, index):
        cluster_covariance_mat(
            self.am_cov_c, self.am_mean_c, self.am_n_c, delta,
            self.am_cov_i, self.am_mean_i, self.curr_iter, np.arange(self.nTemp),
            )
        return self.am_cov_c[index]

    def sample_zeta(self, zeta, delta, alpha, beta):
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
            cholesky(covs), 
            normal(size = (idx[0].shape[0], self.tCol), scale = 0.3),
            )
        zcand = np.exp(lzcand)
        # am_alpha += np.einsum('ntj,tnj->tj', self.log_likelihood(zcand), delta_ind_mat)
        # am_alpha -= np.einsum('ntj,tnj->tj', self.log_likelihood(zcurr), delta_ind_mat)
        am_alpha += self.log_zeta_likelihood(zcand, delta, delta_ind_mat)
        am_alpha -= self.log_zeta_likelihood(zcurr, delta, delta_ind_mat)
        with np.errstate(invalid = 'ignore'):
            am_alpha *= self.itl[:,None]
        # am_alpha[idx] += logd_loggamma_paired(lzcand[idx], alpha[idx[0]], beta[idx[0]])
        # am_alpha[idx] -= logd_loggamma_paired(lzcurr[idx], alpha[idx[0]], beta[idx[0]])
        am_alpha[idx] += self.log_logzeta_prior(lzcand, idx, alpha, beta)
        am_alpha[idx] -= self.log_logzeta_prior(lzcurr, idx, alpha, beta)
        keep = np.where(np.log(uniform(size = am_alpha.shape)) < am_alpha)
        zcurr[keep] = zcand[keep]
        return zcurr

    def sample_alpha(self, zeta, curr_alpha, extant_clusters):
        """
        Args:
            zeta       : (t,j,d)
            curr_alpha : (t,d)
            extant_clusters : (t,j)
        Returns:
            new_alpha  : (t,d)
        """
        # assert np.all(zeta[extant_clusters] > 0)
        # with np.errstate(divide = 'ignore', invalid = 'ignore'):
        #     sz = np.nansum(zeta * extant_clusters[:,:,None], axis = 1) # (t,d)
        #     slz = np.nansum(np.log(zeta) * extant_clusters[:,:,None], axis = 1) # (t,d)
        # n = np.ones((self.nTemp, self.tCol)) # (t,d)
        # n *= extant_clusters.sum(axis = 1)[:,None]       # (t,d)
        # args = zip(
        #     curr_alpha.ravel(), n.ravel(), sz.ravel(), slz.ravel(),   # (t x d)
        #     repeat(self.priors.alpha.a), repeat(self.priors.alpha.b),
        #     # repeat(self.priors.beta.a), repeat(self.priors.beta.b),
        #     )
        # # res = map(sample_gamma_shape_wrapper, args)
        # res = self.pool.map(sample_gamma_shape_wrapper, args)
        # return np.array(list(res)).reshape(curr_alpha.shape) # (t,d)
        return np.ones(curr_alpha.shape)

    def sample_beta(self, zeta, alpha, extant_clusters):
        # """
        # Args:
        #     zeta  : (t,j,d)
        #     alpha : (t,d)
        #     extant_clusters : (t,j)
        # Returns:
        #     beta  : (t,d)
        # """
        # n = np.ones((self.nTemp, self.tCol))
        # n *= extant_clusters.sum(axis = 1)[:,None]                 # (t,d)
        # zs = np.nansum(zeta * extant_clusters[:,:,None], axis = 1) # (t,d)
        # As = n * alpha + self.priors.beta.a
        # Bs = zs + self.priors.beta.b
        # return gamma(shape = As, scale = 1 / Bs)
        return np.ones(alpha.shape)

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
        out += pt_logd_projgamma_my_mt(self.data.Yp, zeta[:,:,:self.nCol], self.sigma_ph1)
        out += pt_logd_cumdirmultinom_mx_ma(self.data.W, zeta[:,:,self.nCol:], self.CatMat)
        return out
    
    def log_zeta_likelihood(self, zeta, delta, dmat):
        zs = zeta[self.temp_unravel, delta.ravel()].reshape(self.zeta_shape)
        out = np.zeros((self.nTemp, self.max_clust_count))
        out += np.einsum(
            'tn,tnj->tj',
            pt_logd_cumdirmultinom_paired_yt(
                self.data.W, zs[:,:,self.nCol:], self.CatMat,
                ),
            dmat,
            )
        out += np.einsum(
            'tn,tnj->tj',
            pt_logd_projgamma_paired_yt(
                self.data.Yp, zs[:,:,:self.nCol], self.sigma_ph2,
                ),
            dmat,
            )
        return out
    
    def log_logzeta_prior(self, lzeta, idx,  alpha, beta):
        out = np.zeros(idx[0].shape[0]) # (m)
        out += logd_loggamma_paired(lzeta[idx], alpha[idx[0]], beta[idx[0]])
        return out
    
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
        extant_clusters = (bincount2D_vectorized(self.curr_delta, self.max_clust_count) > 0)
        with np.errstate(invalid = 'ignore'):
            out += np.nansum(
                extant_clusters * pt_logd_prodgamma_my_st(
                    self.curr_zeta, self.curr_alpha, self.curr_beta,
                    ),
                axis = 1,
                )
        out += logd_gamma_my(self.curr_alpha, *self.priors.alpha).sum(axis = 1)
        out += logd_gamma_my(self.curr_beta, *self.priors.beta).sum(axis = 1)
        out += logd_gamma_my(self.curr_eta, *self.priors.eta)
        return out

    def initialize_sampler(self, ns):
        # Samples
        self.samples = Samples(ns, self.nDat, self.nCol, self.nCat, self.nTemp)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
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
        self.am_cov_c     = np.empty((self.nTemp, self.max_clust_count, 
                                                        self.tCol, self.tCol))
        self.am_mean_c    = np.empty((self.nTemp, self.max_clust_count, self.tCol))
        self.am_n_c       = np.zeros((self.nTemp, self.max_clust_count))
        self.am_alpha     = np.zeros((self.nTemp, self.max_clust_count))
        self.samples.lzhist[0] = np.swapaxes(
            np.log(self.samples.zeta[0][
                self.temp_unravel, self.samples.delta[0].ravel()
                ].reshape(self.nTemp, self.nDat, self.tCol)), 
            0, 1,
            )
        self.swap_attempts = np.zeros((self.nTemp, self.nTemp))
        self.swap_succeeds = np.zeros((self.nTemp, self.nTemp))
        # PlaceHolders
        self.sigma_ph1 = np.ones((self.nTemp, self.max_clust_count, self.nCol))
        self.sigma_ph2 = np.ones((self.nTemp, self.nDat, self.nCol))
        self.zeta_shape = (self.nTemp, self.nDat, self.tCol)
        return

    def update_am_cov_initial(self):
        """ Initial update for Adaptive Metropolis Covariance per obsv."""
        self.am_mean_i[:] = self.samples.lzhist[:self.curr_iter].mean(axis = 0)
        self.am_cov_i[:] = 1 / self.curr_iter * np.einsum(
            'intj,intk->ntjk',
            self.samples.lzhist[:self.curr_iter] - self.am_mean_i,
            self.samples.lzhist[:self.curr_iter] - self.am_mean_i,
            )
        return
    
    def update_am_cov(self):
        """ Online updating for Adaptive Metropolis Covariance per obsv. """
        c = self.curr_iter 
        c1 = self.curr_iter + 1
        self.am_mean_i += (
            (self.samples.lzhist[self.curr_iter] - self.am_mean_i) / c
            )
        self.am_cov_i[:] = (
            + (c/c1) * self.am_cov_i
            + (c/c1/c1) * np.einsum(
                'tej,tel->tejl',
                self.samples.lzhist[self.curr_iter] - self.am_mean_i,
                self.samples.lzhist[self.curr_iter] - self.am_mean_i,
                )
            )
        return

    def try_tempering_swap(self):
        ci = self.curr_iter
        # declare log-likelihood, log-prior
        lpl = self.log_tempering_likelihood()
        lpp = self.log_tempering_prior()
        # declare swap attempts
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
        # calculate swaps
        logp = np.log(uniform(size = sw_alpha.shape))
        for tt in sw[np.where(logp < sw_alpha)[0]]:
            # report successful swap
            self.swap_succeeds[tt[0],tt[1]] += 1
            self.swap_succeeds[tt[1],tt[0]] += 1
            # do the swap
            self.samples.zeta[ci][tt[0]], self.samples.zeta[ci][tt[1]] = \
                self.samples.zeta[ci][tt[1]].copy(), self.samples.zeta[ci][tt[0]].copy()
            self.samples.alpha[ci][tt[0]], self.samples.alpha[ci][tt[1]] = \
                self.samples.alpha[ci][tt[1]].copy(), self.samples.alpha[ci][tt[0]].copy()
            self.samples.beta[ci][tt[0]], self.samples.beta[ci][tt[1]] = \
                self.samples.beta[ci][tt[1]].copy(), self.samples.beta[ci][tt[0]].copy()
            self.samples.delta[ci][tt[0]], self.samples.delta[ci][tt[1]] = \
                self.samples.delta[ci][tt[1]].copy(), self.samples.delta[ci][tt[0]].copy()
            self.samples.eta[ci][tt[0]], self.samples.eta[ci][tt[1]] = \
                self.samples.eta[ci][tt[1]].copy(), self.samples.eta[ci][tt[0]].copy()
        return

    def update_logzeta_historical(self):
        self.samples.lzhist[self.curr_iter] = np.swapaxes(
            np.log(self.curr_zeta[self.temp_unravel, 
                    self.curr_delta.ravel(),].reshape(self.zeta_shape)
                ),
            0, 1,
            )
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy()
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = self.curr_zeta.copy()
        eta   = self.curr_eta

        if self.curr_iter > 300:
            self.update_am_cov()
        elif self.curr_iter == 300:
            self.update_am_cov_initial()
        else:
            pass
        # Advance the iterator
        self.curr_iter += 1
        ci = self.curr_iter

        # Sample new candidate clusters
        cand_clusters = np.where(bincount2D_vectorized(delta, self.max_clust_count) == 0)
        zeta[cand_clusters] = self.sample_zeta_new(alpha, beta)[cand_clusters]
        
        # Update cluster assignments and re-index
        self.sample_delta(delta, zeta, eta)
        self.clean_delta_zeta(delta, zeta)
        self.samples.delta[ci] = delta
        
        # do rest of sampling
        extant_clusters = bincount2D_vectorized(self.curr_delta, self.max_clust_count) > 0
        self.samples.zeta[ci] = self.sample_zeta(
            zeta, self.curr_delta, alpha, beta,
            )
        self.samples.alpha[ci] = self.sample_alpha(
            self.curr_zeta, alpha, extant_clusters
            )
        self.samples.beta[ci]  = self.sample_beta(
            self.curr_zeta, self.curr_alpha, extant_clusters,
            )
        self.samples.eta[ci] = self.sample_eta(eta, self.curr_delta)

        # Attempt Swap:
        if self.curr_iter >= self.swap_start:
            self.try_tempering_swap()
        
        self.update_logzeta_historical()
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
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        alphas = self.samples.alpha[nBurn :: nThin, 0]
        betas  = self.samples.beta[nBurn :: nThin, 0]
        deltas = self.samples.delta[nBurn :: nThin, 0]
        etas   = self.samples.eta[nBurn :: nThin, 0]
        # make output dictionary
        out = {
            'zetas'  : zetas,
            'alphas' : alphas,
            'betas'  : betas,
            'deltas' : deltas,
            'etas'   : etas,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'nCat'   : self.nCat,
            'cats'   : self.data.Cats,
            'V'      : self.data.V,
            'W'      : self.data.W,
            }
        # try to add outcome to dictionary
        try:
            out['Y'] = self.data.Y
        except AttributeError:
            pass
        # write to disk
        with open(path, 'wb') as file:
            pickle.dump(out, file)
        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        self.data.Yp[self.data.Yp <= 1e-6] = 1e-6
        return
    
    def categorical_considerations(self):
        """ Builds the CatMat """
        self.CatMat = category_matrix(self.data.Cats)
        return
    
    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_alpha = GammaPrior(1., 1.),
            prior_beta  = GammaPrior(1., 1.),
            p           = 10,
            max_clust_count = 300,
            ntemps = 5,
            stepping = 1.05,
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
        self.priors = Prior(prior_eta, prior_alpha, prior_beta)
        self.set_projection()
        self.categorical_considerations()
        # self.pool = Pool(processes = 8, initializer = limit_cpu())

        self.nTemp = ntemps
        self.itl = 1 / stepping**np.arange(ntemps)
        self.temp_unravel = np.repeat(np.arange(self.nTemp), self.nDat)
        self.nSwap_per = self.nTemp // 2
        self.swap_start = 100
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
                size = (m, self.tCol),
                )
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
        indices = list(np.arange(self.tCol))
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

    def generate_posterior_predictive_spheres(self):
        rhos = self.generate_posterior_predictive_gammas() # (s,D)
        CatMat = category_matrix(self.data.Cats) # (C,d)
        shro = rhos[:,self.nCol:] @ CatMat.T # (s,C)
        nrho = np.einsum('sc,cd->sd', shro, CatMat) # (s,d)
        pis = rhos / nrho
        return pis

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
        zetas = np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]) # (s,n,d)
        W = np.hstack((np.zeros((self.nDat, self.nCol)), self.data.W)) # (n,d)
        return gamma(shape = zetas + W[None,:,:])

    def generate_conditional_posterior_predictive_spheres(self):
        """ pi | zeta, delta = normalized rho
        currently discarding generated Y's, keeping latent pis
        """
        rhos = self.generate_conditional_posterior_predictive_gammas() # (s,n,D)
        CatMat = category_matrix(self.data.Cats) # (C,d)
        shro = rhos[:,:,self.nCol:] @ CatMat.T # (s,n,C)
        nrho = np.einsum('snc,cd->snd', shro, CatMat) # (s,n,d)
        pis = rhos / nrho
        return pis

    def load_data(self, path):        
        with open(path, 'rb') as file:
            out = pickle.load(file)
        
        deltas = out['deltas']
        etas   = out['etas']
        zetas  = out['zetas']
        alphas = out['alphas']
        betas  = out['betas']
        cats   = out['cats']
        
        self.data = MixedDataBase(out['V'], out['W'], out['cats'])
        self.nSamp  = deltas.shape[0]
        self.nDat   = deltas.shape[1]
        self.nCat   = self.data.nCat
        self.nCol   = self.data.nCol
        self.tCol   = self.nCol + self.nCat
        self.nCats  = cats.shape[0]
        self.cats   = cats
        
        if 'Y' in out.keys():
            self.data.fill_outcome(out['Y'])
        
        self.samples       = Samples(self.nSamp, self.nDat, self.nCol, self.nCat, self.nCats)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
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

    p = argparser()
    # d = {
    #     'in_data_path'    : './ad/mammography/data.csv',
    #     'in_outcome_path' : './ad/mammography/outcome.csv',
    #     'out_path' : './ad/mammography/results_mdppprg_pt.pkl',
    #     'cat_vars' : '[5,6,7,8]',
    #     'decluster' : 'False',
    #     'quantile' : 0.95,
    #     'nSamp' : 3000,
    #     'nKeep' : 2000,
    #     'nThin' : 1,
    #     'eta_alpha' : 2.,
    #     'eta_beta' : 1.,
    #     }
    # p = Heap(**d)

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
