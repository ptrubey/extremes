"""
Model description for Dirichlet-process Mixture of Projected Gammas on unit p-sphere
---
PG is unrestricted (allow betas to vary)
Centering distribution is product of Gammas
"""
from numpy.random import choice, gamma, beta, uniform, normal
from collections import namedtuple
from itertools import repeat
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log
from scipy.special import gammaln
from io import BytesIO

from cUtility import diriproc_cluster_sampler, generate_indices
from samplers import DirichletProcessSampler, bincount2D_vectorized,            \
    pt_dp_sample_cluster_crp8
from cProjgamma import sample_alpha_k_mh_summary, sample_alpha_1_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube, Data_From_Sphere
from projgamma import GammaPrior, logd_prodgamma_my_mt, logd_prodgamma_my_st,   \
    logd_prodgamma_paired, logd_gamma, logd_gamma_my, pt_logd_gamma_my,         \
    pt_logd_projgamma_my_mt_inplace_unstable, pt_logd_projgamma_paired_yt,      \
    pt_logpost_loggammagamma

def log_density_log_zeta_j_inplace(
        logdensity, 
        log_zeta_j, 
        n, 
        log_y_sv,
        y_sv, 
        alpha, 
        beta, 
        xi, 
        tau,
        ):
    """
    logdensity : (t, j, d) <- Target
    log_zeta_j : (t, j, d)
    log_y_sv   : (t, j, d)
    y_sv       : (t, j, d)
    n          : (t, j)
    alpha      : (t, d)
    beta       : (t, d)
    xi         : (t, d - 1)
    tau        : (t, d - 1)
    """
    zeta_j = np.exp(log_zeta_j)
    logdensity += (zeta_j - 1) * log_y_sv
    logdensity += alpha[:,None] * log_zeta_j
    logdensity -= beta[:,None] * zeta_j
    logdensity[:,:,0]  -= n * gammaln(zeta_j[:,:,0])
    logdensity[:,:,1:] += gammaln(n[:,:,None] * alpha[:,None] + xi)
    logdensity[:,:,1:] -= \
        (n[:,:,None] * alpha[:,None] + xi[:,None]) * np.log(y_sv[:,:,1:] + tau[:,None])
    return

def log_density_log_zeta_j(log_zeta_j, n, log_y_sv, y_sv, alpha, beta, xi, tau):
    ld = np.zeros(log_zeta_j.shape)
    log_density_log_zeta_j_inplace(
        ld, log_zeta_j, n, log_y_sv, y_sv, alpha, beta, xi, tau,
        )
    return ld

def log_density_log_zeta_tj(log_zeta_j, n, log_y_sv, y_sv, alpha, beta, xi, tau):
    """
    log_zeta_j : (m, d)
    log_y_sv   : (m, d)
    y_sv       : (m, d)
    n          : (m)
    alpha      : (m, d)
    beta       : (m, d)
    xi         : (m, d - 1)
    tau        : (m, d - 1)
    """
    logdensity = np.zeros(log_zeta_j.shape)
    zeta_j     = np.exp(log_zeta_j)
    logdensity += (zeta_j - 1) * log_y_sv
    logdensity += alpha * log_zeta_j
    logdensity -= beta * zeta_j
    logdensity[:,0] -= n * gammaln(zeta_j[:,0])
    logdensity[:,1:] += gammaln(n[:,None] * alpha[:,1:] + xi)
    logdensity[:,1:] -= (n[:,None] * alpha[:,1:] + xi) * np.log(y_sv[:,1:] + tau)
    return logdensity

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
    ld    = None

    def __init__(self, nSamp, nDat, nCol, nTemp, nClustMax):
        self.zeta  = np.empty((nSamp + 1, nTemp, nClustMax, nCol))
        self.sigma = np.empty((nSamp + 1, nTemp, nClustMax, nCol))
        self.sigma[:,:,:,0] = 1.
        self.alpha = np.empty((nSamp + 1, nTemp, nCol))
        self.beta  = np.empty((nSamp + 1, nTemp, nCol))
        self.xi    = np.empty((nSamp + 1, nTemp, nCol - 1))
        self.tau   = np.empty((nSamp + 1, nTemp, nCol - 1))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nTemp, nDat))
        self.eta   = np.empty((nSamp + 1, nTemp))
        self.ld    = np.empty((nSamp + 1)) # log-density of cold chain
        return

class Samples_(Samples): # same as samples, but with 1 temperature
    def __init__(self, nSamp, nDat, nCol, nClustMax):
        self.zeta = np.empty((nSamp, nClustMax, nCol))
        self.sigma = np.empty((nSamp, nClustMax, nCol))
        self.alpha = np.empty((nSamp, nCol))
        self.beta  = np.empty((nSamp, nCol))
        self.xi    = np.empty((nSamp, nCol - 1))
        self.tau   = np.empty((nSamp, nCol - 1))
        self.delta = np.empty((nSamp, nDat))
        self.r     = np.empty((nSamp, nDat))
        self.eta   = np.empty((nSamp))
        self.ld    = np.empty((nSamp))
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
    
    max_clust_count = None
    swap_attempts = None
    swap_succeeds = None
    
    def log_delta_likelihood(self, zeta, sigma):
        """
        inputs:
            zeta  : (t, j, d)
            sigma : (t, j, d)
        outputs:
            out   : (t, j)
        """
        out = np.zeros((self.nDat, self.nTemp, self.max_clust_count))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pt_logd_projgamma_my_mt_inplace_unstable(
                out, self.data.Yp, zeta, sigma,
                )
        np.nan_to_num(out, False, -np.inf)
        # out *= self.itl[:,None]
        return out

    def sample_delta(self, delta, zeta, sigma, eta):
        log_likelihood = self.log_delta_likelihood(zeta, sigma)
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_cluster_state = (curr_cluster_state == 0)
        # log_likelihood *= self.itl[None,:,None]
        tidx = np.arange(self.nTemp)
        p = uniform(size = (self.nDat, self.nTemp))
        p += tidx.reshape(1, -1)
        scratch = np.empty(curr_cluster_state.shape)
        for i in range(self.nDat):
            curr_cluster_state[tidx, delta.T[i]] -= 1
            scratch[:] = 0.
            scratch += curr_cluster_state
            scratch += cand_cluster_state * \
                (eta / (cand_cluster_state.sum(axis = 1) + 1e-9))[:,None]
            with np.errstate(divide = 'ignore', invalid = 'ignore', under = 'ignore'):
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
        return delta
    
    def sample_zeta(self, zeta, delta, r, alpha, beta, xi, tau):
        Y = r[:,:,None] * self.data.Yp[None] # (t, n, 1) * (1, n, d) -> (t,n,d)
        lY = np.log(Y)

        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        cand_cluster_state = (curr_cluster_state == 0)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)

        idx = np.where(~cand_cluster_state)
        nidx = np.where(cand_cluster_state)
        
        lz_curr = np.log(zeta)
        lz_cand = lz_curr.copy()
        lz_cand[idx] += normal(
            scale = 0.3**(self.itl[idx[0]])[:,None], 
            size = (idx[0].shape[0], self.nCol),
            )
        Ysv  = np.einsum('tnd,tnj->tjd', Y, delta_ind_mat)[idx]
        lYsv = np.einsum('tnd,tnj->tjd', lY, delta_ind_mat)[idx]
        nj   = curr_cluster_state[idx]

        logalpha = np.empty(lz_curr.shape)
        logalpha[:] = -np.inf
        logalpha[idx] = self.itl[idx[0], None] * (
            + log_density_log_zeta_tj(
                lz_cand[idx], nj, lYsv, Ysv, 
                alpha[idx[0]], beta[idx[0]], 
                xi[idx[0]], tau[idx[0]],
                )
            - log_density_log_zeta_tj(
                lz_curr[idx], nj, lYsv, Ysv, 
                alpha[idx[0]], beta[idx[0]], 
                xi[idx[0]], tau[idx[0]],
                )
            )
        keep = np.where(np.log(uniform(size = logalpha.shape)) < logalpha)
        zeta[keep] = np.exp(lz_cand[keep])
        zeta[nidx] = gamma(shape = alpha[nidx[0]], scale = 1 / beta[nidx[0]])
        return zeta

    def sample_sigma(self, zeta, delta, r, xi, tau):
        """
        zeta  : (t x J x d)
        delta : (t x n)
        r     : (t x n)
        xi    : (t x d-1)
        tau   : (t x d-1)
        """
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)
        Ysv = np.einsum('tnd,tnj->tjd', r[:,:,None] * self.data.Yp[None,:,:], delta_ind_mat)

        shape = np.zeros((self.nTemp, self.max_clust_count, self.nCol - 1))
        rate  = np.zeros((self.nTemp, self.max_clust_count, self.nCol - 1))
        
        shape += xi[:,None,:]
        rate  += tau[:,None,:]
        
        shape += curr_cluster_state[:,:,None] * zeta[:,:,1:]
        rate  += Ysv[:,:,1:]
        rate  *= self.itl[:,None,None]
        
        shape -= 1
        shape *= self.itl[:,None,None]
        shape += 1

        sigma_new = np.ones(zeta.shape)
        sigma_new[:,:,1:] = gamma(shape = shape, scale = 1 / rate)
        sigma_new[sigma_new < 1e-12] = 1e-12
        return sigma_new

    def clean_delta_zeta_sigma(self, delta, zeta, sigma):
        """
        delta : cluster indicator vector (n)
        zeta  : cluster parameter matrix (J* x d)
        sigma : cluster parameter matrix (J* x d)
        """
        for t in range(self.nTemp):
            keep, delta[t] = np.unique(delta[t], return_inverse = True)
            zeta[t][:keep.shape[0]] = zeta[t][keep]
            sigma[t][:keep.shape[0]] = sigma[t][keep]
        return

    def sample_alpha(self, curr_alpha, zeta, extant_clusters):
        # n = np.repeat(extant_clusters.sum(axis = 1), self.nCol)
        n = extant_clusters.sum(axis = 1)
        assert np.all(zeta[extant_clusters] > 0.)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            ls = np.nansum(np.log(zeta) * extant_clusters[:,:,None], axis = 1)  # (t,d)
            s  = np.einsum('tjd,tj->td', zeta, extant_clusters)                 # (t,d)
        
        a_curr = curr_alpha.copy()
        l_curr = np.log(a_curr)
        l_cand = l_curr + normal(scale = 0.3, size = l_curr.shape)

        logalpha = np.zeros(a_curr.shape)
        logalpha += pt_logpost_loggammagamma(
            l_cand, n, s, ls, *self.priors.alpha, *self.priors.beta,
            )
        logalpha -= pt_logpost_loggammagamma(
            l_curr, n, s, ls, *self.priors.alpha, *self.priors.beta,
            )
        logalpha *= self.itl[:,None]
        
        keep = np.where(np.log(uniform(size = logalpha.shape)) < logalpha)
        a_curr[keep] = np.exp(l_cand[keep])
        return a_curr
    
    def sample_beta(self, zeta, alpha, extant_clusters):
        n = extant_clusters.sum(axis = 1)                       # (t)
        s = (zeta * extant_clusters[:,:,None]).sum(axis = 1)    # (t, d)
        shape = n[:,None] * alpha + self.priors.beta.a          # (t, d)
        shape -= 1
        shape *= self.itl[:,None]
        shape += 1
        rate  = s + self.priors.beta.b                          # (t, d)
        rate  *= self.itl[:,None]
        return gamma(shape = shape, scale = 1 / rate)           # (t, d)
    
    def sample_xi(self, curr_xi, sigma, extant_clusters):
        n = extant_clusters.sum(axis = 1)
        assert np.all(sigma[extant_clusters] > 0.)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            ls = np.nansum(np.log(sigma[:,:,1:]) * extant_clusters[:,:,None], axis = 1)  # (t,d)
            s  = np.einsum('tjd,tj->td', sigma[:,:,1:], extant_clusters)                 # (t,d)
        
        x_curr = curr_xi.copy()
        l_curr = np.log(x_curr)
        l_cand = l_curr + normal(scale = 0.3, size = l_curr.shape)

        logalpha = np.zeros(x_curr.shape)
        logalpha += pt_logpost_loggammagamma(
            l_cand, n, s, ls, *self.priors.alpha, *self.priors.beta,
            )
        logalpha -= pt_logpost_loggammagamma(
            l_curr, n, s, ls, *self.priors.alpha, *self.priors.beta,
            )
        logalpha *= self.itl[:,None]
        
        keep = np.where(np.log(uniform(size = logalpha.shape)) < logalpha)
        x_curr[keep] = np.exp(l_cand[keep])
        return x_curr

    def sample_tau(self, sigma, xi, extant_clusters):
        # return np.ones(xi.shape)
        n = extant_clusters.sum(axis = 1) # .reshape(-1, 1)
        s = (sigma[:,:,1:] * extant_clusters[:,:,None]).sum(axis = 1) # (t, d-1)
        shape = n[:, None] * xi + self.priors.tau.a # (t, d-1)
        shape -= 1
        shape *= self.itl[:,None]
        shape += 1
        rate  = s + self.priors.tau.b
        rate  *= self.itl[:,None]               # (t, d-1)
        return gamma(shape = shape, scale = 1 / rate)  # (t, d-1)

    def sample_r(self, delta, zeta, sigma):
        # As = np.einsum('il->i', zeta[delta])
        Zs = zeta.sum(axis = 2)
        As = Zs[self.temp_unravel, delta.ravel()].reshape(self.nTemp, self.nDat)
        # Bs = np.einsum('il,il->i', self.data.Yp, sigma[delta])
        Bs = np.einsum(
            'nd,tnd->tn', 
            self.data.Yp,
            sigma[
                self.temp_unravel, delta.ravel()
                ].reshape(self.nTemp, self.nDat, self.nCol),
            )
        return gamma(shape = As, scale = 1 / Bs)
    
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

    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol, self.nTemp, self.max_clust_count)
        self.samples.alpha[0] = 1.
        self.samples.beta[0] = 1.
        self.samples.xi[0] = 1.
        self.samples.tau[0] = 1.
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = (self.nTemp, self.max_clust_count, self.nCol))
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = (self.nTemp, self.max_clust_count, self.nCol))
        self.samples.eta[0] = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 30, size = (self.nTemp, self.nDat))
        # self.samples.delta[0][-1] = np.arange(self.max_clust_count - 30)[-1]
        self.samples.r[0] = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0], self.samples.sigma[0],
                )
        self.curr_iter = 0
        return

    def log_likelihood(self):
        ll = np.zeros(self.nTemp)
        ll += pt_logd_projgamma_paired_yt(
            self.data.Yp, 
            self.curr_zeta[
                self.temp_unravel, self.curr_delta.ravel()
                ].reshape(self.nTemp, self.nDat, self.nCol),
            self.curr_sigma[
                self.temp_unravel, self.curr_delta.ravel()
                ].reshape(self.nTemp, self.nDat, self.nCol),
            ).sum(axis = 1)
        return ll

    def log_prior(self):
        extant_clusters = (bincount2D_vectorized(self.curr_delta, self.max_clust_count) > 0)
        lp = np.zeros(self.nTemp)
        lp += np.einsum(
            'tj,tj->t',
            pt_logd_gamma_my(self.curr_zeta, self.curr_alpha, self.curr_beta),
            extant_clusters,
            )
        lp += np.einsum(
            'tj,tj->t',
            pt_logd_gamma_my(self.curr_sigma[:,:,1:], self.curr_xi, self.curr_tau),
            extant_clusters,
            )
        lp += logd_gamma_my(self.curr_alpha, *self.priors.alpha).sum(axis = 1)
        lp += logd_gamma_my(self.curr_beta, *self.priors.beta).sum(axis = 1)
        lp += logd_gamma_my(self.curr_xi, *self.priors.xi).sum(axis = 1)
        lp += logd_gamma_my(self.curr_tau, *self.priors.tau).sum(axis = 1)
        lp += logd_gamma_my(self.curr_eta, *self.priors.eta)
        return lp

    def record_log_density(self):
        ll = self.log_likelihood()
        lp = self.log_prior()
        self.samples.ld[self.curr_iter] = (ll + lp)[0]
        # lpl = 0.
        # lpp = 0.
        # Y = self.curr_r[:,:,None] * self.data.Yp[None]
        # lpl += logd_prodgamma_paired(
        #     Y,
        #     self.curr_zeta[
        #         self.temp_unravel, self.curr_delta.ravel(),
        #         ].reshape(self.nTemp, self.nDat, self.nCol),
        #     self.curr_sigma[
        #         self.temp_unravel, self.curr_delta.ravel(),
        #         ].reshape(self.nTemp, self.nDat, self.nCol),
        #     ).sum()
        # lpl += logd_prodgamma_my_st(self.curr_zeta, self.curr_alpha, self.curr_beta).sum()
        # lpl += logd_prodgamma_my_st(self.curr_sigma[:,1:], self.curr_xi, self.curr_tau).sum()
        # lpp += logd_gamma(self.curr_alpha, *self.priors.alpha).sum()
        # lpp += logd_gamma(self.curr_beta, *self.priors.beta).sum()
        # lpp += logd_gamma(self.curr_xi, *self.priors.xi).sum()
        # lpp += logd_gamma(self.curr_tau, *self.priors.tau).sum()
        # self.samples.ld[self.curr_iter] = lpl + lpp
        # return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy()
        alpha = self.curr_alpha
        beta  = self.curr_beta
        xi    = self.curr_xi
        tau   = self.curr_tau
        zeta = self.curr_zeta
        sigma = self.curr_sigma
        eta   = self.curr_eta
        r     = self.curr_r

        self.curr_iter += 1
        # Sample new cluster membership indicators 
        delta = self.sample_delta(delta, zeta, sigma, eta)
        # clean indices and re-index
        self.clean_delta_zeta_sigma(delta, zeta, sigma)
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta, sigma)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_delta, self.curr_r, alpha, beta, xi, tau,
                )
        self.samples.sigma[self.curr_iter] = self.sample_sigma(
                zeta, self.curr_delta, self.curr_r, xi, tau,
                )
        extant_clusters = bincount2D_vectorized(delta, self.max_clust_count) > 0
        self.samples.alpha[self.curr_iter] = self.sample_alpha(
            alpha, self.curr_zeta, extant_clusters,
            )
        self.samples.beta[self.curr_iter]  = self.sample_beta(
            self.curr_zeta, self.curr_alpha, extant_clusters,
            )
        self.samples.xi[self.curr_iter]    = self.sample_xi(
            xi, self.curr_sigma, extant_clusters,
            )
        self.samples.tau[self.curr_iter]   = self.sample_tau(
            self.curr_sigma, self.curr_xi, extant_clusters,
            )
        self.samples.eta[self.curr_iter]   = self.sample_eta(eta, self.curr_delta)

        if self.curr_iter > self.swap_start:
            self.try_tempering_swap()

        self.record_log_density()
        return
    
    def write_to_disk(self, path, nBurn, nThin = 1):
        if type(path) is str:
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
            'logd'   : self.samples.ld
            }
        
        # try to add outcome / radius to dictionary
        for attr in ['Y','R']:
            if hasattr(self.data, attr):
                out[attr] = self.data.__dict__[attr]
        
        if type(path) is BytesIO:
            path.write(pickle.dumps(out))
        else:
            with open(path, 'wb') as file:
                pickle.dump(out, file)

        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        return

    def try_tempering_swap(self):
        ci = self.curr_iter
        # declare log-likelihood, log-prior
        lp = self.log_likelihood() + self.log_prior()
        # declare swap choices
        sw = choice(self.nTemp, 2 * self.nSwap_per, replace = False).reshape(-1,2)
        for s in sw:
            # record attempted swap
            self.swap_attempts[s[0],s[1]] += 1
            self.swap_attempts[s[1],s[0]] += 1
        # compute swap log-probability
        sw_alpha = np.zeros(sw.shape[0])
        sw_alpha += lp[sw.T[1]] - lp[sw.T[0]]
        sw_alpha *= self.itl[sw.T[1]] - self.itl[sw.T[0]]
        logp = np.log(uniform(size = sw_alpha.shape))
        for tt in sw[np.where(logp < sw_alpha)[0]]:
            # report successful swap
            self.swap_succeeds[tt[0],tt[1]] += 1
            self.swap_succeeds[tt[1],tt[0]] += 1
            self.samples.zeta[ci][tt[0]], self.samples.zeta[ci][tt[1]] = \
                self.samples.zeta[ci][tt[1]].copy(), self.samples.zeta[ci][tt[0]].copy()
            self.samples.sigma[ci][tt[0]], self.samples.sigma[ci][tt[1]] = \
                self.samples.sigma[ci][tt[1]].copy(), self.samples.sigma[ci][tt[0]].copy()
            self.samples.alpha[ci][tt[0]], self.samples.alpha[ci][tt[1]] = \
                self.samples.alpha[ci][tt[1]].copy(), self.samples.alpha[ci][tt[0]].copy()
            self.samples.beta[ci][tt[0]], self.samples.beta[ci][tt[1]] = \
                self.samples.beta[ci][tt[1]].copy(), self.samples.beta[ci][tt[0]].copy()
            self.samples.xi[ci][tt[0]], self.samples.xi[ci][tt[1]] = \
                self.samples.xi[ci][tt[1]].copy(), self.samples.xi[ci][tt[0]].copy()
            self.samples.tau[ci][tt[0]], self.samples.tau[ci][tt[1]] = \
                self.samples.tau[ci][tt[1]].copy(), self.samples.tau[ci][tt[0]].copy()
            self.samples.eta[ci][tt[0]], self.samples.eta[ci][tt[1]] = \
                self.samples.eta[ci][tt[1]].copy(), self.samples.eta[ci][tt[0]].copy()
            self.samples.delta[ci][tt[0]], self.samples.delta[ci][tt[1]] = \
                self.samples.delta[ci][tt[1]].copy(), self.samples.delta[ci][tt[0]].copy()
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
            ntemps      = 5,
            stepping    = 1.15,
            **kwargs
            ):
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        _prior_eta = GammaPrior(*prior_eta)
        _prior_alpha = GammaPrior(*prior_alpha)
        _prior_beta = GammaPrior(*prior_beta)
        _prior_xi  = GammaPrior(*prior_xi)
        _prior_tau = GammaPrior(*prior_tau)
        self.priors = Prior(
            _prior_eta, _prior_alpha, _prior_beta, _prior_xi, _prior_tau,
            )
        self.set_projection()
        self.itl = 1 / stepping**np.arange(ntemps)
        self.nTemp = ntemps
        self.nSwap_per = self.nTemp // 2
        self.swap_start = 100
        self.temp_unravel = np.repeat(np.arange(self.nTemp), self.nDat)
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
            deltas = generate_indices(prob, n_per_sample)
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
        if type(path) is BytesIO:
            out = pickle.loads(path.getvalue())
        else:
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

        self.data = Data_From_Sphere(out['V'])
        try:
            self.data.fill_outcome(out['Y'])
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
        self.samples.ld    = out['logd']
        return

    def __init__(self, path):
        self.load_data(path)
        return

if __name__ == '__main__':
    pass

    from data import Data_From_Raw
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    raw = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(10000)
    model.write_to_disk('./test/results.pkl', 5000, 10)
    res = Result('./test/results.pkl')
    res.write_posterior_predictive('./test/postpred.csv')

# EOF
