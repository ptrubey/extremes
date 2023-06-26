import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
from numpy.random import choice, gamma, beta, normal, uniform
from numpy.linalg import cholesky, slogdet, inv
from scipy.stats import invwishart
from scipy.special import gammaln, multigammaln
from collections import namedtuple
from itertools import repeat
import pandas as pd
import os
import pickle
from math import log
from io import BytesIO

from samplers import DirichletProcessSampler, bincount2D_vectorized
from cov import PerObsTemperedOnlineCovariance
from cUtility import generate_indices
from cProjgamma import sample_alpha_k_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube, Data_From_Sphere
from projgamma import GammaPrior, pt_logd_prodgamma_my_mt, pt_logd_prodgamma_paired, \
        pt_logd_prodgamma_my_st, pt_logd_mvnormal_mx_st, logd_mvnormal_mx_st,\
        logd_gamma_my, logd_invwishart_ms

NormalPrior     = namedtuple('NormalPrior', 'mu SCho SInv')
InvWishartPrior = namedtuple('InvWishartPrior', 'nu psi')
Prior = namedtuple('Prior', 'eta mu Sigma xi tau')

# Wrappers

def sample_xi_wrapper(args):
    return sample_alpha_k_mh_summary(*args)

# Utility functions

# Log Densities

# def dprodgamma_log_paired_yt(aY, aAlpha, aBeta):
#     """
#     product of gammas log-density for paired y, theta
#     ----
#     aY     : array of Y     (t x n x d) [Y in R^d]
#     aAlpha : array of alpha (t x n x d)
#     aBeta  : array of beta  (t x n x d)
#     ----
#     returns: (t x n)
#     """
#     ld = np.zeros(aY.shape[:-1])                             # n temps x n Y
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         ld += np.einsum('tnd,tnd->tn', aAlpha, np.log(aBeta))    # beta^alpha
#         ld -= np.einsum('tnd->tn', gammaln(aAlpha))              # gamma(alpha)
#         ld += np.einsum('tnd,tnd->tn', np.log(aY), (aAlpha - 1)) # y^(alpha - 1)
#     ld -= np.einsum('tnd,tnd->tn', aY, aBeta)                # e^(-y beta)
#     return ld                                                # per-temp,Y log-density

# def dprodgamma_log_my_mt(aY, aAlpha, aBeta):
#     """
#     product of gammas log-density for multiple y, multiple theta 
#     ----
#     aY     : array of Y     (t x n x d) [Y in R^d]
#     aAlpha : array of alpha (t x J x d)
#     aBeta  : array of beta  (t x J x d)
#     ----
#     returns: (n x t x J)
#     """
#     t, n, d = aY.shape; j = aAlpha.shape[1] # set dimensions
#     ld = np.zeros((n, t, j))
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         ld += np.einsum('tjd,tjd->tj', aAlpha, np.log(aBeta)).reshape(1, t, j)
#         ld -= np.einsum('tjd->tj', gammaln(aAlpha)).reshape(1, t, j)
#         ld += np.einsum('tnd,tjd->ntj', np.log(aY), aAlpha - 1)
#     ld -= np.einsum('tnd,tjd->ntj', aY, aBeta)
#     return ld

# def dprodgamma_log_my_st(aY, aAlpha, aBeta):
#     """
#     Log-density for product of Gammas 

#     Args:
#         aY      (t, n, d)
#         aAlpha  (t, d)
#         aBeta   (t, d)
#     Returns:
#         log-density (t, n)
#     """
#     ld = np.zeros(aY.shape[:-1])
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         ld += np.einsum('td,td->t', aAlpha, np.log(aBeta)).reshape(-1,1)
#         ld -= np.einsum('td->t', gammaln(aAlpha)).reshape(-1,1)
#         ld += np.einsum('tnd,td->tn', np.log(aY), aAlpha - 1)
#     ld -= np.einsum('tnd,td->tn', aY, aBeta)
#     return ld

# def dprojgamma_log_my_mt(aY, aAlpha, aBeta):
#     t,n,d = aY.shape; j = aAlpha.shape[1] # set dimensions
#     ld = np.zeros((n,t,j))
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         ld += np.einsum('tjd,tjd->tj', aAlpha, np.log(aBeta))[None,:,:]
#         ld -= np.einsum('tjd->tj', gammaln(aAlpha))[None,:,:]
#         ld += np.einsum('tnd,tjd->ntj', np.log(aY), aAlpha - 1)
#         ld += gammaln(np.einsum('tjd->tj', aAlpha))[None,:,:]
#         ld -= np.einsum('tjd->tj', aAlpha)[None,:,:] * np.log(np.einsum('nd,tjd->ntj', aY, aBeta))
#     np.nan_to_num(ld, False, -np.inf)
#     return ld

# def dprojgamma_log_paired_yt(aY, aAlpha, aBeta):
#     """
#     projected gamma log-density (proportional) for paired y, theta
#     ----
#     aY     : array of Y     (n, d) [Y in S_p^{d-1}]
#     aAlpha : array of alpha (t, n, d)
#     aBeta  : array of beta  (t, n, d)
#     ----
#     returns: (t, n)
#     """
#     ld = np.zeros(aAlpha.shape[:-1])
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         ld += np.einsum('tnd,tnd->tn', aAlpha, np.log(aBeta))
#         ld -= np.einsum('tnd->tn', gammaln(aAlpha))
#         ld += np.einsum('nd,tnd->tn', np.log(aY), (aAlpha - 1))
#     ld += gammaln(np.einsum('tnd->tn', aAlpha))
#     ld -= np.einsum('tnd->tn',aAlpha) * np.log(np.einsum('nd,tnd->tn', aY, aBeta))
#     return ld

# def dgamma_log_my(aY, alpha, beta):
#     """
#     log-density of Gamma distribution

#     Args:
#         aY    : (n x d)
#         alpha : float
#         beta  : float
#     """
#     lp = np.zeros(aY.shape[0])
#     with np.errstate(divide = 'ignore',invalid = 'ignore'):
#         lp += alpha * np.log(beta)
#         lp -= gammaln(alpha)
#         lp += (alpha - 1) * np.log(aY)
#         lp -= beta * aY
#     return lp

# def dmvnormal_log_mx(x, mu, cov_chol, cov_inv):
#     """ 
#     multivariate normal log-density for multiple x, single theta per temp
#     ------
#     x        : array of alphas    (t x j x d)
#     mu       : array of mus       (t x d)
#     cov_chol : array of cov chols (t x d x d)
#     cov_inv  : array of cov mats  (t x d x d)    
#     """
#     ld = np.zeros(x.shape[:-1])
#     with np.errstate(divide = 'ignore', invalid = 'ignore'):
#         ld -= np.einsum('tdd->t', np.log(cov_chol)).reshape(-1,1)
#     ld -= 0.5 * np.einsum(
#         'tjd,tdl,tjl->tj', 
#         x - mu.reshape(-1,1,cov_chol.shape[1]), 
#         cov_inv, 
#         x - mu.reshape(-1,1,cov_chol.shape[1]),
#         )
#     return ld

# def dmvnormal_log_mx_st(x, mu, cov_chol, cov_inv):
#     ld = np.zeros(x.shape[:-1])
#     ld -= np.log(np.diag(cov_chol)).sum()
#     ld -= 0.5 * np.einsum('td,dl,tl->t',x - mu, cov_inv, x - mu)
#     return ld

# def dinvwishart_log_ms(Sigma, nu, psi):
#     ld = np.zeros(Sigma.shape[0])
#     ld += 0.5 * nu * slogdet(psi)[1]
#     ld -= multigammaln(nu / 2, psi.shape[-1])
#     ld -= 0.5 * nu * psi.shape[-1] * log(2.)
#     ld -= 0.5 * (nu + Sigma.shape[-1] + 1) * slogdet(Sigma)[1]
#     ld -= 0.5 * np.einsum(
#             '...ii->...', np.einsum('ji,...ij->...ij', psi, inv(Sigma)),
#             )
#     return ld

def log_density_log_zeta_j(log_zeta_j, log_yj_sv, yj_sv, nj, Sigma_inv, mu, xi, tau):
    """
    log-density for log-zeta (shape parameter vector for gamma likelihood) per cluster
    with rate parameter integrated out, and summary statistics for gamma RV's
    pre-calculated. (log Y and Y summed vertically per-cluster)
    ---
    log_zeta_j : (m x d)
    log_yj_sv  : (m x d)
    yj_sv      : (m x d)
    nj         : (m)
    Sigma_inv  : (m x d x d)
    mu         : (m x d)
    xi         : (m x d-1)
    tau        : (m x d-1)
    ---
    returns:
    ld         : (m)
    """
    zeta_j = np.exp(log_zeta_j)
    ld = np.zeros(nj.shape[0])
    ld += np.einsum('md,md->m', zeta_j - 1, log_yj_sv)
    ld -= nj * np.einsum('md->m', gammaln(zeta_j))
    ld += np.einsum('md->m', gammaln(nj.reshape(-1,1) * zeta_j[:,1:] + xi))
    ld -= np.einsum('md,md->m', nj.reshape(-1,1) * zeta_j[:,1:] + xi, np.log(yj_sv[:,1:] + tau))
    ld -= 0.5 * np.einsum('ml,mld,md->m', log_zeta_j - mu, Sigma_inv, log_zeta_j - mu)
    return ld

def cluster_covariance_mat(S, mS, nS, delta, covs, mus, n, temps):
    """
    S      : cluster cov mat                      : (t x J x d x d)
    mS     : cluster mean mat                     : (t x J x d)
    nS     : cluster sample size                  : (t x J)
    delta  : matrix of cluster identification     : (t x n)
    covs   : running covariance matrix per datum  : (t x n x d x d)
    mus    : running mean per datum               : (t x n x d)
    n      : running sample size                  : int
    temps  : np.arange(self.nTemp)               : (t)
    """
    S[:] = 0    # cluster covariance
    mS[:] = 0   # cluster mean
    nS[:] = 0   # cluster Sample Size
    mC = np.empty((delta.shape[0], S.shape[-1])) # temporary mean
    nC = np.zeros((delta.shape[0], 1))           # temporary sample size
    for j in range(delta.shape[1]):
        nC[:] = nS[temps, delta.T[j], None] + n
        mC[:] = 1 / nC * (
            + nS[temps, delta.T[j], None] * mS[temps, delta.T[j]] 
            + n * mus[j]
            )
        S[temps, delta.T[j]] = 1 / nC[:,:,None] * (
            + nS[temps, delta.T[j], None, None] * S[temps, delta.T[j]]
            + n * covs[j]
            + np.einsum(
                't,tp,tq->tpq', 
                nS[temps, delta.T[j]], 
                mS[temps, delta.T[j]] - mC,
                mS[temps, delta.T[j]] - mC,
                )
            + n * np.einsum(
                'tp,tq->tpq', 
                mus[j] - mC, 
                mus[j] - mC,
                )
            )
    S += np.eye(S.shape[-1]) * 1e-9
    return

class Samples(object):
    zeta  = None
    sigma = None
    mu    = None
    Sigma = None
    xi    = None
    tau   = None
    delta = None
    r     = None
    eta   = None

    def __init__(self, nSamp, nDat, nCol, nTemp):
        self.zeta  = [None] * (nSamp + 1)
        self.sigma = [None] * (nSamp + 1)
        self.mu    = np.empty((nSamp + 1, nTemp, nCol))
        self.Sigma = np.empty((nSamp + 1, nTemp, nCol, nCol))
        self.xi    = np.empty((nSamp + 1, nTemp, nCol - 1))
        self.tau   = np.empty((nSamp + 1, nTemp, nCol - 1))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nTemp, nDat))
        self.eta   = np.empty((nSamp + 1, nTemp))
        self.ld    = np.empty((nSamp + 1))
        return

class Samples_(Samples):
    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = [None] * (nSamp + 1)
        self.sigma = [None] * (nSamp + 1)
        self.mu    = np.empty((nSamp + 1, nCol))
        self.Sigma = np.empty((nSamp + 1, nCol, nCol))
        self.xi    = np.empty((nSamp + 1, nCol - 1))
        self.tau   = np.empty((nSamp + 1, nCol - 1))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.eta   = np.empty((nSamp + 1))
        self.ld    = np.empty((nSamp + 1))
        return

class Chain(DirichletProcessSampler):
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_sigma(self):
        return self.samples.sigma[self.curr_iter]
    @property
    def curr_mu(self):
        return self.samples.mu[self.curr_iter]
    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter]
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

    @property
    def curr_cluster_count(self):
        return self.curr_delta[0].max() + 1
    
    def average_cluster_count(self, ns):
        acc = self.samples.delta[(ns//2):][:,0].max(axis = 1).mean() + 1
        return '{:.2f}'.format(acc)
    
    am_alpha  = None
    max_clust_count = None

    #updated
    def sample_delta(self, delta, r, zeta, sigma, eta):
        """modify cluster assignments in-place.

        Args:
            delta : (t, n)
            r     : (t, n)
            zeta  : (t, J, d)
            sigma : (t, J, d)
            eta   : (t)

        Returns:
            delta
        """
        Y = r[:,:,None] * self.data.Yp[None,:,:]               # (t, n, d)
        # Y = np.einsum('tn,nd->tnd', r, self.data.Yp)           # (t, n, d)
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count) # (t, J)
        cand_cluster_state = (curr_cluster_state == 0)         # (t, J)
        log_likelihood = pt_logd_prodgamma_my_mt(Y, zeta, sigma) # dprodgamma_log_my_mt(Y, zeta, sigma)  # (n, t, J)
        # Parallel Tempering
        log_likelihood *= self.itl.reshape(1,-1,1)
        tidx = np.arange(self.nTemp)                          # (t)
        p = uniform(size = (Y.shape[1], Y.shape[0]))           # (n, t)
        p += tidx.reshape(1,-1)
        scratch = np.empty(curr_cluster_state.shape)           # (t, J)
        for i in range(self.nDat):
            curr_cluster_state[tidx, delta.T[i]] -= 1
            scratch[:] = 0
            scratch += curr_cluster_state
            scratch += cand_cluster_state * (eta / (cand_cluster_state.sum(axis = 1) + 1e-9)).reshape(-1,1)
            with np.errstate(divide = 'ignore', invalid = 'ignore', under = 'ignore'):
                np.log(scratch, out = scratch)
            scratch += log_likelihood[i]
            np.nan_to_num(scratch, False, -np.inf)
            scratch -= scratch.max(axis = 1).reshape(-1,1)
            with np.errstate(under='ignore'):
                np.exp(scratch, out = scratch)
            np.cumsum(scratch, axis = 1, out = scratch)
            scratch /= scratch.T[-1].reshape(-1,1)
            scratch += tidx.reshape(-1,1)
            delta.T[i] = np.searchsorted(scratch.ravel(), p[i]) % self.max_clust_count
            curr_cluster_state[tidx, delta.T[i]] += 1
            cand_cluster_state[tidx, delta.T[i]] = False
        
        return
    
    # updated
    def clean_delta_zeta_sigma(self, delta, zeta, sigma):
        """
        Clean in-place cluster assignments and cluster values
    
        Args:
            delta (t x n)
            zeta  (t x J)
            sigma (t x J)
        """
        for t in range(self.nTemp):
            keep, delta[t] = np.unique(delta[t], return_inverse = True)
            zeta[t][:keep.shape[0]] = zeta[t, keep]
            sigma[t][:keep.shape[0]] = sigma[t, keep]
        return
    
    # updated 
    def sample_zeta_new(self, mu, Sigma_chol):
        """
        mu         : (t x d)
        Sigma_chol : (t x d x d)
        out        : (t x J x d) # Modified in-place
        """
        out = np.empty((self.nTemp, self.max_clust_count, self.nCol))
        np.einsum(
            'tzy,tjy->tjz',
            np.triu(Sigma_chol), 
            normal(size = (self.nTemp, self.max_clust_count, self.nCol)),
            out = out
            )
        out += mu.reshape(self.nTemp, 1, self.nCol)
        np.exp(out, out=out)
        return out
    
    # updated
    def sample_sigma_new(self, xi, tau):
        """
        xi  : (t x (d-1))
        tau : (t x (d-1))
        out : (t x J x d) # Modified in-place
        """
        # out[:,:,0] = 1
        out = np.ones((self.nTemp, self.max_clust_count, self.nCol))
        out[:,:,1:] = gamma(
            xi.reshape(self.nTemp, 1, -1), 
            scale = 1 / tau.reshape(self.nTemp, 1, self.nCol - 1), 
            size = (self.nTemp, self.max_clust_count, self.nCol - 1),
            )
        return out

    def am_covariance_matrices(self, delta, index):
        return self.am_Sigma.cluster_covariance(delta)[index]

    def sample_zeta(self, zeta, delta, r, xi, tau, mu, Sigma_inv):
        """
        zeta      : (t x J x d)
        delta     : (t x n)
        r         : (t x n)
        mu        : (t x d)
        Sigma_cho : (t x d x d)
        Sigma_inv : (t x d x d)
        """
        Y = np.einsum('tn,nd->tnd', r, self.data.Yp) # (t x n x d)
        lY = np.log(Y)                               # (t x n x d)
        
        curr_cluster_state = bincount2D_vectorized(delta, self.max_clust_count) # (t x J)
        cand_cluster_state = (curr_cluster_state == 0)                          # (t x J)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)          # (t x n x J)
        
        idx     = np.where(~cand_cluster_state) # length m
        covs    = self.am_scale * self.am_covariance_matrices(delta, idx)
        lz_curr = np.log(zeta)
        lz_cand = lz_curr.copy()
        lz_cand[idx] += np.einsum('mpq,mq->mp', cholesky(covs), normal(size = (idx[0].shape[0], self.nCol)))
        Ysv     = np.einsum('tnd,tnj->tjd', Y, delta_ind_mat)[idx]  # Y sum vertical
        lYsv    = np.einsum('tnd,tnj->tjd', lY, delta_ind_mat)[idx] # logY sum vertical
        nj      = curr_cluster_state[idx]
        
        self.am_alpha[:]   = -np.inf
        self.am_alpha[idx] = self.itl[idx[0]] * (
            + log_density_log_zeta_j(
                lz_cand[idx], lYsv, Ysv, nj, Sigma_inv[idx[0]], mu[idx[0]], xi[idx[0]], tau[idx[0]],
                )
            - log_density_log_zeta_j(
                lz_curr[idx], lYsv, Ysv, nj, Sigma_inv[idx[0]], mu[idx[0]], xi[idx[0]], tau[idx[0]],
                )
            )
        keep = np.where(np.log(uniform(size = self.am_alpha.shape)) < self.am_alpha)
        zeta[keep] = np.exp(lz_cand[keep])
        return zeta
    
    def sample_sigma(self, zeta, r, delta, xi, tau):
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
        
        sigma_new = np.ones(zeta.shape)
        sigma_new[:,:,1:] = gamma(shape = shape, scale = 1 / rate)
        
        return sigma_new

    def sample_mu(self, zeta, Sigma_inv, extant_clusters):
        """
        zeta            : (t x J x d)
        Sigma_Inv       : (t x d x d)
        extant_clusters : (t x J)     (flag indicating an extant cluster)
        """
        n     = extant_clusters.sum(axis = 1) # number of clusters per temp (t)
        assert np.all(zeta[extant_clusters] > 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            lzbar = np.nansum(np.log(zeta) * extant_clusters[:,:,None], axis = 1) / n[:, None] # (t x d)
        # lzbar = np.log(zeta * extant_clusters[:,:,None]).sum(axis = 1) / n
        # _Sigma = inv((n * self.itl).reshape(-1,1,1) * Sigma_inv + self.priors.mu.SInv)
        _Sigma = inv(n[:,None,None] * Sigma_inv + self.priors.mu.SInv)
        _mu = np.einsum(
            'tjl,tl->tj',
            _Sigma,
            + self.priors.mu.SInv @ self.priors.mu.mu 
            + np.einsum('tjl,tl->tj', Sigma_inv, n[:,None] * lzbar)
            )
        #_mu = np.einsum('tkl,td,tdl->tk', _Sigma, (n * self.itl)[:,None] * lzbar, Sigma_inv)
        return _mu + np.einsum('tkl,tl->tk', cholesky(_Sigma), normal(size = (self.nTemp, self.nCol)))

    def sample_Sigma(self, zeta, mu, extant_clusters):
        """
        zeta            : (t x J x d)
        mu              : (t x d)
        extant_clusters : (t x J)
        """
        n = extant_clusters.sum(axis = 1)
        diff = (np.log(zeta) - mu[:,None]) * extant_clusters[:,:,None]
        C = np.einsum('tjd,tje->tde', diff, diff)
        _psi = self.priors.Sigma.psi + C * self.itl.reshape(-1,1,1)
        _nu  = self.priors.Sigma.nu + n * self.itl
        # horribly inefficient code!
        out = np.empty((self.nTemp, self.nCol, self.nCol))
        for i in range(self.nTemp):
            out[i] = invwishart.rvs(df = _nu[i], scale = _psi[i])
        return out
    
    def sample_xi(self, curr_xi, sigma, extant_clusters):
        """
        curr_xi : (t x J x d)
        sigma   : (t x J x d)
        extant_clusters : (t x J)
        """
        # return np.ones(curr_xi.shape)
        n  = np.repeat(extant_clusters.sum(axis = 1), sigma.shape[-1] - 1)          # (t x (d-1))
        assert np.all(sigma[extant_clusters] > 0.) # verify inputs
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            ls = np.nansum(np.log(sigma[:,:,1:]) * extant_clusters[:,:,None], axis = 1) # (t, d-1)
        s  = (sigma[:,:,1:] * extant_clusters[:,:,None]).sum(axis = 1)              # (t, d-1)
        args = zip(
            curr_xi.ravel(), n, s.ravel(), ls.ravel(),
            repeat(self.priors.xi.a), repeat(self.priors.xi.b), 
            repeat(self.priors.tau.a), repeat(self.priors.tau.b),
            )
        res = map(sample_xi_wrapper, args)
        return np.array(list(res)).reshape(curr_xi.shape)

    def sample_tau(self, sigma, xi, extant_clusters):
        # return np.ones(xi.shape)
        n = extant_clusters.sum(axis = 1) # .reshape(-1, 1)
        s = (sigma[:,:,1:] * extant_clusters[:,:,None]).sum(axis = 1) # (t, d-1)
        shape = n[:, None] * xi + self.priors.tau.a # (t, d-1)
        rate  = s + self.priors.tau.b               # (t, d-1)
        return gamma(shape = shape, scale = 1 / rate)  # (t, d-1)

    def sample_r(self, delta, zeta, sigma):
        """
        delta : (t x n)
        zeta  : (t x J x d)
        sigma : (t x J x d)
        """
        # As = zeta[self.temp_unravel, delta.ravel()].reshape(self.nTemp, self.nDat, self.nCol).sum(axis = 2)
        As = zeta[self.temp_unravel, delta.ravel()].sum(axis = 1).reshape(self.nTemp, self.nDat)
        Bs = np.einsum(
            'nd,tnd->tn', 
            self.data.Yp, 
            sigma[self.temp_unravel, delta.ravel()].reshape(self.nTemp, self.nDat, self.nCol),
            )
        return gamma(shape = As, scale = 1/Bs)
    
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
        # aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)

    def update_am_cov(self):
        """ Online updating for Adaptive Metropolis Covariance per obsv. """
        lzeta = np.swapaxes(
            np.log(
                self.curr_zeta[
                    self.temp_unravel, self.curr_delta.ravel()
                    ].reshape(
                        self.nTemp, self.nDat, self.nCol
                        )
                ),
            0, 1,
            )
        self.am_Sigma.update(lzeta)
        return

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol, self.nTemp)
        self.samples.zeta[0]  = np.exp(normal(size = (self.nTemp, self.max_clust_count, self.nCol)))
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = (self.nTemp, self.max_clust_count, self.nCol))
        self.samples.Sigma[0] = np.eye(self.nCol) * 2
        self.samples.mu[0]    = 0.
        self.samples.xi[0]    = 1.
        self.samples.tau[0]   = 1.
        self.samples.eta[0]   = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 20, size = (self.nTemp, self.nDat))
        self.samples.r[0]     = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0], self.samples.sigma[0],
                )
        self.am_alpha     = np.zeros((self.nTemp, self.max_clust_count))
        self.candidate_zeta  = np.zeros((self.nTemp, self.max_clust_count, self.nCol))
        self.candidate_sigma = np.zeros((self.nTemp, self.max_clust_count, self.nCol))
        self.curr_iter = 0
        return
    
    def record_log_density(self):
        lpl = np.zeros(self.nTemp)
        lpp = np.zeros(self.nTemp)
        Y = self.curr_r[:,:,None] * self.data.Yp[None,:,:] 
            # np.einsum('tn,nd->tnd', self.curr_r, self.data.Yp)
        # Compute log-likelihood
        lpl += pt_logd_prodgamma_paired(
            Y,
            self.curr_zeta[
                self.temp_unravel, self.curr_delta.ravel(),
                ].reshape(
                    self.nTemp, self.nDat, self.nCol,
                    ),
            self.curr_sigma[
                self.temp_unravel, self.curr_delta.ravel(),
                ].reshape(
                    self.nTemp, self.nDat, self.nCol,
                    ),
            ).sum(axis = 1)
        ext_clust = bincount2D_vectorized(self.curr_delta, self.max_clust_count) > 0
        lpl += ((self.nCol - 1) * np.log(self.curr_r)).sum(axis = 1)
        lpl += np.einsum('tj,tj->t', 
                pt_logd_mvnormal_mx_st(np.log(self.curr_zeta), 
                    self.curr_mu, 
                    cholesky(self.curr_Sigma),
                    inv(self.curr_Sigma),
                    ), 
                ext_clust,
                )
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            lpl += np.nansum(
                pt_logd_prodgamma_my_st(
                    self.curr_sigma[:,:,1:], 
                    self.curr_xi, 
                    self.curr_tau,
                    ) * ext_clust, 
                    axis = 1,
                    )
        # Compute prior log-density
        lpp += logd_mvnormal_mx_st(self.curr_mu, *self.priors.mu)
        lpp += logd_invwishart_ms(self.curr_Sigma, *self.priors.Sigma)
        lpp += logd_gamma_my(self.curr_eta, *self.priors.eta)
        lpp += logd_gamma_my(self.curr_xi, *self.priors.xi).sum(axis = 1)
        lpp += logd_gamma_my(self.curr_tau, *self.priors.tau).sum(axis = 1)
        
        # assemble them, record cold chain value
        self.samples.ld[self.curr_iter] = (lpl + lpp)[0]
        return
    
    def iter_sample(self):
        # Setup, parsing
        delta = self.curr_delta.copy()
        zeta  = self.curr_zeta.copy()
        sigma = self.curr_sigma.copy()
        mu    = self.curr_mu
        Sigma = self.curr_Sigma
        Sigma_cho = cholesky(Sigma)
        Sigma_inv = inv(Sigma)
        xi    = self.curr_xi
        tau   = self.curr_tau
        eta   = self.curr_eta
        r     = self.curr_r

        # Adaptive Metropolis Update
        self.update_am_cov()
        
        # Advance the iterator
        self.curr_iter += 1

        # Sample New Candidate Clusters
        cand_clusters = np.where(bincount2D_vectorized(delta, self.max_clust_count) == 0)
        zeta[cand_clusters] = self.sample_zeta_new(mu, Sigma_cho)[cand_clusters]
        sigma[cand_clusters] = self.sample_sigma_new(xi, tau)[cand_clusters]

        # Compute Cluster assignments & re-index
        self.sample_delta(delta, r, zeta, sigma, eta)
        self.clean_delta_zeta_sigma(delta, zeta, sigma)
        self.samples.delta[self.curr_iter] = delta

        # Compute additional parameters
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta, sigma)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_delta, self.curr_r, xi, tau, mu, Sigma_inv
                )
        self.samples.sigma[self.curr_iter] = self.sample_sigma(
                self.curr_zeta, self.curr_r, self.curr_delta, xi, tau,
                )
        extant_clusters = bincount2D_vectorized(delta, self.max_clust_count) > 0
        self.samples.xi[self.curr_iter]  = self.sample_xi(xi, self.curr_sigma, extant_clusters)
        self.samples.tau[self.curr_iter] = self.sample_tau(self.curr_sigma, self.curr_xi, extant_clusters)
        self.samples.mu[self.curr_iter] = self.sample_mu(self.curr_zeta, Sigma_inv, extant_clusters)        
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_zeta, self.curr_mu, extant_clusters)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, self.curr_delta)

        # attempt swap
        if self.curr_iter >= self.swap_start:
            lpl = np.zeros(self.nTemp)
            lpp = np.zeros(self.nTemp)
            Y = np.einsum('tn,nd->tnd', self.curr_r, self.data.Yp) # (t x n x d)
            lpl += pt_logd_prodgamma_paired( # dprodgamma_log_paired_yt(
                Y, 
                self.curr_zeta[self.temp_unravel, delta.ravel()].reshape(self.nTemp, self.nDat, self.nCol),
                self.curr_sigma[self.temp_unravel, delta.ravel()].reshape(self.nTemp, self.nDat, self.nCol),
                ).sum(axis = 1)
            lpl += ((self.nCol - 1) * np.log(self.curr_r)).sum(axis = 1)
            lpl += np.einsum('tj,tj->t', 
                    pt_logd_mvnormal_mx_st(np.log(self.curr_zeta), mu, Sigma_cho, Sigma_inv), extant_clusters,
                    )
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                lpl += np.nansum(pt_logd_prodgamma_my_st(self.curr_sigma[:,:,1:], self.curr_xi, self.curr_tau) * extant_clusters, axis = 1)
            lpp += logd_mvnormal_mx_st(self.curr_mu, *self.priors.mu)
            lpp += logd_invwishart_ms(self.curr_Sigma, *self.priors.Sigma)
            lpp += logd_gamma_my(self.curr_xi, *self.priors.xi).sum(axis = 1)
            lpp += logd_gamma_my(self.curr_tau, *self.priors.tau).sum(axis = 1)
            lpp += logd_gamma_my(self.curr_eta, *self.priors.eta)
            
            sw = choice(self.nTemp, 2 * self.nSwap_per, replace = False).reshape(-1,2)
            lpo = lpl + lpp
            sw_alpha = (self.itl[sw.T[0]] - self.itl[sw.T[1]]) * (lpo[sw.T[0]] - lpo[sw.T[1]])
            # sw_alpha = (self.itl[sw.T[1]] - self.itl[sw.T[0]]) * (lpl[sw.T[0]] - lpl[sw.T[1]]) + (lpp[sw.T[0]] - lpp[sw.T[1]])
            logp = np.log(uniform(size = sw.shape[0]))
            for tt in sw[np.where(logp < sw_alpha)[0]]:
                self.samples.r[self.curr_iter, tt[0]], self.samples.r[self.curr_iter, tt[1]] = \
                    self.samples.r[self.curr_iter, tt[1]].copy(), self.samples.r[self.curr_iter, tt[0]].copy()
                self.samples.zeta[self.curr_iter][tt[0]], self.samples.zeta[self.curr_iter][tt[1]] = \
                    self.samples.zeta[self.curr_iter][tt[1]].copy(), self.samples.zeta[self.curr_iter][tt[0]].copy()
                self.samples.sigma[self.curr_iter][tt[0]], self.samples.sigma[self.curr_iter][tt[1]] = \
                    self.samples.sigma[self.curr_iter][tt[1]].copy(), self.samples.sigma[self.curr_iter][tt[0]].copy()
                self.samples.mu[self.curr_iter, tt[0]], self.samples.mu[self.curr_iter, tt[1]] = \
                    self.samples.mu[self.curr_iter, tt[1]].copy(), self.samples.mu[self.curr_iter, tt[0]].copy()
                self.samples.Sigma[self.curr_iter, tt[0]], self.samples.Sigma[self.curr_iter, tt[1]] = \
                    self.samples.Sigma[self.curr_iter, tt[1]].copy(), self.samples.Sigma[self.curr_iter, tt[0]].copy()
                self.samples.xi[self.curr_iter, tt[0]], self.samples.xi[self.curr_iter, tt[1]] = \
                    self.samples.xi[self.curr_iter, tt[1]].copy(), self.samples.xi[self.curr_iter, tt[0]].copy()
                self.samples.tau[self.curr_iter, tt[0]], self.samples.tau[self.curr_iter, tt[1]] = \
                    self.samples.tau[self.curr_iter, tt[1]].copy(), self.samples.tau[self.curr_iter, tt[0]].copy()
                self.samples.delta[self.curr_iter, tt[0]], self.samples.delta[self.curr_iter, tt[1]] = \
                    self.samples.delta[self.curr_iter, tt[1]].copy(), self.samples.delta[self.curr_iter, tt[0]].copy()
        
        self.record_log_density()
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        if type(path) is str:
            folder = os.path.split(path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            if os.path.exists(path):
                os.remove(path)

        nclust = np.array([delta.max() for delta in self.samples.delta[nBurn :: nThin, 0]]) + 1
        zetas  = np.vstack([
            np.hstack((np.ones((nclust[i], 1)) * i, zeta[0][:nclust[i]]))
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        sigmas = np.vstack([
            np.hstack((np.ones((nclust[i], 1)) * i, sigma[0][:nclust[i]]))
            for i, sigma in enumerate(self.samples.sigma[nBurn :: nThin])
            ])
        mus    = self.samples.mu[nBurn :: nThin, 0]
        Sigmas = self.samples.Sigma[nBurn :: nThin, 0]
        xis    = self.samples.xi[nBurn :: nThin, 0]
        taus   = self.samples.tau[nBurn :: nThin, 0]
        deltas = self.samples.delta[nBurn :: nThin, 0]
        rs     = self.samples.r[nBurn :: nThin, 0]
        etas   = self.samples.eta[nBurn :: nThin, 0]

        out = {
            'nclust' : nclust,
            'zetas'  : zetas,
            'sigmas' : sigmas,
            'mus'    : mus,
            'Sigmas' : Sigmas,
            'xis'    : xis,
            'taus'   : taus,
            'deltas' : deltas,
            'rs'     : rs,
            'etas'   : etas,
            'V'      : self.data.V,
            'logd'   : self.samples.ld,
            'time'   : self.time_elapsed_numeric,
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
    
    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_mu    = (0., 4.),
            prior_Sigma = (20, 1.),
            prior_xi    = GammaPrior(1., 1.),
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
        _prior_mu = NormalPrior(
            np.ones(self.nCol) * prior_mu[0],
            np.eye(self.nCol) * np.sqrt(prior_mu[1]),
            np.eye(self.nCol) / np.sqrt(prior_mu[1]),
            )
        _prior_Sigma = InvWishartPrior(
            self.nCol + prior_Sigma[0],
            (self.nCol + prior_Sigma[0]) * np.eye(self.nCol) * prior_Sigma[1],
            )
        _prior_eta = GammaPrior(*prior_eta)
        _prior_xi  = GammaPrior(*prior_xi)
        _prior_tau = GammaPrior(*prior_tau)
        self.priors = Prior(
            _prior_eta, _prior_mu, _prior_Sigma, _prior_xi, _prior_tau
            )
        self.set_projection()
        self.itl = 1 / stepping**np.arange(ntemps)
        self.nTemp = ntemps
        self.temp_unravel = np.repeat(np.arange(self.nTemp), self.nDat)
        self.nSwap_per = self.nTemp // 2
        self.swap_start = 100
        
        self.am_Sigma = PerObsTemperedOnlineCovariance(
            self.nTemp, self.nDat, self.nCol, self.max_clust_count
            )
        self.am_scale = 2.38**2 / self.nCol
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            njs = np.bincount(
                self.samples.delta[s], 
                minlength = int(self.samples.delta[s].max() + 1 + m),
                )
            ljs = njs + (njs == 0) * self.samples.eta[s] / m
            new_zetas = np.exp(
                + self.samples.mu[s].reshape(1,self.nCol) 
                + (cholesky(self.samples.Sigma[s]) @ normal(size = (self.nCol, m))).T
                )
            new_sigmas = np.hstack((
                np.ones((m, 1)),
                gamma(shape = self.samples.xi[s], scale = self.samples.tau[s], size = (m, self.nCol - 1)),
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
        zetas  = out['zetas']
        sigmas = out['sigmas']
        mus    = out['mus']
        Sigmas = out['Sigmas']
        xis    = out['xis']
        taus   = out['taus']
        rs     = out['rs']
        etas   = out['etas']
        
        self.data = Data_From_Sphere(out['V'])
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass
        
        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = mus.shape[1]

        self.samples       = Samples_(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.xi    = xis
        self.samples.tau   = taus
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.sigma = [sigmas[np.where(sigmas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.mu    = mus
        self.samples.Sigma = Sigmas.reshape(self.nSamp, self.nCol, self.nCol)
        self.samples.r     = rs
        self.samples.ld    = out['logd']
        return

    def __init__(self, path):
        self.load_data(path)
        return

if __name__ == '__main__':
    from data import Data_From_Raw
    from projgamma import GammaPrior
    from pandas import read_csv
    import os
    import time

    raw = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(10000)
    model.write_to_disk('./test/results.pkl', 5000, 5)
    res = Result('./test/results.pkl')
    res.write_posterior_predictive('./test/postpred.csv')


# EOF
