from re import S
from tkinter import N
import numpy as np
np.errstate(divide = 'ignore')
from numpy.random import choice, gamma, beta, normal, uniform
from numpy.linalg import cholesky, slogdet, inv
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import invwishart
from scipy.special import gammaln, multigammaln
from collections import namedtuple
from itertools import repeat
import pandas as pd
import os
import sqlite3 as sql
from math import ceil, log

# import pt
from data import Data
import cUtility as cu
from cProjgamma import sample_alpha_1_mh, sample_alpha_k_mh, sample_beta_fc, \
                        logddirichlet, logdgamma, logdgamma_restricted
from data import euclidean_to_angular, euclidean_to_simplex, euclidean_to_hypercube, Data
from projgamma import GammaPrior
from energy import limit_cpu

NormalPrior     = namedtuple('NormalPrior', 'mu SCho SInv')
InvWishartPrior = namedtuple('InvWishartPrior', 'nu psi')
Prior = namedtuple('Prior', 'mu Sigma xi tau')

def bincount2D_vectorized(arr, m):
    """
    code from stackoverflow:
        https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    
    Args:
        arr : (np.ndarray(int)) -- matrix of cluster assignments by temperature (t x n)
        m   : (int)             -- maximum number of clusters

    Returns:
        (np.ndarray(int)): matrix of cluster counts by temperature (t x J)
    """
    arr_offs = arr + np.arange(arr.shape[0])[:,None] * m
    return np.bincount(arr_offs.ravel(), minlength=arr.shape[0] * m).reshape(-1, m)

def dprodgamma_log_mt_(vY, aAlpha, aBeta, vlogConstant, out):
    """ 
    log density -- product of gammas -- single y (vector) against multiple thetas (array)
    modifies out vector in-place

    vY     : vector of Y    (t x d)
    aAlpha : array of alpha (t x J x d)
    aBeta  : array of beta  (t x J x d)
    vlogC  : array of logC  (t x J) (normalizing constant for product of betas)
    out    : output vector  (t x J)
    """
    out *= 0
    out += vlogConstant
    out += np.dot(np.log(vY), (aAlpha - 1).T)
    out -= np.dot(vY, aBeta.T)
    return

def dprodgamma_log_paired_yt(aY, aAlpha, aBeta):
    """
    product of gammas log-density for paired y, theta
    ----
    aY     : array of Y     (t x n x d) [Y in R^d]
    aAlpha : array of alpha (t x n x d)
    aBeta  : array of beta  (t x n x d)
    ----
    returns: (t x n)
    """
    ld = np.zeros(aY.shape[:-1])                             # n temps x n Y
    ld += np.einsum('tnd,tnd->tn', aAlpha, np.log(aBeta))    # beta^alpha
    ld -= np.einsum('tnd->tn', gammaln(aAlpha))              # gamma(alpha)
    ld += np.einsum('tnd,tnd->tn', np.log(aY), (aAlpha - 1)) # y^(alpha - 1)
    ld -= np.einsum('tnd,tnd->tn', aY, aBeta)                # e^(-y beta)
    return ld                                                # per-temp,Y log-density

def dprodgamma_log_my_mt(aY, aAlpha, aBeta):
    """
    product of gammas log-density for paired y, theta
    ----
    aY     : array of Y     (t x n x d) [Y in R^d]
    aAlpha : array of alpha (t x J x d)
    aBeta  : array of beta  (t x J x d)
    ----
    returns: (n x t x J)
    """
    t, n, d = aY.shape; j = aAlpha.shape[1] # set dimensions
    ld = np.zeros((n, t, d))
    ld += np.einsum('tjd,tjd->tj', aAlpha, np.log(aBeta)).reshape(1, t, j)
    ld -= np.einsum('tnd->tn', gammaln(aAlpha))
    ld += np.einsum('tnd,tjd->ntj', np.log(aY), aAlpha - 1)
    ld -= np.einsum('tnd,tjd->ntj', aY, aBeta)
    return ld

def dprojgamma_log_paired_yt(aY, aAlpha, aBeta):
    """
    projected gamma log-density (proportional) for paired y, theta
    ----
    aY     : array of Y     (t x n x d) [Y in S_p^{d-1}]
    aAlpha : array of alpha (t x n x d)
    aBeta  : array of beta  (t x n x d)
    ----
    returns: (t x n)
    """
    ld = np.zeros(aY.shape[:-1])
    ld += np.einsum('tnd,tnd->tn', aAlpha, np.log(aBeta))
    ld -= np.einsum('tnd->tn', gammaln(aAlpha))
    ld += np.einsum('tnd,tnd->tn', np.log(aY), (aAlpha - 1))
    ld += gammaln(np.einsum('tnd->tn', aAlpha))
    ld -= np.einsum('tnd->tn',aAlpha) * np.log(np.einsum('tnd,tnd->tn', aY, aBeta))
    return ld

def dgamma_log_my(aY, alpha, beta):
    """
    log-density of Gamma distribution

    Args:
        aY    : (n x d)
        alpha : float
        beta  : float
    """
    lp = (
        + alpha * log(beta)
        - gammaln(alpha)
        + (alpha - 1) * np.log(aY)
        - beta * aY
        )
    return lp


def dmvnormal_log_mx(x, mu, cov_chol, cov_inv):
    """ 
    multivariate normal log-density for multiple x, single theta per temp
    ------
    x        : array of alphas    (t x j x d)
    mu       : array of mus       (t x d)
    cov_chol : array of cov chols (t x d x d)
    cov_inv  : array of cov mats  (t x d x d)    
    """
    ld = np.zeros(x.shape[:-1])
    ld -= 0.5 * 2 * np.log(np.diag(cov_chol)).sum()
    ld -= 0.5 * np.einsum('tjd,tdl,tjl->tj', 
            x - mu.reshape(-1,1,cov_chol.shape[1]), 
            cov_inv, 
            x - mu.reshape(-1,1,cov_chol.shape[1]),
            )
    return ld

def dinvwishart_log_ms(Sigma, nu, psi):
    ld = np.zeros(Sigma.shape[0])
    ld += 0.5 * nu * slogdet(psi)[1]
    ld -= multigammaln(nu / 2, psi.shape[-1])
    ld -= 0.5 * nu * psi.shape[-1] * log(2.)
    ld -= 0.5 * (df + Sigma.shape[-1] + 1) * slogdet(Sigma)[1]
    ld -= 0.5 * np.einsum(
            '...ii->...', np.einsum('ji,...ij->...ij', psi, inv(Sigma)),
            )
    return ld

def log_density_log_zeta_j(log_zeta_j, log_yj_sv, yj_sh, nj, Sigma_inv, mu, xi, tau):
    """
    log_zeta_j : (m x d)
    log_yj_sv  : (m x d)   [Summed over n_j]
    yj_sh      : (m x nj)  [Summed over d]
    nj         : (m)
    Sigma_cho  : (m x d x d)
    Sigma_inv  : (m x d x d)
    mu         : (m x d)
    xi         : (m x d-1)
    tau        : (m x d-1)
    """
    zeta_j = np.exp(log_zeta_j)
    ld = np.zeros(log_zeta_j.shape[0])
    ld += np.einsum('md,md->m', zeta_j - 1, log_yj_sv)
    ld -= nj.reshape(-1,1) * np.einsum('md->m', gammaln(zeta_j))
    ld += np.einsum('md->m', gammaln(nj.reshape(-1,1) * zeta_j[:,1:] + xi))
    ld -= np.einsum('md,md->m', nj.reshape(-1,1) * zeta_j[:,1:], np.log(yj_sh + tau))
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
    temps  : np.arange(self.nTemps)               : (t)
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
            + n * mus[i][temps, j]
            )
        S[temps, delta.T[j]] = 1 / nC[:,:,None] * (
            + nS[temps, delta.T[j], None, None] * S[temps, delta.T[j]]
            + n * covs[temps, j]
            + np.einsum(
                't,tp,tq->tpq', 
                nS[temps, delta.T[j]], 
                mS[temps, delta.T[j]] - mC,
                mS[temps, delta.T[j]] - mC,
                )
            + n * np.einsum(
                'tp,tq->tpq', 
                mus[temps, j] - mC, 
                mus[temps, j] - mC,
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
        self.log_zeta_hist = np.empty((nSamp + 1, nTemp, nDat, nCol))
        self.sigma = [None] * (nSamp + 1)
        self.mu    = np.empty((nSamp + 1, nTemp, nCol))
        self.Sigma = np.empty((nSamp + 1, nTemp, nCol, nCol))
        self.xi    = np.empty((nSamp + 1, nTemp, nCol - 1))
        self.tau   = np.empty((nSamp + 1, nTemp, nCol - 1))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nTemp, nDat))
        self.eta   = np.empty((nSamp + 1, nTemp))
        return

class Chain(object):
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
    
    # Adaptive Metropolis Placeholders
    am_cov_c  = None
    am_cov_i  = None
    am_mean_c = None
    am_mean_i = None
    am_n_c    = None
    am_alpha  = None
    max_clust_count = None


    # updated
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
        Y = np.einsum('tn,nd->tnd', r, self.data.Yp)           # (t, n, d)
        curr_cluster_state = bincount2D_vectorized(delta) # (t, J)
        cand_cluster_state = (curr_cluster_state == 0)         # (t, J)
        log_likelihood = dprodgamma_log_my_mt(Y, zeta, sigma)  # (n, t, J)
        # Parallel Tempering
        log_likelihood *= self.inv_temp_ladder.reshape(1,-1,1)
        tidx = np.arange(self.nTemps)                          # (t)
        p = uniform(size = (Y.shape[1], Y.shape[0]))           # (n, t)
        p += tidx.reshape(1,-1)
        scratch = np.empty(curr_cluster_state.shape)           # (t, J)

        for i in range(self.nDat):
            curr_cluster_state[tidx, delta.T[i]] -= 1
            scratch[:] = 0
            scratch += curr_cluster_state
            scratch += cand_cluster_state * (eta / cand_cluster_state.sum(axis = 1)).reshape(-1,1)
            np.log(scratch, out = scratch)
            scratch += log_likelihood[i]
            scratch -= scratch.max(axis = 1).reshape(-1,1)
            np.exp(scratch, out = scratch)
            np.cumsum(scratch, axis = 1, out = scratch)
            scratch /= scratch[:,-1].reshape(-1,1)
            scratch += tidx.reshape(-1,1)
            delta.T[i] = np.searchsorted(scratch.ravel(), p[i]) // self.max_clust_count
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
            zeta[t][:keep.shape[0]] = zeta[t,keep]
            sigma[t][:keep.shape[0]] = sigma[t,keep]
        return
    
    # updated 
    def sample_zeta_new(self, mu, Sigma_chol, out):
        """
        mu         : (t x d)
        Sigma_chol : (t x d x d)
        out        : (t x J x d) # Modified in-place
        """
        out[:] = 0
        np.einsum(
            'tzy,tjy->tjz',
            np.triu(Sigma_chol), 
            normal(size = (self.nTemp, self.max_clust_count, self.nCol)),
            out = out
            )
        out += mu.reshape(self.nTemp, 1, self.nCol)
        return
    
    # updated
    def sample_sigma_new(self, xi, tau, out):
        """
        xi  : (t x (d-1))
        tau : (t x (d-1))
        out : (t x J x d) # Modified in-place
        """
        out[:,:,0] = 1
        out[:,:,1:] = gamma(
            xi.reshape(self.nTemp, 1, -1), 
            scale = 1/tau.reshape(self.nTemp, 1, -1), 
            size = (self.nTemp, self.max_clust_count, self.nCol - 1),
            )
        return

    def am_covariance_matrices(self, delta, index):
        self.am_n_c[:] = 0      # cluster numbers
        self.am_mean_c[:] = 0.  # cluster means
                                # cluster covs
        self.am_cov_c[:] = np.eye(self.am_cov_c.shape[-1]) * 1e-9
        
        temps = np.arange(self.nTemps)

        n = np.empty((self.nTemp))
        m = np.empty((self.nTemp, self.nCol))
        C = np.empty((self.nTemp, self.nCol, self.nCol))

        for i in range(self.nDat):
            n[:] = self.curr_iter
            m[:] = self.am_mean_i[i]
            C[:] = self.am_cov_i[i]

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
        covs    = self.am_covariance_matrices(delta, idx)
        lz_curr = np.log(zeta)
        lz_cand = lz_curr + np.einsum('mpq,mq->mp', cholesky(covs), normal(size = lz_curr.shape))
        Ysh     = np.einsum('tnd,tnj->tjn', Y, delta_ind_mat)[idx]  # Y sum horizontal
        lYsv    = np.einsum('tnd,tnj->tjd', lY, delta_ind_mat)[idx] # logY sum vertical
        nj      = curr_cluster_state[idx]
        
        self.am_alpha[:]   = -np.inf
        self.am_alpha[idx] = self.inv_temp_ladder[idx[0]] * (
            + log_density_log_zeta_j(
                lz_cand[idx], lYsv, Ysh, nj, Sigma_inv[idx[0]], mu[idx[0]], xi[idx[0]], tau[idx[0]],
                )
            - log_density_log_zeta_j(
                lz_curr[idx], lYsv, Ysh, nj, Sigma_inv[idx[0]], mu[idx[0]], xi[idx[0]], tau[idx[0]],
                )
            )
        keep = np.where(np.log(uniform(size = self.am_alpha.shape[0])) < self.am_alpha)
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
        idx = np.where(curr_cluster_state > 0)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)

        shape = np.zeros((self.nTemp, self.max_clust_count, self.nCol - 1))
        shape += xi.reshape(self.nTemp, 1, self.nCol - 1)
        rate  = np.zeros((self.nTemp, self.max_clust_count, self.nCol - 1))
        rate  += tau.reshape(self.nTemp, 1, self.nCol - 1)
        
        Ysv = np.einsum('tnd,tnj->tjd', r.reshape(self.nTemps, self.nDat, 1) * self.data.Yp, delta_ind_mat)

        shape += curr_cluster_state.reshape(self.nTemp, self.max_clust_count, 1) * zeta
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
        n     = extant_clusters.sum(axis = 1)   # number of clusters per temp
        lzbar = np.log(zeta * extant_clusters).sum(axis = 1) / n
        _Sigma = cho_solve(
            cho_factor(n * self.inv_temper_temp * Sigma_inv + self.priors.mu.SInv),
            np.eye(self.nCol),
            )
        _mu = np.einsum('tkl,td,tdl->tk', _Sigma, n * self.inv_temper_temp * lzbar, Sigma_inv)
        return _mu + np.einsum('tkl,tl->tk', cholesky(_Sigma), normal(size = (self.nTemps, self.nCol)))

    def sample_Sigma(self, zeta, mu, extant_clusters):
        """
        zeta            : (t x J x d)
        mu              : (t x d)
        extant_clusters : (t x J)
        """
        n = extant_clusters.sum(axis = 1)
        diff = (np.log(zeta) - mu) * extant_clusters
        C = np.einsum('tjd,tje->tde', diff, diff)
        _psi = self.priors.Sigma.psi + C * self.inv_temper_temp
        _nu  = self.priors.Sigma.nu + n * self.inv_temper_temp
        return invwishart.rvs(df = _nu, scale = _psi)

    # TODO
    def sample_xi(self, curr_xi, sigma, extant_clusters):
        extant_clusters.sum(axis = 1)
        (np.log(sigma) * extant_clusters).sum(axis = 1)
        sigma * extant_clusters.sum(axis = 1)

        pass


    def sample_xi(self, sigma, curr_xi):
        args = zip(
            curr_xi,
            sigma.T[1:],
            repeat(self.priors.xi.a),
            repeat(self.priors.xi.b),
            repeat(self.priors.tau.a),
            repeat(self.priors.tau.b),
            )
        res = map(update_xi_l_wrapper, args)
        # res = self.pool.map(update_xi_l_wrapper, args)
        return np.array(list(res))

    def sample_tau(self, sigma, xi):
        args = zip(
            xi,
            sigma.T[1:],
            repeat(self.priors.tau.a),
            repeat(self.priors.tau.b),
            )
        res = map(update_tau_l_wrapper, args)
        # res = self.pool.map(update_tau_l_wrapper, args)
        return np.array(list(res))

    def sample_r(self, delta, zeta, sigma):
        As = zeta[delta].sum(axis = 1)
        Bs = (self.data.Yp * sigma[delta]).sum(axis = 1)
        return gamma(shape = As, scale = 1/Bs)

    def sample_eta(self, curr_eta, delta):
        g = beta(curr_eta + 1, self.nDat)
        aa = self.priors.eta.a + delta.max() + 1
        bb = self.priors.beta.b - log(g)
        eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        return gamma(shape = aaa, scale = 1 / bb)


    def update_am_cov_initial(self):
        self.am_mean[:] = self.samples.theta_hist[:self.curr_iter].mean(axis = 0)
        self.am_cov[:] = np.einsum(
            'itnj,itnk->tnjk', 
            self.samples.theta_hist[:self.curr_iter] - self.am_mean,
            self.samples.theta_hist[:self.curr_iter] - self.am_mean,
            ) / self.curr_iter
        return

    def update_am_cov(self):
        self.am_mean_i += (self.samples.log_zeta_hist[self.curr_iter] - self.am_mean_i) / self.curr_iter
        self.am_cov_i[:] = (
            + (self.curr_iter / (self.curr_iter + 1)) * self.am_cov_i
            + (self.curr_iter / self.curr_iter / self.curr_iter) * np.einsum(
                'tej,tel->tejl', 
                self.samples.zeta_hist[self.curr_iter] - self.am_mean_i,
                self.samples.zeta_hist[self.curr_iter] - self.am_mean_i,
                )
            )
        return

    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.zeta[0]  = np.exp(normal(size = (self.max_clust_count - 5, self.nCol)))
        self.samples.Sigma[0] = np.eye(self.nCol) * 2
        self.samples.alpha[0] = 1.
        self.samples.beta[0]  = 1.
        self.samples.xi[0]    = 1.
        self.samples.tau[0]   = 1.
        self.samples.zeta[0]  = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 5, self.nCol))
        self.samples.sigma[0] = gamma(shape = 2., scale = 2., size = (self.max_clust_count - 5, self.nCol))
        self.samples.eta[0]   = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 5, size = self.nDat)
        self.samples.r[0]     = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0], self.samples.sigma[0],
                )
        self.am_cov_i = np.empty((self.nDat, self.nTemp, self.nCol, self.nCol))
        self.am_cov_i[:] = np.eye(self.nCol, self.nCol).reshape(1,1,self.nCol, self.nCol) * 1e-3
        self.am_mean_i = np.empty((self.nDat, self.nTemp, self.nCol))
        self.am_mean_i[:] = 0.
        self.am_cov_c = np.empty((self.nTemps, self.max_clust_count, self.nCol, self.nCol))
        self.am_mean_c = np.empty((self.nTemps, self.max_clust_count, self.nCol))
        self.am_n_c = np.zeros((self.nTemps, self.max_clust_count))
        
        self.temp_unravel = np.repeat(np.arange(self.nTemps), self.nDat)
        self.curr_iter = 0
        return

    def iter_sample(self):
        # Setup, parsing
        delta = self.curr_delta.copy()
        zeta  = self.curr_zeta.copy()
        sigma = self.curr_sigma.copy()
        mu    = self.curr_mu
        Sigma = self.curr_Sigma
        Sigma_cho = cho_factor(Sigma)
        Sigma_inv = cho_solve(Sigma_cho, np.eye(self.nCol))
        xi    = self.curr_xi
        tau   = self.curr_tau
        eta   = self.curr_eta
        r     = self.curr_r
        # Adaptive Metropolis Update
        if self.curr_iter > 300:
            self.update_am_cov()
        elif self.curr_iter == 300:
            self.update_am_cov_initial()
        else:
            pass
        # Advance the iterator
        self.curr_iter += 1

        # Compute Cluster assignments & re-index
        self.sample_delta(delta, r, zeta, sigma, eta)
        self.clean_delta_zeta_sigma(delta, zeta, sigma)
        
        self.samples.delta[self.curr_iter] = delta
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta, sigma)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_r, self.curr_delta, alpha, beta, xi, tau,
                )
        self.samples.sigma[self.curr_iter] = self.sample_sigma(
                zeta, self.curr_r, self.curr_delta, xi, tau,
                )
        curr_cluster_dummy = bincount2D_vectorized(delta) > 0
        self.samples.xi[self.curr_iter]  = self.sample_xi(self.curr_sigma, xi)
        self.samples.tau[self.curr_iter] = self.sample_tau(self.curr_sigma, self.curr_xi)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, self.curr_delta)

        # attempt swap
        if self.curr_iter >= self.swap_start:
            lp = np.zeros(self.nTemps)
            lp += dprojgamma_log_paired_yt(
                self.data.Yp, 
                zeta[self.temp_unravel, delta.ravel()],
                sigma[self.temp_unravel, delta.ravel()],
                ).sum(axis = 1)
            lp += dmvnormal_log_mx(log(zeta), mu, Sigma_cho, Sigma_inv)
            lp += dmvnormal_log_mx(mu, *self.priors.mu)
            lp += dinvwishart_log_ms(Sigma, *self.priors.Sigma)
            lp += dgamma_log_my(xi, *self.priors.xi).sum(axis = 1)
            lp += dgamma_log_my(tau, *self.priors.tau).sum(axis = 1)
            lp += dgamma_log_my(eta, *self.priors.eta)

            sw = choice(self.nTemps, 2 * self.nSwap_per, replace = False).reshape(-1,2)
            sw_alpha = (self.itl[sw.T[1]] - self.itl[sw.T[0]]) * (lp[sw.T[0]] - lp[sw.T[1]])
            logp = np.log(uniform(size = sw.shape[0]))
            for tt in sw[np.where(logp < sw_alpha)]:
                pass
        
        # write new values to log_zeta_hist

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
        conn = sql.connect(path)

        zetas  = np.vstack([
            np.hstack((np.ones((zeta.shape[0], 1)) * i, zeta))
            for i, zeta in enumerate(self.samples.zeta[nBurn :: nThin])
            ])
        sigmas = np.vstack([
            np.hstack((np.ones((sigma.shape[0], 1)) * i, sigma))
            for i, sigma in enumerate(self.samples.sigma[nBurn :: nThin])
            ])

        xis    = self.samples.xi[nBurn :: nThin]
        taus   = self.samples.tau[nBurn :: nThin]
        deltas = self.samples.delta[nBurn :: nThin]
        rs     = self.samples.r[nBurn :: nThin]
        etas   = self.samples.eta[nBurn :: nThin]

        zetas_df = pd.DataFrame(
                zetas, columns = ['iter'] + ['zeta_{}'.format(i) for i in range(self.nCol)],
                )
        sigmas_df = pd.DataFrame(
                sigmas, columns = ['iter'] + ['sigma_{}'.format(i) for i in range(self.nCol)],
                )
        alphas_df = pd.DataFrame(alphas, columns = ['alpha_{}'.format(i) for i in range(self.nCol)])
        betas_df  = pd.DataFrame(betas,  columns = ['beta_{}'.format(i)  for i in range(self.nCol)])
        xis_df    = pd.DataFrame(xis,    columns = ['xi_{}'.format(i)    for i in range(self.nCol-1)])
        taus_df   = pd.DataFrame(taus,   columns = ['tau_{}'.format(i)   for i in range(self.nCol-1)])
        deltas_df = pd.DataFrame(deltas, columns = ['delta_{}'.format(i) for i in range(self.nDat)])
        rs_df     = pd.DataFrame(rs,     columns = ['r_{}'.format(i)     for i in range(self.nDat)])
        etas_df   = pd.DataFrame({'eta' : etas})

        zetas_df.to_sql('zetas',   conn, index = False)
        sigmas_df.to_sql('sigmas', conn, index = False)
        alphas_df.to_sql('alphas', conn, index = False)
        betas_df.to_sql('betas',   conn, index = False)
        xis_df.to_sql('xis',       conn, index = False)
        taus_df.to_sql('taus',     conn, index = False)
        deltas_df.to_sql('deltas', conn, index = False)
        rs_df.to_sql('rs',         conn, index = False)
        etas_df.to_sql('etas',     conn, index = False)
        conn.commit()
        conn.close()
        return

    def set_projection(self):
        self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        return

    def __init__(
            self,
            data,
            prior_eta   = GammaPrior(2., 0.5),
            prior_mu    = (0., 4.),
            prior_Sigma = (10, 0.5),
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
        _prior_mu = NormalPrior(
            np.ones(self.nCol) * prior_mu[0],
            np.eye(self.nCol) * np.sqrt(prior_mu[1]),
            np.eye(self.nCol) / np.sqrt(prior_mu[1]),
            )
        _prior_Sigma = InvWishartPrior(
            self.nCol + prior_Sigma[0],
            np.eye(self.nCol) * prior_Sigma[1],
            )
        self.priors = Prior(prior_eta, prior_mu, prior_Sigma, prior_xi, prior_tau)
        self.set_projection()
        # self.pool = Pool(processes = 8, initializer = limit_cpu())
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = cu.counter(self.samples.delta[s], int(dmax + 1 + m))
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
        conn = sql.connect(path)

        deltas = pd.read_sql('select * from deltas;', conn).values.astype(int)
        etas   = pd.read_sql('select * from etas;', conn).values
        zetas  = pd.read_sql('select * from zetas;', conn).values
        sigmas = pd.read_sql('select * from sigmas;', conn).values
        alphas = pd.read_sql('select * from alphas;', conn).values
        betas  = pd.read_sql('select * from betas;', conn).values
        xis    = pd.read_sql('select * from xis;', conn).values
        taus   = pd.read_sql('select * from taus;', conn).values
        rs     = pd.read_sql('select * from rs;', conn).values

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = alphas.shape[1]

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.xi    = xis
        self.samples.tau   = taus
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.sigma = [sigmas[np.where(sigmas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.r     = rs
        return

    def __init__(self, path):
        self.load_data(path)
        self.data = Data(os.path.join(os.path.split(path)[0], 'empirical.csv'))
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
    model.sample(50000)
    model.write_to_disk('./test/results.db', 20000, 30)
    res = Result('./test/results.db')
    res.write_posterior_predictive('./test/postpred.csv')
    # EOL



# EOF 2
