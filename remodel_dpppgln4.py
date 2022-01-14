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

def dprojgamma_log_my_mt(aY, aAlpha, aBeta):
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

def dinvwishart_log_ms(Sigma, psi, nu):
    ld = np.zeros(Sigma.shape[0])
    ld += 0.5 * nu * slogdet(psi)[1]
    ld -= multigammaln(nu / 2, psi.shape[-1])
    ld -= 0.5 * nu * psi.shape[-1] * log(2.)
    ld -= 0.5 * (df + Sigma.shape[-1] + 1) * slogdet(Sigma)[1]
    ld -= 0.5 * np.einsum(
            '...ii->...', np.einsum('ji,...ij->...ij', psi, inv(Sigma)),
            )
    return ld

def log_density_log_zeta_j(log_zeta_j, log_yj_sv, yj_sh, nj,
                            Sigma_cho, Sigma_inv, mu, xi, tau):
    """
    log_zeta_j : (m x d)
    log_yj_sv  : (m x d)
    yj_sh      : (m x nj)
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
        self.zeta_hist = np.empty((nSamp + 1, setup.ntemps, ))
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
    
    am_cov  = None
    am_mean = None
    max_clust_count = None

    # updated
    def sample_delta(self, delta, r, zeta, sigma, eta):
        """AI is creating summary for sample_delta

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
        
        return delta
    
    # updated
    def clean_delta_zeta_sigma(self, delta, zeta, sigma):
        for t in range(self.nTemp):
            keep, delta[t] = np.unique(delta[t], return_inverse = True)
            zeta[t][:keep.shape[0]] = zeta[t,keep]
            sigma[t][:keep.shape[0]] = sigma[t,keep]
        return delta, zeta, sigma
    
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

    def sample_zeta(self, zeta, delta, r, mu, Sigma_cho, Sigma_inv):
        """
        zeta      : (t x J x d)
        delta     : (t x n)
        r         : (t x n)
        mu        : (t x d)
        Sigma_cho : (t x d x d)
        Sigma_inv : (t x d x d)
        """
        Y = np.einsum('tn,nd->tnd', r, self.data.Yp)
        lY = np.log(Y)
        curr_cluster_state = bincount2D_vectorized(delta)
        cand_cluster_state = (curr_cluster_state == 0)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)



        pass

    def sample_zeta(self, curr_zeta, delta, r, Sigma_cho, Sigma_inv, mu):
        Y = (self.data.Yp.T * r).T
        nClust = delta.max() + 1
        Yjs = [Y[np.where(delta == j)[0]] for j in range(nClust)]
        prop_zeta = np.empty(curr_zeta.shape)
        for j in range(nClust):
            prop_zeta[j] = self.sample_zeta_j(curr_zeta[j], Yjs[j], Sigma_cho, Sigma_inv, mu)
        return prop_zeta

    def sample_zeta_j(self, curr_zeta_j, Yj, Sigma_cho, Sigma_inv, mu):
        curr_log_zeta_j = np.log(curr_zeta_j)
        curr_cov = self.localcov(curr_log_zeta_j)
        curr_cov_cho = cho_factor(curr_cov)
        curr_cov_inv = cho_solve(curr_cov_cho, np.eye(self.nCol))

        prop_log_zeta_j = curr_log_zeta_j + np.triu(curr_cov_cho[0]) @ normal(size = self.nCol)
        prop_cov = self.localcov(curr_log_zeta_j)
        prop_cov_cho = cho_factor(prop_cov)
        prop_cov_inv = cho_solve(prop_cov_cho, np.eye(self.nCol))

        curr_lp = log_density_log_zeta_j(
            curr_log_zeta_j, Yj, Sigma_cho, Sigma_inv, mu, self.priors.sigma,
            ) * self.inv_temper_temp
        prop_lp = log_density_log_zeta_j(
            prop_log_zeta_j, Yj, Sigma_cho, Sigma_inv, mu, self.priors.sigma,
            ) * self.inv_temper_temp
        pc_ld = log_density_mvnormal(curr_log_zeta_j, prop_log_zeta_j, np.triu(prop_cov_cho[0]), prop_cov_inv)
        cp_ld = log_density_mvnormal(prop_log_zeta_j, curr_log_zeta_j, np.triu(curr_cov_cho[0]), curr_cov_inv)

        if log(uniform()) < prop_lp + pc_ld - curr_lp - cp_ld:
            return np.exp(prop_log_zeta_j)
        return curr_zeta_j
    
    def sample_mu(self, zeta, Sigma_inv):
        n = zeta.shape[0] # number of clusters
        lzbar = np.log(zeta).mean(axis = 0)
        _Sigma = cho_solve(
            cho_factor(n * Sigma_inv * self.inv_temper_temp + self.priors.mu.SInv),
            np.eye(self.nCol),
            )
        _mu = _Sigma @ (
            + n * lzbar @ Sigma_inv * self.inv_temper_temp
            + self.priors.mu.mu @ self.priors.mu.Sinv
            )
        return _mu + cholesky(_Sigma) @ normal(size = self.nCol)
    
    def sample_Sigma(self, zeta, mu):
        n = zeta.shape[0]
        diff = np.log(zeta) - mu
        C = sum([np.outer(diff[i], diff[i]) for i in range(zeta.shape[0])])
        _psi = self.priors.Sigma.psi + C * self.inv_temper_temp
        _nu  = self.priors.Sigma.nu + n * self.inv_temper_temp
        return invwishart.rvs(df = _nu, scale = _psi)

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

    def sample_sigma(self, zeta, r, delta, xi, tau):
        Y = r.reshape(-1, 1) * self.data.Yp
        args = zip(
            zeta,
            [Y[np.where(delta == j)[0]] for j in range(zeta.shape[0])],
            repeat(xi),
            repeat(tau),
            )
        res = map(update_sigma_j_wrapper, args)
        # res = self.pool.map(update_sigma_j_wrapper, args)
        return np.array(list(res))

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
        self.am_cov = np.empty((self.nDat, self.nTemp, self.nCol, self.nCol))
        self.am_cov[:] = np.eye(self.nCol, self.nCol).reshape(1,1,self.nCol, self.nCol)
        self.am_mean = np.empty((self.nDat, self.nTemp, self.nCol))
        self.am_mean[:] = 0.
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
        logConstant = (zeta * np.log(sigma)).sum(axis = 1) - gammaln(zeta).sum(axis = 1)
        # pre-generate uniforms to inverse-cdf sample cluster indices
        unifs   = uniform(size = self.nDat)
        # provide a cluster index probability placeholder, so it's not being re-allocated for every sample
        scratch = np.empty(self.max_clust_count)
        for i in range(self.nDat):
            # Sample new cluster indices
            delta[i] = self.sample_delta_i(curr_cluster_state, cand_cluster_state, eta, 
                            delta[i], unifs[i], r[i] * self.data.Yp[i], zeta, sigma, logConstant, scratch)
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
        alphas = self.samples.alpha[nBurn :: nThin]
        betas  = self.samples.beta[nBurn :: nThin]
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
        self.priors = Prior(prior_eta, prior_alpha, prior_beta, prior_xi, prior_tau)
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
