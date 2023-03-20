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

from cov import PerObsTemperedOnlineCovariance
from samplers import ParallelTemperingStickBreakingSampler, bincount2D_vectorized,  \
        pt_dp_sample_cluster_bgsb, pt_dp_sample_chi_bgsb, pt_logd_gem_mx_st,        \
        pt_dp_sample_concentration_bgsb
from cUtility import generate_indices
from cProjgamma import sample_alpha_k_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube, Data_From_Sphere
from projgamma import GammaPrior, pt_logd_prodgamma_my_mt, pt_logd_prodgamma_paired, \
        pt_logd_prodgamma_my_st, pt_logd_mvnormal_mx_st, logd_mvnormal_mx_st,\
        logd_gamma_my, logd_invwishart_ms, \
        pt_logd_projgamma_my_mt_inplace_unstable, pt_logd_projgamma_paired_yt

NormalPrior     = namedtuple('NormalPrior', 'mu SCho SInv')
InvWishartPrior = namedtuple('InvWishartPrior', 'nu psi')
Prior = namedtuple('Prior', 'eta mu Sigma xi tau')

# Log Densities

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
    mu    = None
    Sigma = None
    delta = None
    eta   = None
    chi   = None
    logd  = None

    def __init__(self, nSamp, nDat, nCol, nTemp, nClust):
        self.zeta  = np.empty((nSamp + 1, nTemp, nClust, nCol))
        self.log_zeta_hist = np.empty((nSamp + 1, nDat, nTemp, nCol))
        self.mu    = np.empty((nSamp + 1, nTemp, nCol))
        self.Sigma = np.empty((nSamp + 1, nTemp, nCol, nCol))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.eta   = np.empty((nSamp + 1, nTemp))
        self.chi   = np.empty((nSamp + 1, nTemp, nClust))
        self.logd  = np.empty((nSamp + 1))
        return

class Samples_(Samples):
    def __init__(self, nSamp, nDat, nCol, nClust):
        self.zeta  = np.empty((nSamp + 1, nClust, nCol))
        self.chi   = np.empty((nSamp + 1, nClust))
        self.mu    = np.empty((nSamp + 1, nCol))
        self.Sigma = np.empty((nSamp + 1, nCol, nCol))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.eta   = np.empty((nSamp + 1))
        self.logd  = np.empty((nSamp + 1))
        return

class Chain(ParallelTemperingStickBreakingSampler):
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
    def curr_chi(self):
        return self.samples.chi[self.curr_iter]
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter]

    # Adaptive Metropolis Placeholders
    am_Sigma = None
    am_scale = None
    max_clust_count = None
    swap_attempts = None
    swap_succeeds = None

    def sample_delta(self, chi, zeta):
        log_likelihood = self.log_delta_likelihood(zeta)
        delta = pt_dp_sample_cluster_bgsb(chi, log_likelihood)
        return delta
    
    def sample_zeta_new(self, mu, Sigma_chol):
        """
        mu         : (t x d)
        Sigma_chol : (t x d x d)
        out        : (t x J x d) # Modified in-place
        """
        sizes = (self.nTemp, self.max_clust_count, self.nCol)
        out = np.empty(sizes)
        np.einsum('tzy,tjy->tjz', np.triu(Sigma_chol),  
                  normal(size = sizes), out = out)
        out += mu[:,None,:]
        np.exp(out, out = out)
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
            normal(size = (idx[0].shape[0], self.nCol)),
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
        zcurr[cand_cluster_state] = self.sample_zeta_new(mu, Sigma_chol)[cand_cluster_state]
        return zcurr

    def sample_zeta_old(self, zeta, delta, r, xi, tau, mu, Sigma_inv):
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

    def sample_chi(self, delta, eta):
        return pt_dp_sample_chi_bgsb(delta, eta, self.max_clust_count)

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

    def sample_eta(self, chi):
        """
        curr_eta : (t)
        delta    : (t x n)
        """
        # g = beta(curr_eta + 1, self.nDat)
        # aa = self.priors.eta.a + delta.max(axis = 1) + 1
        # bb = self.priors.eta.b - np.log(g)
        # eps = (aa - 1) / (self.nDat  * bb + aa - 1)
        # id = uniform(self.nTemp) > eps
        # aaa = aa * id + (aa - 1) * (1 - id)
        # aaa = choice((aa, aa - 1), 1, p = (eps, 1 - eps))
        # return gamma(shape = aaa, scale = 1 / bb)
        return pt_dp_sample_concentration_bgsb(chi, *self.priors.eta)

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

        sw_alpha += (lpl[sw.T[1]] - lpl[sw.T[0]])
        sw_alpha += (lpp[sw.T[1]] - lpp[sw.T[0]])
        sw_alpha *= (self.itl[sw.T[1]] - self.itl[sw.T[0]])
        # sw_alpha *= self.itl[sw.T[0]] - self.itl[sw.T[1]]
        # sw_alpha += lpp[sw.T[1]] - lpp[sw.T[0]]

        logp = np.log(uniform(size = sw_alpha.shape))
        for tt in sw[np.where(logp < sw_alpha)[0]]:
            # report successful swap
            self.swap_succeeds[tt[0],tt[1]] += 1
            self.swap_succeeds[tt[1],tt[0]] += 1
            # do the swap
            self.samples.zeta[ci][tt[0]], self.samples.zeta[ci][tt[1]] = \
                self.samples.zeta[ci][tt[1]].copy(), self.samples.zeta[ci][tt[0]].copy()
            self.samples.mu[ci][tt[0]], self.samples.mu[ci][tt[1]] = \
                self.samples.mu[ci][tt[1]].copy(), self.samples.mu[ci][tt[0]].copy()
            self.samples.Sigma[ci][tt[0]], self.samples.Sigma[ci][tt[1]] = \
                self.samples.Sigma[ci][tt[1]].copy(), self.samples.Sigma[ci][tt[0]].copy()
            self.samples.delta[ci][tt[0]], self.samples.delta[ci][tt[1]] = \
                self.samples.delta[ci][tt[1]].copy(), self.samples.delta[ci][tt[0]].copy()
            self.samples.chi[ci][tt[0]], self.samples.chi[ci][tt[1]] = \
                self.samples.chi[ci][tt[1]].copy(), self.samples.chi[ci][tt[0]].copy()
            self.samples.eta[ci][tt[0]], self.samples.eta[ci][tt[1]] = \
                self.samples.eta[ci][tt[1]].copy(), self.samples.eta[ci][tt[0]].copy()
        return

    def log_delta_likelihood(self, zeta):
        out = np.zeros((self.nDat, self.nTemp, self.max_clust_count))
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pt_logd_projgamma_my_mt_inplace_unstable(
                out, self.data.Yp, zeta, self.sigma_ph1,
                )
        np.nan_to_num(out)
        return out
    
    def log_zeta_likelihood(self, zeta, delta, delta_ind_mat):
        out = np.zeros((self.nTemp, self.max_clust_count))
        zetas = zeta[
            self.temp_unravel, delta.ravel(),
            ].reshape(self.nTemp, self.nDat, self.nCol)
        out += np.einsum(
            'tn,tnj->tj',
            pt_logd_projgamma_paired_yt(self.data.Yp, zetas, self.sigma_ph2),
            delta_ind_mat,
            )
        return out

    def log_logzeta_prior(self, logzeta, mu, Sigma_chol, Sigma_inv):
        return pt_logd_mvnormal_mx_st(logzeta, mu, Sigma_chol, Sigma_inv)

    def log_tempering_likelihood(self):
        curr_zeta = self.curr_zeta[
            self.temp_unravel, self.curr_delta.ravel()
            ].reshape(self.nTemp, self.nDat, self.nCol)
        out = np.zeros(self.nTemp)
        out += pt_logd_projgamma_paired_yt(
            self.data.Yp,
            curr_zeta,
            self.sigma_ph2,
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
        out += pt_logd_gem_mx_st(self.curr_chi, self.curr_eta, 0.)
        out += logd_gamma_my(self.curr_eta, *self.priors.eta)
        return out
    
    def initialize_sampler(self, ns):
        zeta_sizes = (self.nTemp, self.max_clust_count, self.nCol)
        self.sigma_ph1 = np.ones((self.nTemp, self.max_clust_count, self.nCol))
        self.sigma_ph2 = np.ones((self.nTemp, self.nDat, self.nCol))
        # Samples
        self.samples = Samples(ns, self.nDat, self.nCol, self.nTemp, self.max_clust_count)
        self.samples.zeta[0]  = np.exp(normal(size = zeta_sizes))
        self.samples.mu[0]    = 0.
        self.samples.Sigma[0] = np.eye(self.nCol) * 2
        self.samples.eta[0]   = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 20, size = (self.nTemp, self.nDat))
        self.samples.chi[0]   = 0.05 # beta(0.1, 0.5, size = self.samples.chi[0].shape)
        # iterator
        self.curr_iter = 0
        # Parallel Tempering Related
        self.swap_attempts = np.zeros((self.nTemp, self.nTemp))
        self.swap_succeeds = np.zeros((self.nTemp, self.nTemp))
        return

    def log_logd(self):
        self.samples.logd[self.curr_iter] += self.log_tempering_likelihood()[0]
        self.samples.logd[self.curr_iter] += self.log_tempering_prior()[0]
        return

    def iter_sample(self):
        # Setup, parsing
        delta = self.curr_delta.copy()
        zeta  = self.curr_zeta.copy()
        mu    = self.curr_mu
        Sigma = self.curr_Sigma
        Sigma_cho = cholesky(Sigma)
        Sigma_inv = inv(Sigma)
        eta   = self.curr_eta
        chi   = self.curr_chi

        # Adaptive Metropolis Update
        self.update_am_cov()

        # Advance the iterator
        self.curr_iter += 1
        ci = self.curr_iter

        # if self.curr_iter == 5:
        #     raise

        # Sample New Candidate Clusters
        cand_clusters = np.where(bincount2D_vectorized(delta, self.max_clust_count) == 0)
        zeta[cand_clusters] = self.sample_zeta_new(mu, Sigma_cho)[cand_clusters]

        # Compute Cluster assignments & re-index
        self.samples.delta[ci] = self.sample_delta(chi, zeta)
        self.samples.chi[ci] = self.sample_chi(self.curr_delta, eta)
        self.samples.eta[ci] = self.sample_eta(self.curr_chi)

        # Compute additional parameters
        extant_clusters = bincount2D_vectorized(delta, self.max_clust_count) > 0
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_delta, mu, Sigma_cho, Sigma_inv
                )
        self.samples.mu[self.curr_iter] = self.sample_mu(self.curr_zeta, Sigma_inv, extant_clusters)        
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_zeta, self.curr_mu, extant_clusters)

        # attempt swap
        if self.curr_iter >= self.swap_start:
            self.try_tempering_swap()
        
        self.log_logd()
        return

    def write_to_disk(self, path, nBurn, nThin = 1):
        if type(path) is str:
            folder = os.path.split(path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            if os.path.exists(path):
                os.remove(path)

        zetas = self.samples.zeta[nBurn :: nThin, 0]
        mus    = self.samples.mu[nBurn :: nThin, 0]
        Sigmas = self.samples.Sigma[nBurn :: nThin, 0]
        deltas = self.samples.delta[nBurn :: nThin, 0]
        etas   = self.samples.eta[nBurn :: nThin, 0]
        chis   = self.samples.chi[nBurn :: nThin, 0]

        out = {
            'nclust' : self.max_clust_count,
            'zetas'  : zetas,
            'mus'    : mus,
            'Sigmas' : Sigmas,
            'deltas' : deltas,
            'etas'   : etas,
            'V'      : self.data.V,
            'logd'   : self.samples.logd[1:],
            'chis'   : chis,
            'nburn'  : nBurn - 1,
            'nthin'  : nThin,
            'swap_succeed' : self.swap_succeeds,
            'swap_attempt' : self.swap_attempts,
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
            prior_eta   = (2., 0.5),
            prior_mu    = (0., 4.),
            prior_Sigma = (10, 1.),
            prior_xi    = (1., 1.),
            prior_tau   = (2., 2.),
            p           = 10,
            max_clust_count = 300,
            ntemps      = 5,
            stepping    = 1.3,
            ):
        self.data = data
        self.max_clust_count = max_clust_count
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.set_projection()
        
        # Setting Priors
        _prior_eta = GammaPrior(*prior_eta)
        _prior_xi  = GammaPrior(*prior_xi)
        _prior_tau = GammaPrior(*prior_tau)
        _prior_mu = NormalPrior(
            np.ones(self.nCol) * prior_mu[0],
            np.eye(self.nCol) * np.sqrt(prior_mu[1]),
            np.eye(self.nCol) / np.sqrt(prior_mu[1]),
            )
        _prior_Sigma = InvWishartPrior(
            self.nCol + prior_Sigma[0],
            np.eye(self.nCol) * prior_Sigma[1] * (self.nCol + prior_Sigma[0]),
            )
        self.priors = Prior(
            _prior_eta, _prior_mu, _prior_Sigma, _prior_xi, _prior_tau,
            )

        # Parallel Tempering
        self.nTemp = ntemps
        self.itl = 1 / stepping**np.arange(ntemps)
        self.temp_unravel = np.repeat(np.arange(self.nTemp), self.nDat)
        self.nSwap_per = self.nTemp // 2
        self.swap_start = 100

        # Adaptive Metropolis
        self.am_Sigma = PerObsTemperedOnlineCovariance(
            self.nTemp, self.nDat, self.nCol, self.max_clust_count,
            )
        self.am_scale = 2.38**2 / self.nCol
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            chi = self.samples.chi[s]
            prob = np.zeros(chi.shape[0])
            prob += np.log(np.hstack((chi[:-1],(1,))))
            prob += np.hstack(((0,), np.log(1 - chi[:-1]).cumsum()))
            np.exp(prob, out = prob)
            deltas = generate_indices(prob, n_per_sample)
            zeta = self.samples.zeta[s][deltas]
            new_gammas.append(gamma(shape = zeta))
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
        mus    = out['mus']
        Sigmas = out['Sigmas']
        etas   = out['etas']
        chis   = out['chis']
        
        self.data = Data_From_Sphere(out['V'])
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass
        
        self.logd  = out['logd']
        self.nBurn = out['nburn']
        self.nThin = out['nthin']
        self.swap_attempts = out['swap_attempt']
        self.swap_succeeds = out['swap_succeed']

        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = mus.shape[1]

        self.samples       = Samples_(self.nSamp, self.nDat, self.nCol, out['nclust'])
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.zeta  = zetas
        self.samples.Sigma = Sigmas
        self.samples.chi   = chis
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
    model.sample(50000)
    model.write_to_disk('./test/results.pkl', 40000, 20)
    res = Result('./test/results.pkl')
    res.write_posterior_predictive('./test/postpred.csv')


# EOF
