import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
from numpy.random import choice, gamma, beta, normal, uniform
from numpy.linalg import cholesky, inv
from scipy.stats import invwishart
from scipy.special import gammaln
from collections import namedtuple
from itertools import repeat
import pandas as pd
import os
import pickle
from io import BytesIO

from samplers import DirichletProcessSampler, bincount2D_vectorized
from cov import PerObsTemperedOnlineCovariance
from cUtility import generate_indices

from data import euclidean_to_angular, euclidean_to_hypercube, Data_From_Sphere
from projgamma import GammaPrior, logd_gamma_my, logd_invwishart_ms,    \
    logd_mvnormal_mx_st, pt_logd_mvnormal_mx_st,                        \
    pt_logd_prodgamma_my_mt, pt_logd_prodgamma_paired, NormalPrior, InvWishartPrior
    
from model_sdpppgln import cluster_covariance_mat

Prior           = namedtuple('Prior', 'eta mu Sigma xi tau')

# Log Densities
def log_density_log_zeta_j(log_zeta_j, log_yj_sv, nj, Sigma_inv, mu):
    """
    log-density for log-zeta (shape parameter vector for gamma likelihood) per cluster
    with rate parameter integrated out, and summary statistics for gamma RV's
    pre-calculated. (log Y and Y summed vertically per-cluster)
    ---
    log_zeta_j : (m x d)
    log_yj_sv  : (m x d)
    nj         : (m)
    Sigma_inv  : (m x d x d)
    mu         : (m x d)
    ---
    returns:
    ld         : (m)
    """
    zeta_j = np.exp(log_zeta_j)
    ld = np.zeros(nj.shape[0])
    ld += np.einsum('md,md->m', zeta_j - 1, log_yj_sv)
    ld -= nj * np.einsum('md->m', gammaln(zeta_j))
    ld -= 0.5 * np.einsum('ml,mld,md->m', log_zeta_j - mu, Sigma_inv, log_zeta_j - mu)
    return ld

class Samples(object):
    zeta  = None
    mu    = None
    Sigma = None
    delta = None
    r     = None
    eta   = None
    ld    = None

    def __init__(self, nSamp, nDat, nCol, nTemp):
        self.zeta  = [None] * (nSamp + 1)
        self.mu    = np.empty((nSamp + 1, nTemp, nCol))
        self.Sigma = np.empty((nSamp + 1, nTemp, nCol, nCol))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nTemp, nDat))
        self.eta   = np.empty((nSamp + 1, nTemp))
        self.ld    = np.empty((nSamp + 1))
        return

class Samples_(Samples):
    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = [None] * (nSamp + 1)
        self.mu    = np.empty((nSamp + 1, nCol))
        self.Sigma = np.empty((nSamp + 1, nCol, nCol))
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
    def curr_mu(self):
        return self.samples.mu[self.curr_iter]
    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter]
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
    def sample_delta(self, delta, r, zeta, eta):
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
        log_likelihood = pt_logd_prodgamma_my_mt(Y, zeta, self.sigma_ph1)  # (n, t, J)
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
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                np.log(scratch, out = scratch)
            scratch += log_likelihood[i]
            np.nan_to_num(scratch, False, -np.inf)
            scratch -= scratch.max(axis = 1).reshape(-1,1)
            with np.errstate(under = 'ignore'):
                np.exp(scratch, out = scratch)
            np.cumsum(scratch, axis = 1, out = scratch)
            scratch /= scratch.T[-1].reshape(-1,1)
            scratch += tidx.reshape(-1,1)
            delta.T[i] = np.searchsorted(scratch.ravel(), p[i]) % self.max_clust_count
            curr_cluster_state[tidx, delta.T[i]] += 1
            cand_cluster_state[tidx, delta.T[i]] = False
        
        return
    
    # updated
    def clean_delta_zeta(self, delta, zeta):
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
    
    def am_covariance_matrices(self, delta, index):
        return self.am_Sigma.cluster_covariance(delta)[index]

    def sample_zeta(self, zeta, delta, r, mu, Sigma_inv):
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
        lYsv    = np.einsum('tnd,tnj->tjd', lY, delta_ind_mat)[idx] # logY sum vertical
        nj      = curr_cluster_state[idx]
        
        self.am_alpha[:]   = -np.inf
        self.am_alpha[idx] = self.itl[idx[0]] * (
            + log_density_log_zeta_j(lz_cand[idx], lYsv, nj, Sigma_inv[idx[0]], mu[idx[0]])
            - log_density_log_zeta_j(lz_curr[idx], lYsv, nj, Sigma_inv[idx[0]], mu[idx[0]])
            )
        keep = np.where(np.log(uniform(size = self.am_alpha.shape)) < self.am_alpha)
        zeta[keep] = np.exp(lz_cand[keep])
        return zeta

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

    def sample_r(self, delta, zeta):
        """
        delta : (t x n)
        zeta  : (t x J x d)
        sigma : (t x J x d)
        """
        As = zeta[self.temp_unravel, delta.ravel()].sum(axis = 1).reshape(self.nTemp, self.nDat)
        Bs = self.data.Yp.sum(axis = 1)[None,:]
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
        self.samples.Sigma[0] = np.eye(self.nCol) * 2
        self.samples.mu[0]    = 0.
        self.samples.eta[0]   = 40.
        self.samples.delta[0] = choice(self.max_clust_count - 20, size = (self.nTemp, self.nDat))
        self.samples.r[0]     = self.sample_r(self.samples.delta[0], self.samples.zeta[0])
        self.am_alpha     = np.zeros((self.nTemp, self.max_clust_count))
        self.candidate_zeta  = np.zeros((self.nTemp, self.max_clust_count, self.nCol))
        self.sigma_ph1 = np.ones((self.nTemp, self.max_clust_count, self.nCol))
        self.sigma_ph2 = np.ones((self.nTemp, self.nDat, self.nCol))
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
            self.sigma_ph2,
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
        # Compute prior log-density
        lpp += logd_mvnormal_mx_st(self.curr_mu, *self.priors.mu)
        lpp += logd_invwishart_ms(self.curr_Sigma, *self.priors.Sigma)
        lpp += logd_gamma_my(self.curr_eta, *self.priors.eta)
        # assemble them, record cold chain value
        self.samples.ld[self.curr_iter] = (lpl + lpp)[0]
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
        r     = self.curr_r

        # Adaptive Metropolis Update
        self.update_am_cov()
        
        # Advance the iterator
        self.curr_iter += 1

        # Sample New Candidate Clusters
        cand_clusters = np.where(bincount2D_vectorized(delta, self.max_clust_count) == 0)
        zeta[cand_clusters] = self.sample_zeta_new(mu, Sigma_cho)[cand_clusters]

        # Compute Cluster assignments & re-index
        self.sample_delta(delta, r, zeta, eta)
        self.clean_delta_zeta(delta, zeta)
        self.samples.delta[self.curr_iter] = delta

        # Compute additional parameters
        self.samples.r[self.curr_iter]     = self.sample_r(self.curr_delta, zeta)
        self.samples.zeta[self.curr_iter]  = self.sample_zeta(
                zeta, self.curr_delta, self.curr_r, mu, Sigma_inv
                )
        extant_clusters = bincount2D_vectorized(delta, self.max_clust_count) > 0
        self.samples.mu[self.curr_iter] = self.sample_mu(self.curr_zeta, Sigma_inv, extant_clusters)        
        self.samples.Sigma[self.curr_iter] = self.sample_Sigma(self.curr_zeta, self.curr_mu, extant_clusters)
        self.samples.eta[self.curr_iter] = self.sample_eta(eta, self.curr_delta)

        # attempt swap
        if self.curr_iter >= self.swap_start:
            lpl = np.zeros(self.nTemp)
            lpp = np.zeros(self.nTemp)
            Y = np.einsum('tn,nd->tnd', self.curr_r, self.data.Yp) # (t x n x d)
            lpl += pt_logd_prodgamma_paired(
                Y, 
                self.curr_zeta[self.temp_unravel, delta.ravel()].reshape(self.nTemp, self.nDat, self.nCol),
                self.sigma_ph2,
                ).sum(axis = 1)
            lpl += ((self.nCol - 1) * np.log(self.curr_r)).sum(axis = 1)
            lpl += np.einsum('tj,tj->t', 
                    pt_logd_mvnormal_mx_st(np.log(self.curr_zeta), mu, Sigma_cho, Sigma_inv), extant_clusters,
                    )
            lpp += logd_mvnormal_mx_st(self.curr_mu, *self.priors.mu)
            lpp += logd_invwishart_ms(self.curr_Sigma, *self.priors.Sigma)
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
                self.samples.mu[self.curr_iter, tt[0]], self.samples.mu[self.curr_iter, tt[1]] = \
                    self.samples.mu[self.curr_iter, tt[1]].copy(), self.samples.mu[self.curr_iter, tt[0]].copy()
                self.samples.Sigma[self.curr_iter, tt[0]], self.samples.Sigma[self.curr_iter, tt[1]] = \
                    self.samples.Sigma[self.curr_iter, tt[1]].copy(), self.samples.Sigma[self.curr_iter, tt[0]].copy()
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
        mus    = self.samples.mu[nBurn :: nThin, 0]
        Sigmas = self.samples.Sigma[nBurn :: nThin, 0]
        deltas = self.samples.delta[nBurn :: nThin, 0]
        rs     = self.samples.r[nBurn :: nThin, 0]
        etas   = self.samples.eta[nBurn :: nThin, 0]

        out = {
            'deltas' : deltas,
            'zetas'  : zetas,
            'mus'    : mus,
            'Sigmas' : Sigmas,
            'rs'     : rs,
            'etas'   : etas,
            'V'      : self.data.V,
            'logd'   : self.samples.ld
            }
        
        try:
            out['Y'] = self.data.Y
        except AttributeError:
            pass

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
            prior_Sigma = (10, 1.),
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
            np.eye(self.nCol) * prior_Sigma[1],
            )
        _prior_eta = GammaPrior(*prior_eta)
        _prior_xi  = GammaPrior(*prior_xi)
        _prior_tau = GammaPrior(*prior_tau)
        self.priors = Prior(
            _prior_eta, _prior_mu, _prior_Sigma, _prior_xi, _prior_tau,
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
            prob = ljs / ljs.sum()
            deltas = generate_indices(prob, n_per_sample)
            zeta = np.vstack((self.samples.zeta[s], new_zetas))[deltas]
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
        mus    = out['mus']
        Sigmas = out['Sigmas']
        zetas  = out['zetas']
        etas   = out['etas']
        rs     = out['rs']
        ld     = out['logd']
        
        self.nSamp = deltas.shape[0]
        self.nDat  = deltas.shape[1]
        self.nCol  = mus.shape[1]

        self.data = Data_From_Sphere(out['V'])
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass

        self.samples       = Samples_(self.nSamp, self.nDat, self.nCol)
        self.samples.delta = deltas
        self.samples.eta   = etas
        self.samples.zeta  = [zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)]
        self.samples.mu    = mus
        self.samples.Sigma = Sigmas.reshape(self.nSamp, self.nCol, self.nCol)
        self.samples.r     = rs
        self.samples.ld    = ld
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

    raw = read_csv('./datasets/ivt_updated_nov_mar.csv')
    data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 10)
    model.sample(10000)
    model.write_to_disk('./test/results.pkl', 5000, 5)
    res = Result('./test/results.pkl')
    res.write_posterior_predictive('./test/postpred.csv')

# EOF
