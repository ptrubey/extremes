"""
Model description for Pitman-Yor process Mixture of Projected Gammas on unit p-sphere
---
PG is restricted (Rate parameter := 1)
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
from scipy.special import gammaln
from io import BytesIO
import sparse

from samplers import ParallelTemperingStickBreakingSampler, GEMPrior,           \
    bincount2D_vectorized, pt_logd_gem_mx_st_fixed,                             \
    pt_py_sample_cluster_bgsb_fixed, pt_py_sample_chi_bgsb_fixed,               \
    py_sample_cluster_bgsb_fixed
from data import euclidean_to_psphere, euclidean_to_hypercube, Data_From_Sphere
from projgamma import GammaPrior, UniNormalPrior, logd_gamma_my,                \
    pt_logd_gamma_my, pt_logd_projgamma_my_mt_inplace_unstable,                 \
    pt_logd_projgamma_paired_yt,pt_logd_prodgamma_my_st, pt_logd_mixprojgamma,  \
    pt_logd_lognormal_my, logd_lognormal_my, logd_invgamma_my, logd_normal_my

Prior = namedtuple('Prior', 'mu sigma xi tau chi')

class Samples(object):
    alpha = None # Cluster shape
    beta  = None # Cluster Rate
    delta = None # cluster identtifier
    mu    = None # centering (alpha) log-mean 
    sigma = None # centering (alpha) log-sd
    xi    = None # centering (beta) shape
    tau   = None # centering (beta) rate
    chi   = None # Stick-breaking unnormalized weights
    r     = None # latent radius (per obs)
    ld    = None # log-density of cold chain

    def __init__(self, nSamp, nDat, nCol, nTemp, nClust):
        self.alpha = np.empty((nSamp + 1, nTemp, nClust, nCol))
        self.beta  = np.empty((nSamp + 1, nTemp, nClust, nCol))
        self.mu    = np.empty((nSamp + 1, nTemp, nCol))
        self.sigma = np.empty((nSamp + 1, nTemp, nCol))
        self.xi    = np.empty((nSamp + 1, nTemp, nCol - 1))
        self.tau   = np.empty((nSamp + 1, nTemp, nCol - 1))
        self.r     = np.empty((nSamp + 1, nTemp, nDat))
        self.chi   = np.empty((nSamp + 1, nTemp, nClust - 1))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.ld    = np.zeros((nSamp + 1))
        return
    
class Samples1T(object):
    alpha = None # Cluster shape
    beta  = None # Cluster Rate
    delta = None # cluster identtifier
    mu    = None # centering (alpha) log-mean 
    sigma = None # centering (alpha) log-sd
    xi    = None # centering (beta) shape
    tau   = None # centering (beta) rate
    chi   = None # Stick-breaking unnormalized weights
    r     = None # latent radius (per obs)
    ld    = None # log-density of cold chain

    def __init__(self, nSamp, nDat, nCol, nClust):
        self.alpha = np.empty((nSamp + 1, nClust, nCol))
        self.beta  = np.empty((nSamp + 1, nClust, nCol))
        self.mu    = np.empty((nSamp + 1, nCol))
        self.sigma = np.empty((nSamp + 1, nCol))
        self.xi    = np.empty((nSamp + 1, nCol - 1))
        self.tau   = np.empty((nSamp + 1, nCol - 1))
        self.r     = np.empty((nSamp + 1, nDat))
        self.chi   = np.empty((nSamp + 1, nClust - 1))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.ld    = np.zeros((nSamp + 1))
        return

class Chain(ParallelTemperingStickBreakingSampler):
    """
    Pitman Yor Mixture of Projected Gammas
    ---
    Independent Log-normal/Gamma centering disribution

    
    
    """
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
    def curr_mu(self):
        return self.samples.mu[self.curr_iter]
    @property
    def curr_sigma(self):
        return self.samples.sigma[self.curr_iter]    
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
    def curr_chi(self):
        return self.samples.chi[self.curr_iter]

    # Constants
    nClust  = None          # Stick-breaking truncation point
    nCol    = None          # Dimensionality
    nDat    = None          # Number of observations 
    
    # Parallel Tempering
    nTemp   = None          # Number of Parallel Tempering chains
    swap_attempts = None    # Parallel tempering swap attempts (count)
    swap_succeeds = None    # Parallel tempering swap success  (count)
    itl     = None          # Parallel tempering inverse temp ladder
    tidx    = None          # Parallel Tempering Temp Index : range(self.nTemp)
    
    # Placeholders
    _placeholder_sigma_1 = None # placeholder, ones(nTemp x nClust x nCol)
    _placeholder_sigma_2 = None # Placeholder, ones(nTemp x nDat x nCol)

    # Scratch arrays
    _scratch_dmat  = None # bool (nDat, nTemp, nClust)
    _scratch_delta = None # real (nDat, nTemp, nClust)
    _scratch_alpha = None # real (nTemp, nClust, nCol)
    _curr_cluster_state = None  # int  (nTemp, nClust)
    _cand_cluster_state = None  # bool (nTemp, nClust)
    _extant_clust_state = None  # bool (nTemp, nClust)

    def log_delta_likelihood(self, alpha, beta):
        """
        inputs:
            alpha : (t, j, d)
            sigma : (t, j, d)
        outputs:
            out   : (n, t, j)
        """
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pt_logd_projgamma_my_mt_inplace_unstable(
                self._scratch_delta, self.data.Yp, alpha, beta,
                )
        np.nan_to_num(self._scratch_delta, False, -np.inf)
        self._scratch_delta *= self.itl[None,:,None]
        return

    def sample_delta(self, chi, alpha, beta):
        self._scratch_delta[:] = 0.
        self.log_delta_likelihood(alpha, beta)
        return pt_py_sample_cluster_bgsb_fixed(chi, self._scratch_delta)
    
    def sample_chi(self, delta):
        chi = pt_py_sample_chi_bgsb_fixed(
            delta, *self.priors.chi, trunc = self.nClust,
            )
        return chi

    def sample_alpha_new(self, mu, sigma):
        out = normal(
            loc = mu, scale = sigma, 
            size = (self.nClust, self.nTemp, self.nCol),
            ).swapaxes(0, 1)
        return np.exp(out)

    def log_logalpha_posterior(self, logalpha, n, sy, sly, mu, sigma, xi, tau):
        alpha = np.exp(logalpha)
        out = np.zeros(logalpha.shape)

        anew = n[..., None] * alpha[..., 1:] + xi[:, None]
        bnew = sy[..., 1:] + tau[:, None]

        out += (alpha - 1) * sly
        out -= n[:,:,None] * gammaln(alpha)
        out[..., 1:] += gammaln(anew)
        out[..., 1:] -= anew * np.log(bnew)
        out *= self.itl[:,None,None]
        out -= 0.5 * ((logalpha - mu[:,None]) / sigma[:,None])**2
        return out

    def sample_alpha(self, alpha, delta, r, mu, sigma, xi, tau):
        Y    = r[:,:,None] * self.data.Yp[None] # t,n,d
        dmat = delta[:,:,None] == range(self.nClust)
        try:
            dmatS = sparse.COO.from_numpy(dmat)
            slY   = sparse.einsum('tnj,tnd->tjd', dmatS, np.log(Y)).todense()
            sY    = sparse.einsum('tnj,tnd->tjd', dmatS, Y).todense()
            n     = sparse.einsum('tnj->tj', dmatS).todense()
        except ValueError:
            slY   = np.einsum('tnj,tnd->tjd', dmat, np.log(Y))
            sY    = np.einsum('tnj,tnd->tjd', dmat, Y)
            n     = np.einsum('tnj->tj', dmat)
        
        with np.errstate(divide = 'ignore'):
            acurr  = alpha.copy()
            lacurr = np.log(acurr)
            lacand = lacurr.copy()
            lacand += normal(scale = 0.1, size = lacand.shape)
            acand  = np.exp(lacand)
        
        idx = np.where(self._extant_clust_state)
        ndx = np.where(self._cand_cluster_state)
        
        self._scratch_alpha[:] = -np.inf
        self._scratch_alpha[idx] = 0.
        
        self._scratch_alpha += self.log_logalpha_posterior(
            lacand, n, sY, slY, mu, sigma, xi, tau,
            )
        self._scratch_alpha -= self.log_logalpha_posterior(
            lacurr, n, sY, slY, mu, sigma, xi, tau,
            )

        keep = np.where(np.log(uniform(size = acurr.shape)) < self._scratch_alpha)
        acurr[keep] = acand[keep]
        acurr[ndx] = self.sample_alpha_new(mu, sigma)[ndx]
        return(acurr)

    def sample_beta(self, alpha, delta, r, xi, tau):
        Y    = r[:,:,None] * self.data.Yp[None] # t,n,d
        dmat = delta[:,:,None] == range(self.nClust)
        try:
            dmatS = sparse.COO.from_numpy(dmat)
            sY    = sparse.einsum('tnj,tnd->tjd', dmatS, Y).todense()
            n     = sparse.einsum('tnj->tj', dmatS).todense()
        except ValueError:
            sY    = np.einsum('tnj,tnd->tjd', dmat, Y)
            n     = np.einsum('tnj->tj', dmat)
        
        anew = n[..., None] * alpha[..., 1:] + xi[:,None]
        bnew = sY[..., 1:] + tau[:, None]

        beta = np.ones(alpha.shape)
        beta[..., 1:] = gamma(shape = anew, scale = 1 / bnew)
        return beta

    def sample_mu(self, alpha, sigma):
        labar = np.log(alpha).mean(axis = 1)
        snew  = np.sqrt(1 / (1 / self.priors.mu.sigma**2 + self.nClust / sigma**2))
        mnew  = self.priors.mu.sigma**2 * labar + sigma**2 * self.priors.mu.mu
        mnew /= sigma**2 / self.nClust + self.priors.mu.sigma**2
        return normal(loc = mnew, scale = snew)
    
    def sample_sigma(self, alpha, mu):
        anew = self.priors.sigma.a + self.nClust / 2
        bnew = self.priors.sigma.b + 0.5 * ((np.log(alpha) - mu[:,None])**2).sum(axis = 1)
        return np.sqrt(1 / gamma(anew, scale = 1 / bnew))

    def log_logxi_posterior(self, logxi, sB, slB):
        xi  = np.exp(logxi)
        out = np.zeros(logxi.shape)
        
        out += (xi - 1) * slB
        out -= self.nClust * gammaln(xi)
        out += gammaln(self.nClust * xi + self.priors.tau.a)
        out -= (self.nClust * xi + self.priors.tau.a) * np.log(sB + self.priors.tau.b)
        out -= 0.5 * ((logxi - self.priors.xi.mu) / self.priors.xi.sigma)**2
        return out
    
    def sample_xi(self, xi, beta):
        # n = self._extant_clust_state.sum(axis = 1)
        xcurr = xi.copy()
        lxcurr = np.log(xcurr)
        lxcand = lxcurr.copy()
        lxcand += normal(scale = 0.1, size = lxcand.shape)

        sb = beta[..., 1:].sum(axis = 1)
        slb = np.log(beta[..., 1:]).sum(axis = 1)

        logp = np.zeros(xcurr.shape)
        logp += self.log_logxi_posterior(lxcand, sb, slb)
        logp -= self.log_logxi_posterior(lxcurr, sb, slb)
        
        keep = np.where(np.log(uniform(size = logp.shape)) < logp)
        xcurr[keep] = np.exp(lxcand[keep])
        return xcurr

    def sample_tau(self, xi, beta):
        sb = beta[...,1:].sum(axis = 1)
        shape = self.nClust * xi + self.priors.tau.a
        rate = sb + self.priors.tau.b
        return gamma(shape = shape, scale = 1 / rate)

    def sample_r(self, delta, alpha, beta):
        shape = alpha[self.temp_unravel, delta.ravel()].reshape(
                    self.nTemp, self.nDat, self.nCol,
                    ).sum(axis = -1)
        rate  = np.einsum(
            'tnd,nd->tn',
            beta[self.temp_unravel, delta.ravel()].reshape(
                        self.nTemp, self.nDat, self.nCol,
                        ),
            self.data.Yp,
            )
        return gamma(shape = shape, scale = 1 / rate)
    
    def initialize_sampler(self, ns):
        """ Initialize the sampler """
        # Initialize Placeholders (for restricted gamma):
        # self._placeholder_sigma_1 = np.ones((self.nTemp, self.nClust, self.nCol))
        # self._placeholder_sigma_2 = np.ones((self.nTemp, self.nDat, self.nCol))
        # # Initialize storage
        # self._scratch_dmat  = np.zeros((self.nTemp, self.nDat, self.nClust), dtype = bool)
        self._scratch_delta = np.zeros((self.nDat, self.nTemp, self.nClust))
        self._scratch_alpha = np.zeros((self.nTemp, self.nClust, self.nCol))
        # Initialize Samples
        self.samples = Samples(ns, self.nDat, self.nCol, self.nTemp, self.nClust)
        self.samples.alpha[0] = gamma(
            shape = 2, scale = 2, size = (self.nTemp, self.nClust, self.nCol),
            )
        self.samples.beta[..., 0] = 1.
        self.samples.beta[0][..., 1:] = gamma(
            shape = 3, scale = 1 / 3, size = (self.nTemp, self.nClust, self.nCol - 1),
            )
        self.samples.mu[0] = normal(loc = 0, scale = 1, size = (self.nTemp, self.nCol))
        self.samples.sigma[0] = np.sqrt(1 / gamma(shape = 2, scale = 1 / 2))
        self.samples.xi[0] = gamma(
            shape = 2., scale = 1 / 2., size = (self.nTemp, self.nCol - 1),
            ) + 1
        self.samples.tau[0] = gamma(
            shape = 2., scale = 1 / 2., size = (self.nTemp, self.nCol - 1),
            )
        self.samples.delta[0] = choice(
            self.nClust, size = (self.nTemp, self.nDat),
            )
        self.samples.r[0] = self.sample_r(
            self.samples.delta[0], self.samples.alpha[0], self.samples.beta[0],
            )
        self.samples.chi[0] = self.sample_chi(self.samples.delta[0])
        # Parallel Tempering related
        self.swap_attempts = np.zeros((self.nTemp, self.nTemp))
        self.swap_succeeds = np.zeros((self.nTemp, self.nTemp))
        # Initialize the iterator
        self.curr_iter = 0
        # Initialize the cluster states
        self._curr_cluster_state = np.zeros((self.nTemp, self.nClust), dtype = int)
        self._cand_cluster_state = np.zeros((self.nTemp, self.nClust), dtype = bool)
        self._extant_clust_state = np.zeros((self.nTemp, self.nClust), dtype = bool)
        self.update_cluster_state()
        return

    def log_likelihood(self):
        ll = np.zeros(self.nTemp)
        ll += pt_logd_mixprojgamma(
            self.data.Yp, 
            self.curr_alpha, 
            self.curr_beta, 
            self.curr_chi,
            ).sum(axis = 0)
        ll += pt_logd_gem_mx_st_fixed(self.curr_chi, *self.priors.chi)
        # ll += pt_logd_gamma_my(self.curr_alpha, self.curr_xi, self.curr_tau).sum(axis = 1)
        return ll

    def log_prior(self):
        lp = np.zeros(self.nTemp)
        lp += pt_logd_lognormal_my(self.curr_alpha, self.curr_mu, self.curr_sigma).sum(axis = (1,2))
        lp += pt_logd_gamma_my(self.curr_beta[..., 1:], self.curr_xi, self.curr_tau).sum(axis = -1)
        lp += logd_lognormal_my(self.curr_xi, *self.priors.xi).sum(axis = -1)
        lp += logd_gamma_my(self.curr_tau, *self.priors.tau).sum(axis = -1)
        lp += logd_normal_my(self.curr_mu, *self.priors.mu).sum(axis = -1)
        lp += logd_invgamma_my(self.curr_sigma, *self.priors.sigma).sum(axis = -1)        
        return lp

    def record_log_density(self):
        ll = self.log_likelihood()
        lp = self.log_prior()
        self.samples.ld[self.curr_iter] = (ll + lp)[0]

    def update_cluster_state(self):
        self._curr_cluster_state[:] = bincount2D_vectorized(self.curr_delta, self.nClust)
        self._extant_clust_state[:] = (self._curr_cluster_state > 0)
        self._cand_cluster_state[:] = (~self._curr_cluster_state)
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        alpha = self.curr_alpha
        beta  = self.curr_beta
        mu    = self.curr_mu
        sigma = self.curr_sigma
        xi    = self.curr_xi
        tau   = self.curr_tau
        r     = self.curr_r
        chi   = self.curr_chi

        # Advance the iterator
        self.curr_iter += 1
        ci = self.curr_iter

        # Update cluster assignments
        self.samples.delta[ci] = self.sample_delta(chi, alpha, beta)
        self.samples.chi[ci] = self.sample_chi(self.curr_delta)
        self.update_cluster_state()
        self.samples.alpha[ci] = self.sample_alpha(
            alpha, self.curr_delta, r, mu, sigma, xi, tau,
            )
        self.samples.beta[ci] = self.sample_beta(
            alpha, self.curr_delta, r, xi, tau,
            )
        self.samples.r[ci] = self.sample_r(
            self.curr_delta, self.curr_alpha, self.curr_beta,
            )
        # self.samples.mu[ci] = self.sample_mu(self.curr_alpha, sigma)
        self.samples.mu[ci] = 0.
        self.samples.sigma[ci] = self.sample_sigma(self.curr_alpha, self.curr_mu)
        self.samples.xi[ci] = self.sample_xi(xi, self.curr_beta)
        self.samples.tau[ci] = self.sample_tau(self.curr_xi, self.curr_beta)

        if self.curr_iter > self.swap_start:
            self.try_tempering_swap()
            self.update_cluster_state()

        self.record_log_density()
        return
    
    def write_to_disk(self, path, nBurn, nThin = 1):
        if type(path) is str:
            folder = os.path.split(path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            if os.path.exists(path):
                os.remove(path)
        
        alphas = self.samples.alpha[nBurn :: nThin, 0]
        betas  = self.samples.beta[nBurn :: nThin, 0]
        mus    = self.samples.mu[nBurn :: nThin, 0]
        sigmas = self.samples.sigma[nBurn :: nThin, 0]
        xis    = self.samples.xi[nBurn :: nThin, 0]
        taus   = self.samples.tau[nBurn :: nThin, 0]
        deltas = self.samples.delta[nBurn :: nThin, 0]
        chis   = self.samples.chi[nBurn :: nThin, 0]
        rs     = self.samples.r[nBurn :: nThin, 0]
        
        out = {
            'alphas' : alphas,
            'betas'  : betas,
            'mus'    : mus,
            'sigmas' : sigmas,
            'xis'    : xis,
            'taus'   : taus,
            'rs'     : rs,
            'deltas' : deltas,
            'chis'   : chis,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'V'      : self.data.V,
            'logd'   : self.samples.ld,
            'priors' : self.priors,
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

    def set_projection(self, p):
        self.data.Yp = euclidean_to_psphere(self.data.V, p)
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
        sw_alpha += lp[sw.T[0]] - lp[sw.T[1]]
        sw_alpha *= self.itl[sw.T[0]] - self.itl[sw.T[1]]
        logp = np.log(uniform(size = sw_alpha.shape))
        for tt in sw[np.where(logp < sw_alpha)[0]]:
            # report successful swap
            self.swap_succeeds[tt[0],tt[1]] += 1
            self.swap_succeeds[tt[1],tt[0]] += 1
            self.samples.alpha[ci][tt[0]], self.samples.alpha[ci][tt[1]] =      \
                self.samples.alpha[ci][tt[1]].copy(), self.samples.alpha[ci][tt[0]].copy()
            self.samples.beta[ci][tt[0]], self.samples.beta[ci][tt[1]] =        \
                self.samples.beta[ci][tt[1]].copy(), self.samples.beta[ci][tt[0]].copy()
            self.samples.mu[ci][tt[0]], self.samples.mu[ci][tt[1]] =            \
                self.samples.mu[ci][tt[1]].copy(), self.samples.mu[ci][tt[0]].copy()
            self.samples.sigma[ci][tt[0]], self.samples.sigma[ci][tt[1]] =      \
                self.samples.sigma[ci][tt[1]].copy(), self.samples.sigma[ci][tt[0]].copy()
            self.samples.xi[ci][tt[0]], self.samples.xi[ci][tt[1]] =            \
                self.samples.xi[ci][tt[1]].copy(), self.samples.xi[ci][tt[0]].copy()
            self.samples.tau[ci][tt[0]], self.samples.tau[ci][tt[1]] =          \
                self.samples.tau[ci][tt[1]].copy(), self.samples.tau[ci][tt[0]].copy()
            self.samples.r[ci][tt[0]], self.samples.r[ci][tt[1]] =              \
                self.samples.r[ci][tt[1]].copy(), self.samples.r[ci][tt[0]].copy()
            self.samples.delta[ci][tt[0]], self.samples.delta[ci][tt[1]] =      \
                self.samples.delta[ci][tt[1]].copy(), self.samples.delta[ci][tt[0]].copy()
            self.samples.chi[ci][tt[0]], self.samples.chi[ci][tt[1]] =          \
                self.samples.chi[ci][tt[1]].copy(), self.samples.chi[ci][tt[0]].copy()
        return

    def __init__(
            self,
            data,
            prior_mu  = (0., 2.),
            prior_sigma = (2., 2.),
            prior_xi  = (0., 2.),
            prior_tau = (3., 3.),
            prior_chi = (0.1, 0.1),
            p         = 10,
            max_clust = 200,
            ntemps    = 5,
            stepping  = 1.15,
            **kwargs
            ):
        self.data = data
        self.nClust = max_clust
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        _prior_mu  = UniNormalPrior(*prior_mu)
        _prior_sigma = GammaPrior(*prior_sigma)
        _prior_chi = GEMPrior(*prior_chi)
        _prior_xi  = UniNormalPrior(*prior_xi)
        _prior_tau = GammaPrior(*prior_tau)
        self.priors = Prior(_prior_mu, _prior_sigma, _prior_xi, _prior_tau, _prior_chi,)
        self.set_projection(p)
        self.itl = 1 / stepping**np.arange(ntemps)
        self.nTemp = ntemps
        self.nSwap_per = self.nTemp // 2
        self.swap_start = 100
        self.temp_unravel = np.repeat(np.arange(self.nTemp), self.nDat)
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1):
        new_gammas = []
        placeholder = np.zeros((n_per_sample, self.nClust))
        for s in range(self.nSamp):
            deltas = py_sample_cluster_bgsb_fixed(self.samples.chi[s], placeholder)
            shapes = self.samples.alpha[s][deltas]
            rates  = self.samples.beta[s][deltas]
            new_gammas.append(gamma(shape = shapes, scale = 1 / rates))
        return np.vstack(new_gammas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 1):
        gammas = self.generate_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_hypercube(gammas)

    def load_data(self, path):
        if type(path) is BytesIO:
            out = pickle.loads(path.getvalue())
        else:
            with open(path, 'rb') as file:
                out = pickle.load(file)
        
        deltas = out['deltas']
        chis   = out['chis']
        alphas = out['alphas']
        betas  = out['betas']
        mus    = out['mus']
        sigmas = out['sigmas']
        xis    = out['xis']
        taus   = out['taus']
        rs     = out['rs']
        
        self.nSamp  = deltas.shape[0]
        self.nDat   = deltas.shape[1]
        self.nCol   = alphas.shape[1]
        self.nClust = chis.shape[1] + 1
        
        self.data = Data_From_Sphere(out['V'])
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass

        self.samples       = Samples1T(self.nSamp, self.nDat, self.nCol, self.nClust)
        self.samples.chi   = chis
        self.samples.delta = deltas
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.mu    = mus
        self.samples.sigma = sigmas
        self.samples.xi    = xis
        self.samples.tau   = taus
        self.samples.r     = rs
        self.samples.ld    = out['logd']

        self.time_elapsed = out['time']
        return

    def __init__(self, path):
        self.load_data(path)
        return

if __name__ == '__main__':
    from data import Data_From_Raw, Data_From_Sphere
    from projgamma import GammaPrior
    from pandas import read_csv
    import os

    raw = read_csv('./datasets/ivt_nov_mar.csv')
    dat = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    # raw = read_csv('./simulated/sphere2/data_m1_r16_i5.csv').values
    # dat = Data_From_Sphere(raw)
    model = Chain(dat, ntemps = 1, max_clust = 200)
    model.sample(15000, verbose = True)
    model.write_to_disk('./test/results.pkl', 5000, 10)
    res = Result('./test/results.pkl')
    postpred = res.generate_posterior_predictive_hypercube(n_per_sample = 10)

    print(postpred.mean(axis = 0))

    from matplotlib import pyplot as plt
    plt.plot(res.samples.ld)
    plt.show()
    raise

# EOF
