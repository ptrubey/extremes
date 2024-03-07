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
from projgamma import GammaPrior, logd_gamma_my, pt_logd_gamma_my,              \
    pt_logd_projgamma_my_mt_inplace_unstable, pt_logd_projgamma_paired_yt,      \
    pt_logd_prodgamma_my_st, pt_logd_mixprojresgamma

Prior = namedtuple('Prior', 'xi tau chi')

class Samples(object):
    alpha = None  # Cluster shape
    xi    = None  # Centering Shape
    tau   = None  # Centering Rate
    chi   = None  # Stickbreaking unnormalized cluster weights
    r     = None  # Radius
    delta = None  # Cluster Assignment
    ld    = None  # Log-density of cold chain

    def __init__(self, nSamp, nDat, nCol, nTemp, nClust):
        self.alpha = np.empty((nSamp + 1, nTemp, nClust, nCol))
        self.xi    = np.empty((nSamp + 1, nTemp, nCol))
        self.tau   = np.empty((nSamp + 1, nTemp, nCol))
        self.r     = np.empty((nSamp + 1, nTemp, nDat))
        self.chi   = np.empty((nSamp + 1, nTemp, nClust - 1))
        self.delta = np.empty((nSamp + 1, nTemp, nDat), dtype = int)
        self.ld    = np.empty((nSamp + 1))
        return

class Samples1T(object):
    alpha = None  # Cluster shape
    xi    = None  # Centering Shape
    tau   = None  # Centering Rate
    nu    = None  # Stickbreaking unnormalized cluster weights
    r     = None  # Radius
    delta = None  # Cluster Assignment
    ld    = None  # Log-density of cold chain

    def __init__(self, nSamp, nDat, nCol, nClust):
        self.alpha = np.empty((nSamp + 1, nClust, nCol))
        self.xi    = np.empty((nSamp + 1, nCol))
        self.tau   = np.empty((nSamp + 1, nCol))
        self.r     = np.empty((nSamp + 1, nDat))
        self.chi   = np.empty((nSamp + 1, nClust - 1))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.ld    = np.empty((nSamp + 1))
        return

class Chain(ParallelTemperingStickBreakingSampler):
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter]
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
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter]
    
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

    def log_delta_likelihood(self, alpha):
        """
        inputs:
            alpha : (t, j, d)
            sigma : (t, j, d)
        outputs:
            out   : (n, t, j)
        """
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            pt_logd_projgamma_my_mt_inplace_unstable(
                self._scratch_delta, 
                self.data.Yp, 
                alpha, self._placeholder_sigma_1,
                )
        np.nan_to_num(self._scratch_delta, False, -np.inf)
        self._scratch_delta *= self.itl[None,:,None]
        return

    def sample_delta(self, chi, alpha):
        self._scratch_delta[:] = 0.
        self.log_delta_likelihood(alpha)
        return pt_py_sample_cluster_bgsb_fixed(chi, self._scratch_delta)
    
    def sample_chi(self, delta):
        chi = pt_py_sample_chi_bgsb_fixed(
            delta, *self.priors.chi, trunc = self.nClust,
            )
        return chi

    def sample_alpha_new(self, xi, tau):
        out = gamma(
            shape = xi, scale = 1 / tau, 
            size = (self.nClust, self.nTemp, self.nCol),
            ).swapaxes(0,1)
        return out

    def log_alpha_likelihood(self, alpha, r, delta):
        Y = r[:,:, None] * self.data.Yp[None, :, :]  # (t,n,1)*(1,n,d) = t,n,d
        self._scratch_dmat[:] = delta[:,:,None] == range(self.nClust)
        try:
            slY = sparse.einsum(
                'tnj,tnd->tjd', 
                sparse.COO.from_numpy(self._scratch_dmat),
                np.log(Y), 
                ).todense() # (t,j,d)
        except ValueError:
            slY = np.einsum('tnd,tnj->tjd', np.log(Y), self._scratch_dmat)
        out = np.zeros(alpha.shape)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            out += (alpha - 1) * slY
            out -= self._curr_cluster_state[:,:,None] * gammaln(alpha)
        return out

    def log_logalpha_prior(self, logalpha, xi, tau):
        logd = np.zeros(logalpha.shape)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            # logd += xi[:,None] * np.log(tau[:,None])
            # logd -= gammaln(tau[:,None])
            logd += xi[:,None] * logalpha
            logd -= tau[:,None] * np.exp(logalpha)
        np.nan_to_num(logd, False, -np.inf)
        return logd

    def sample_alpha(self, alpha, delta, r, xi, tau):
        idx = np.where(self._extant_clust_state)
        ndx = np.where(self._cand_cluster_state)

        self._scratch_alpha[:] = -np.inf
        self._scratch_alpha[idx] = 0.
        
        assert(~(alpha[self._extant_clust_state] == 0).any())

        with np.errstate(divide = 'ignore'):
            acurr = alpha.copy()
            lacurr = np.log(acurr)
            lacand = lacurr.copy()
            # lacand[idx] += normal(scale = 0.1, size = (idx[0].shape[0], self.nCol))
            lacand += normal(scale = 0.1, size = lacand.shape)
            acand = np.exp(lacand)
        
        assert (~np.any(acand == 0))

        self._scratch_alpha += self.log_alpha_likelihood(acand, r, delta)
        self._scratch_alpha -= self.log_alpha_likelihood(acurr, r, delta)
        with np.errstate(invalid = 'ignore'):
            self._scratch_alpha *= self.itl[:,None,None]
        self._scratch_alpha += self.log_logalpha_prior(lacand, xi, tau)
        self._scratch_alpha -= self.log_logalpha_prior(lacurr, xi, tau)

        keep = np.where(np.log(uniform(size = acurr.shape)) < self._scratch_alpha)
        acurr[keep] = acand[keep]
        acurr[ndx] = self.sample_alpha_new(xi, tau)[ndx]
        
        assert (~(acurr[keep] == 0.).any())
        return(acurr)

    def log_logxi_posterior(self, logxi, sum_alpha, sum_log_alpha):
        out = np.zeros(logxi.shape)
        xi = np.exp(logxi)
        out += (xi - 1) * sum_log_alpha
        out -= self.nClust * gammaln(xi)
        out += self.priors.xi.a * logxi
        out -= self.priors.xi.b * xi
        out += gammaln(self.nClust * xi + self.priors.tau.a)
        out -= (self.nClust * xi + self.priors.tau.a) * np.log(sum_alpha + self.priors.tau.b)
        # out += self.itl[:,None] * (xi - 1) * sum_log_alpha
        # out -= self.itl[:,None] * n[:,None] * gammaln(xi)
        # out += self.priors.xi.a * logxi
        # out -= self.priors.xi.b * xi
        # out += gammaln((n * self.itl)[:,None] * xi + self.priors.tau.a)
        # out -= ((n * self.itl)[:,None] * xi + self.priors.tau.a) * \
        #             np.log(self.itl[:,None] * sum_alpha + self.priors.tau.b)
        return out
    
    def sample_xi(self, xi, alpha):
        # n = self._extant_clust_state.sum(axis = 1)
        xcurr = xi.copy()
        lxcurr = np.log(xcurr)
        lxcand = lxcurr.copy()
        lxcand += normal(scale = 0.2, size = lxcand.shape)

        # with np.errstate(divide = 'ignore'):
        #     sa = np.einsum('tjd,tj->td', alpha, self._extant_clust_state)
        #     sla = np.einsum('tjd,tj->td', np.log(alpha), self._extant_clust_state)
        sa = alpha.sum(axis = 1)
        sla = np.log(alpha).sum(axis = 1)

        logp = np.zeros(xcurr.shape)
        logp += self.log_logxi_posterior(lxcand, sa, sla)
        logp -= self.log_logxi_posterior(lxcurr, sa, sla)
        # Tempering handled internally
        
        keep = np.where(np.log(uniform(size = logp.shape)) < logp)
        xcurr[keep] = np.exp(lxcand[keep])
        return xcurr

    def sample_tau(self, xi, alpha):
        n = self._extant_clust_state.sum(axis = 1)
        sa = np.einsum('tjd,tj->td', alpha, self._extant_clust_state)
        shape = (n * self.itl)[:,None] * xi + self.priors.tau.a
        rate = sa * self.itl[:,None] + self.priors.tau.b
        return gamma(shape = shape, scale = 1 / rate)

    def sample_r(self, delta, alpha):
        sa = alpha.sum(axis = -1)
        shape = sa[self.temp_unravel, delta.ravel()].reshape(self.nTemp, self.nDat)
        rate  = self.data.Yp.sum(axis = 1)[None,:]
        return gamma(shape = shape, scale = 1 / rate)
    
    def initialize_sampler(self, ns):
        """ Initialize the sampler """
        # Initialize Placeholders (for restricted gamma):
        self._placeholder_sigma_1 = np.ones((self.nTemp, self.nClust, self.nCol))
        self._placeholder_sigma_2 = np.ones((self.nTemp, self.nDat, self.nCol))
        # Initialize storage
        self._scratch_dmat  = np.zeros((self.nTemp, self.nDat, self.nClust), dtype = bool)
        self._scratch_delta = np.zeros((self.nDat, self.nTemp, self.nClust))
        self._scratch_alpha = np.zeros((self.nTemp, self.nClust, self.nCol))
        # Initialize Samples
        self.samples = Samples(ns, self.nDat, self.nCol, self.nTemp, self.nClust)
        self.samples.alpha[0] = gamma(
            shape = 1.0001, scale = 1, size = (self.nTemp, self.nClust, self.nCol),
            )        
        self.samples.xi[0] = gamma(
            shape = 2., scale = 2., size = (self.nTemp, self.nCol),
            )
        self.samples.tau[0] = gamma(
            shape = 2., scale = 2., size = (self.nTemp, self.nCol),
            )
        self.samples.delta[0] = choice(
            self.nClust, size = (self.nTemp, self.nDat),
            )
        self.samples.r[0] = self.sample_r(
            self.samples.delta[0], self.samples.alpha[0],
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
        # ll += pt_logd_projgamma_paired_yt(
        #     self.data.Yp, 
        #     self.curr_alpha[
        #         self.temp_unravel, self.curr_delta.ravel()
        #         ].reshape(self.nTemp, self.nDat, self.nCol),
        #     self._placeholder_sigma_2,
        #     ).sum(axis = 1)
        # ll += np.einsum(
        #     'tj,tj->t',
        #     pt_logd_gamma_my(self.curr_alpha, self.curr_xi, self.curr_tau),
        #     extant_clusters,
        #     )
        ll += pt_logd_mixprojresgamma(self.data.Yp, self.curr_alpha, self.curr_chi).sum(axis = 0)
        ll += pt_logd_gem_mx_st_fixed(self.curr_chi, *self.priors.chi)
        ll += pt_logd_gamma_my(self.curr_alpha, self.curr_xi, self.curr_tau).sum(axis = 1)
        return ll

    def log_prior(self):
        lp = np.zeros(self.nTemp)
        lp += logd_gamma_my(self.curr_xi, *self.priors.xi).sum(axis = 1)
        lp += logd_gamma_my(self.curr_tau, *self.priors.tau).sum(axis = 1)
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
        chi   = self.curr_chi
        xi    = self.curr_xi
        tau   = self.curr_tau
        r     = self.curr_r

        # Advance the iterator
        self.curr_iter += 1
        ci = self.curr_iter

        # Update cluster assignments
        self.samples.delta[ci] = self.sample_delta(chi, alpha)
        self.samples.chi[ci] = self.sample_chi(self.curr_delta)
        self.update_cluster_state()
        self.samples.alpha[ci] = self.sample_alpha(
            alpha, self.curr_delta, r, xi, tau
            )
        self.samples.r[ci] = self.sample_r(self.curr_delta, self.curr_alpha)
        self.samples.xi[ci] = self.sample_xi(xi, self.curr_alpha)
        self.samples.tau[ci] = self.sample_tau(self.curr_xi, self.curr_alpha)

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
        xis    = self.samples.xi[nBurn :: nThin, 0]
        taus   = self.samples.tau[nBurn :: nThin, 0]
        deltas = self.samples.delta[nBurn :: nThin, 0]
        chis   = self.samples.chi[nBurn :: nThin, 0]
        rs     = self.samples.r[nBurn :: nThin, 0]
        
        out = {
            'alphas' : alphas,
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
            prior_xi  = (3., 3.),
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
        _prior_chi = GEMPrior(*prior_chi)
        _prior_xi  = GammaPrior(*prior_xi)
        _prior_tau = GammaPrior(*prior_tau)
        self.priors = Prior(_prior_xi, _prior_tau, _prior_chi,)
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
            new_gammas.append(gamma(shape = shapes))
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
    model = Chain(dat, ntemps = 5, max_clust = 200)
    model.sample(15000, verbose = True)
    model.write_to_disk('./test/results.pkl', 5000, 10)
    res = Result('./test/results.pkl')
    postpred = res.generate_posterior_predictive_hypercube(n_per_sample = 10)

    print(postpred.mean(axis = 0))

    from matplotlib import pyplot as plt
    plt.plot(res.samples.ld)
    plt.show()

# EOF
