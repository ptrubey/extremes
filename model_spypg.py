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
    bincount2D_vectorized, pt_py_sample_cluster_bgsb, pt_py_sample_chi_bgsb
from data import euclidean_to_angular, euclidean_to_hypercube, Data_From_Sphere
from projgamma import GammaPrior, logd_prodgamma_my_mt, logd_prodgamma_my_st,   \
    logd_prodgamma_paired, logd_gamma, logd_gamma_my, pt_logd_gamma_my,         \
    pt_logd_projgamma_my_mt_inplace_unstable, pt_logd_projgamma_paired_yt,      \
    pt_logpost_loggammagamma, pt_logd_prodgamma_my_st

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
        self.chi   = np.empty((nSamp + 1, nTemp, nClust))
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
        self.chi   = np.empty((nSamp + 1, nClust))
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
    
    max_clust_count = None  # Stick-breaking truncation point

    swap_attempts = None    # Parallel tempering swap attempts (count)
    swap_succeeds = None    # Parallel tempering swap success  (count)
    itl = None              # Parallel tempering inverse temp ladder
    tidx = None             # Parallel Tempering Temp Index : range(self.nTemp)
    
    ph_sigma = None         # placeholder, ones(nTemp x nClust x nCol)
    
    # Scratch arrays
    _scratch_dmat  = None # bool (nDat, nTemp, nClust)
    _scratch_delta = None # real (nDat, nTemp, nClust)
    _scratch_alpha = None # real (nTemp, nClust, nCol)

    _curr_cluster_state = None
    _cand_cluster_state = None

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
                self._scratch_delta, self.data.Yp, alpha, self.ph_sigma,
                )
        np.nan_to_num(self._log_delta_likelihood, False, -np.inf)
        self._scratch_delta *= self.itl[None,:,None]
        return

    def sample_delta(self, chi, alpha):
        self._scratch_delta[:] = 0.
        self.log_delta_likelihood(alpha)
        return pt_py_sample_cluster_bgsb(chi, self._scratch_delta)
    
    def sample_chi(self, delta):
        chi = pt_py_sample_chi_bgsb(
            delta, *self.priors.chi, trunc = self.max_clust_count,
            )
        return chi

    def sample_alpha_new(self, xi, tau):
        out = gamma(
            shape = xi, scale = 1 / tau, 
            size = (self.max_clust_count, self.nTemp, self.nCol),
            ).swapaxes(0,1)
        return(out)

    def log_alpha_likelihood(self, alpha, r, delta):
        Y = r[:,:, None] * self.data.Yp[None, :, :] # (t,n,1)*(1,n,d) = t,n,d
        dmat = sparse.COO(coords = delta.T, data = 1, fill_value = 0) # (t,n,j)
        slY = sparse.einsum('tnd,tnj->tjd', np.log(Y), dmat)        # (t,j,d)
        out = np.zeros(alpha.shape)
        out -= self._curr_cluster_state[:,:,None] * gammaln(alpha)
        out += (alpha - 1) * slY
        return out

    def sample_alpha(self, alpha, delta, r, xi, tau):
        self._curr_cluster_state[:] = bincount2D_vectorized(
            delta, self.max_clust_count,
            )
        self._cand_cluster_state[:] = (self._curr_cluster_state == 0)
        delta_ind_mat = delta[:,:,None] == range(self.max_clust_count)
        idx = np.where(~self._cand_cluster_state)
        ndx = np.where(self._cand_cluster_state)


        self._scratch_alpha[:] = -np.inf
        self._scratch_alpha[idx] = 0.
        
        acurr = alpha.copy()
        lacurr = np.log(acurr)
        lacand = lacurr.copy()
        lacand[idx] += normal(scale = 0.1, size = (idx[0].shape[0], self.nCol))
        acand = np.exp(lacand)
        
        self._scratch_alpha += self.log_alpha_likelihood(acand, r, delta, delta_ind_mat)
        self._scratch_alpha -= self.log_alpha_likelihood(acurr, r, delta, delta_ind_mat)
        with np.errstate(invalid = 'ignore'):
            self._scratch_alpha *= self.itl[:,None]
        self._scratch_alpha += self.log_logalpha_prior(lacand, xi, tau)
        self._scratch_alpha -= self.log_logalpha_prior(lacurr, xi, tau)

        keep = np.where(np.log(uniform(size = alpha.shape)) < self._scratch_alpha)
        acurr[keep] = acand[keep]
        
        acurr[ndx] = self.sample_alpha_new(xi, tau)[ndx]
        
        return(acurr)

    def log_logxi_posterior(self, logxi, sum_alpha, sum_log_alpha, n):
        out = np.zeros(logxi.shape)
        xi = np.exp(logxi)
        out += self.itl[:,None] * (xi - 1) * sum_log_alpha
        out -= self.itl[:,None] * n[:,None] * gammaln(xi)
        out += self.priors.xi.a * logxi
        out -= self.priors.xi.b * xi
        out += gammaln((n * self.itl)[:,None] * xi + self.priors.tau.a)
        out -= ((n * self.itl)[:,None] * xi + self.priors.tau.a) * \
                    np.log(self.itl[:,None] * sum_alpha + self.priors.tau.b)
        return out
    
    def sample_xi(self, xi, alpha, extant_clusters):
        # n = (~self._cand_cluster_state).sum(axis = 1)
        n = extant_clusters.sum(axis = 1)
        xcurr = xi.copy()
        lxcurr = np.log(xcurr)
        lxcand = lxcurr.copy()
        lxcand += normal(scale = 0.1, size = lxcand.shape)

        sa = np.einsum('tjd,tj->td', alpha, extant_clusters)
        sla = np.einsum('tjd,tj->td', np.log(alpha), extant_clusters)

        logp = np.zeros(xcurr.shape)
        logp += self.log_logxi_posterior(lxcand, sa, sla, n)
        logp -= self.log_logxi_posterior(lxcurr, sa, sla, n)
        # Tempering handled internally
        
        keep = np.where(np.log(uniform(size = logp.shape)) < logp)
        xcurr[keep] = np.exp(lxcand[keep])
        return xcurr

    def sample_tau(self, xi, alpha, extant_clusters):
        n = extant_clusters.sum(axis = 1)
        sa = np.einsum('tjd,tj->td', alpha, extant_clusters)
        shape = (n * self.itl)[:,None] * alpha + self.priors.tau.a
        rate = sa * self.itl[:,None] + self.priors.tau.b
        return gamma(shape = shape, scale = 1 / rate)

    def sample_r(self, delta, alpha):
        sa = alpha.sum(axis = 2)
        shape = sa[self.temp_unravel, delta.ravel()].reshape(self.nTemp, self.nDat)
        rate  = self.data.Yp.sum(axis = 1)[None,:]
        return gamma(shape = shape, scale = 1 / rate)
    
    def initialize_sampler(self, ns):
        """ Initialize the sampler """
        # Initialize storage
        self._scratch_dmat  = np.zeros((self.nDat, self.nTemp, self.max_clust_count), dtype = bool)
        self._scratch_delta = np.zeros((self.nDat, self.nTemp, self.max_clust_count))
        self._scratch_alpha = np.zeros((self.nTemp, self.max_clust_count, self.nCol))
        self._curr_cluster_state = np.zeros((self.nTemp, self.max_clust_count), dtype = int)
        self._cand_cluster_state = np.zeros((self.nTemp, self.max_clust_count), dtype = bool)
        # Initialize Samples
        self.samples = Samples(ns, self.nDat, self.nCol, self.nTemp, self.max_clust_count)
        self.samples.alpha[0] = gamma(shape = 1.0001, scale = 1, size = (self.nTemp, self.max_clust_count, self.nCol))        self.samples.xi[0] = gamma(shape = 2., scale = 2., size = (self.nTemp, self.nCol))
        self.samples.tau[0] = gamma(shape = 2., scale = 2., size = (self.nTemp, self.nCol))
        self.samples.delta[0] = choice(self.max_clust_count - 20, size = (self.nTemp, self.nDat))
        self.samples.r[0] = self.sample_r(
                self.samples.delta[0], self.samples.zeta[0], self.samples.sigma[0],
                )
        # Initialize the iterator
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

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta = self.curr_delta.copy()
        alpha = self.curr_alpha
        xi    = self.curr_xi
        tau   = self.curr_tau
        r     = self.curr_r

        self.curr_iter += 1
        # Sample new cluster membership indicators 
        delta = self.sample_delta(delta, zeta, sigma, eta)
        # clean indices and re-index








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
