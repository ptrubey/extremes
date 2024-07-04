"""
Implementation of SPYPPRG-R

Regression extension of SPYPPRG.

"""
from collections import namedtuple
from itertools import repeat
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log
from scipy.special import gammaln
from scipy.stats import invwishart, multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import cholesky, multi_dot
from numpy.random import choice, gamma, uniform, normal
from io import BytesIO

from cUtility import pityor_cluster_sampler, generate_indices,                  \
    softplus_1d_inplace, softplus_2d_inplace, softplus_3d_inplace
from samplers import DirichletProcessSampler
from data import euclidean_to_angular, euclidean_to_hypercube, RealData
from projgamma import pt_logd_projgamma_my_mt_inplace_unstable, logd_gamma,     \
    logd_prodgamma_my_mt, logd_prodgamma_paired, logd_prodgamma_my_st
from cov import PerObsOnlineCovariance, OnlineCovariance
from data import Data_From_Raw

def softplus(X : np.ndarray, inplace : bool):
    """ 
    softplus function -- less agressive than log-transformation for 
        unbounding X to the real number line.
    """
    if not inplace:
        return np.log(1. + np.exp(X))
    else:
        np.exp(X, out = X)
        X += 1
        np.log(X, out = X)
    return

def softplus_inplace(X : np.ndarray):
    if len(X.shape) == 1:
        return softplus_1d_inplace(X)
    elif len(X.shape) == 2:
        return softplus_2d_inplace(X)
    elif len(X.shape) == 3:
        return softplus_3d_inplace(X)

def softplus_inplace_old(X : np.ndarray, threshold : float = 20.) -> None:
    X[X < threshold] = np.log(1. + np.exp(X[X < threshold]))
    return

NormalPrior = namedtuple('NormalPrior','mu sigma')
NIWPrior    = namedtuple('NIWPrior','mu kappa nu psi')
Prior       = namedtuple('Prior', 'mu_Sigma epsilon')
Dimensions  = namedtuple('Dimensions','beta gamma zeta')
Bounds      = namedtuple('Bounds', 'beta gamma zeta')

class Regressors(object):
    obs = None
    loc = None
    int = None

    def __init__(
            self,
            observation : np.ndarray,  # (N,d1)
            location    : np.ndarray,  # (S,d2)
            interaction : np.ndarray,  # (N,S,D3)
            ):
        # Bounds checking (omitting)
        # assert observation.shape[0] == interaction.shape[0]
        # assert location.shape[0] == interaction.shape[1]
        # instantiation
        self.obs = np.asarray(observation)
        self.loc = np.asarray(location)
        self.int = np.asarray(interaction)
        return

class RegressionData(RealData):
    def load_regressors(
            self, 
            observation : np.ndarray,
            location    : np.ndarray, 
            interaction : np.ndarray,
            ):
        obsi = observation[self.I]
        inti = interaction[self.I]

        assert obsi.shape[0] == self.nDat
        assert inti.shape[0] == self.nDat
        assert location.shape[0] == self.nCol

        self.X = Regressors(obsi, location, inti)
        return
    
    def __init__(self, observation, location, interaction, **kwargs):
        super().__init__(**kwargs)
        self.load_regressors(observation, location, interaction)
        return

class Samples(object):
    theta = None
    epsilon = None
    Sigma = None
    mu    = None
    delta = None
    r     = None
    ld    = None
    
    def __init__(self, nSamp, nDat, nDim, nCol):
        self.theta = [None] * (nSamp + 1)
        self.epsilon = np.empty((nSamp + 1, nDim))
        self.Sigma = np.empty((nSamp + 1, nCol, nCol))
        self.mu    = np.empty((nSamp + 1, nCol))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.ld    = np.empty((nSamp + 1))
        return

class ChainBase(object):
    bounds = None
    data   = None

    def linkfn(self, arr : np.ndarray, inplace : bool):
        if not inplace:
            return softplus(arr, False)
        else:
            return softplus_inplace(arr)

    def compute_shape_theta(
            self,
            delta   : np.ndarray,
            theta   : np.ndarray,
            epsilon : np.ndarray,
            ):
        """
        in : 
            delta   : (N)
            theta   : (J,D)
            epsilon : (S)
        out: 
            shapes  : (N,S)
        """
        # Parsing the theta vector
        beta  = theta[:, self.bounds.beta[0]  : self.bounds.beta[1] ]
        gamma = theta[:, self.bounds.gamma[0] : self.bounds.gamma[1]]
        zeta  = theta[:, self.bounds.zeta[0]  : self.bounds.zeta[1] ]
        # Computing shape parameters
        Ase = np.zeros((self.N, self.S))
        Ase += np.einsum('nd,nd->n', self.data.X.obs, beta[delta])[:,None] # [n,d1]->[n]
        Ase += np.einsum('sd,nd->ns', self.data.X.loc, gamma[delta])       # [s,d2]->[s]
        Ase += np.einsum('nsd,nd->ns', self.data.X.int, zeta[delta])   # [n,s,d3]->[n,s]
        Ase += epsilon[None]
        self.linkfn(Ase, inplace = True)
        return Ase

class Chain(DirichletProcessSampler, ChainBase):
    concentration = None
    discount      = None
    llik_delta    = None 

    @property
    def curr_theta(self):
        return self.samples.theta[self.curr_iter]
    @property
    def curr_Sigma(self):
        return self.samples.Sigma[self.curr_iter]
    @property
    def curr_mu(self):
        return self.samples.mu[self.curr_iter]
    @property
    def curr_epsilon(self):
        return self.samples.epsilon[self.curr_iter]
    @property
    def curr_delta(self):
        return self.samples.delta[self.curr_iter]
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter]
    
    def clean_delta_theta(
            self, 
            delta : np.ndarray,
            theta : np.ndarray,
            ):
        """
        delta : cluster indicator vector (n)
        beta  : cluster parameter matrix (J* x d1)
        gamma : cluster parameter matrix (J* x d2)
        phi   : cluster parameter matrix (J* x d3)
        """
        # reindex those clusters
        keep, delta[:] = np.unique(delta, return_inverse = True)
        # return new indices, cluster parameters associated with populated clusters
        return delta, theta[keep]
    
    def compute_shape_delta(
            self,
            theta : np.ndarray,                                       # [J,D]
            epsilon : np.ndarray,                                     # [S]
            ):
        """
        in : 
            theta   : (J,D)
            epsilon : (S)
        out: 
            shapes  : (N,J,S)
        """
        beta  = theta[:, self.bounds.beta[0]  : self.bounds.beta[1] ] # [J,d1]
        gamma = theta[:, self.bounds.gamma[0] : self.bounds.gamma[1]] # [J,d2]
        zeta  = theta[:, self.bounds.zeta[0]  : self.bounds.zeta[1] ] # [J,d3]
        # [n,j,s]
        # Computing Shape Parameters
        self.shape_delta[:] = 0.
        # self.shape_delta_t1[:] = 0.
        # self.shape_delta_t2[:] = 0.
        # self.shape_delta_t3[:] = 0.
        self.shape_delta[:] = np.matmul(self.data.X.int, zeta.T).transpose(0,2,1)
        self.shape_delta += np.matmul(self.data.X.obs, beta.T)[:,:,None]
        self.shape_delta += np.matmul(gamma, self.data.X.loc.T)[None]
        # np.matmul(gamma, self.data.X.loc.T, out = self.shape_delta_t2)

        # np.einsum('nd,jd->nj', self.data.X.obs, beta, out = self.shape_delta_t1)
        # np.einsum('sd,jd->js', self.data.X.loc, gamma, out = self.shape_delta_t2)
        # np.einsum('nsd,jd->njs', self.data.X.int, zeta, out = self.shape_delta_t3)
        # self.shape_delta += np.einsum('nd,jd->nj', self.data.X.obs, beta)[:,:,None]
        # self.shape_delta += np.einsum('sd,jd->js', self.data.X.loc, gamma)[None]
        # self.shape_delta += np.einsum('nsd,jd->njs', self.data.X.int, zeta)
        # self.shape_delta += self.shape_delta_t1
        # self.shape_delta += self.shape_delta_t2
        # self.shape_delta += self.shape_delta_t3
        self.shape_delta += epsilon[None, None]
        self.linkfn(self.shape_delta, inplace = True)
        return
    
    def sample_r(
            self, 
            delta   : np.ndarray, # cluster identifiers 
            theta   : np.ndarray, # Regression Coef's
            epsilon : np.ndarray, # location specific effect [s]
            ):
        """
        n : iterator over storms; 1,...,N
        s : iterator over sites;  1,...,S
        d : iterator over dimensions; 1,...,d[1-3]
        ---
        storm-matrix is [n,s]
        """
        Ase = self.compute_shape_theta(delta, theta, epsilon)
        As  = Ase.sum(axis = -1)            # sum over last dimension
        Bs  = self.data.Yp.sum(axis = -1)   # Sum over last dimension
        r   = gamma(shape = As, scale = 1 / Bs)
        r[r < 1e-10] = (As / Bs)[r < 1e-10] # if it gives numerical 0, return MAP
        return r
    
    def sample_delta(
            self, 
            delta   : np.ndarray,
            # r       : np.ndarray,
            theta   : np.ndarray, 
            epsilon : np.ndarray,
            ):
        self.log_likelihood_delta(theta, epsilon)
        # loglik = self.log_likelihood_delta(r, theta, epsilon)
        unifs = uniform(size = self.N)
        delta = pityor_cluster_sampler(
            delta, self.lglik_delta, unifs, self.concentration, self.discount,
            )
        return delta

    def sample_theta(
            self,
            delta   : np.ndarray, # cluster ID
            r       : np.ndarray, # radius
            theta   : np.ndarray, # regression coef's
            epsilon : np.ndarray, # location specific effect
            mu      : np.ndarray, # mean of centering distribution
            Sigma   : np.ndarray, # cov of centering distribution
            ):
        tcurr    =  theta.copy()
        propchol =  np.linalg.cholesky(
            self.cov_theta.cluster_covariance(delta)[:tcurr.shape[0]] * 5.6644 / self.D,
            )        
        tcand    =  np.einsum('jde,je->jd', propchol, normal(size = tcurr.shape))
        tcand += tcurr
        
        lp_curr  =  self.log_posterior_theta(delta, r, tcurr, epsilon, mu, Sigma)
        lp_cand  =  self.log_posterior_theta(delta, r, tcand, epsilon, mu, Sigma)
        logalpha = lp_cand - lp_curr

        keep = np.log(uniform(size = lp_curr.shape)) < logalpha
        tcurr[keep] = tcand[keep]

        self.theta_mh_try[:keep.shape[0]] += 1
        self.theta_mh_keep[:keep.shape[0]][keep] += 1
        return tcurr

    def sample_theta_new(
            self, 
            mu    : np.ndarray, 
            Sigma : np.ndarray, 
            m     : int,
            ):
        out = np.zeros((m, self.D))
        out += (cholesky(Sigma) @ normal(size = (self.D, m))).T
        out += mu[None]
        return out

    def sample_mu_Sigma(
            self,
            theta : np.ndarray,
            ):
        # Parsing prior parameters
        mu_0, ka_0, nu_0, ps_0 = self.priors.mu_Sigma
        tbar = theta.mean(axis = 0)
        S1 = (theta - tbar[None]).T @ (theta - tbar[None])
        S2 = (tbar - mu_0)[:,None] @ (tbar - mu_0)[None]
        n = theta.shape[0]
        # Posterior Parameters
        mu_n = (ka_0 / (ka_0 + n)) * mu_0 + (n / (ka_0 + n)) * tbar
        ka_n = ka_0 + n
        nu_n = nu_0 + n
        ps_n = ps_0 + S1 + (ka_0 * n)/(ka_0 + n) * S2
        # la_ni = cho_solve(cho_factor(la_n), np.eye(self.D)) 
        # Sigma = invwishart.rvs(df = nu_n, scale = la_ni @ la_ni.T)
        Sigma = invwishart.rvs(df = nu_n, scale = ps_n)
        C = cholesky(Sigma / ka_n)
        mu = mu_n + C @ normal(size = self.D)
        return mu, Sigma

    def sample_epsilon(
            self, 
            delta   : np.ndarray,
            r       : np.ndarray, 
            theta   : np.ndarray, 
            epsilon : np.ndarray,
            ):
        if not self.fixed_effects:
            return np.zeros(epsilon.shape)
        eps_curr = epsilon.copy()
        propchol = np.linalg.cholesky(self.cov_epsil.Sigma) 
        eps_cand = eps_curr + propchol @ normal(size = self.S)
        # eps_cand = normal(loc = eps_curr, scale = 1e-3)
        lpo_curr = self.log_posterior_epsilon(delta, r, theta, eps_curr)
        lpo_cand = self.log_posterior_epsilon(delta, r, theta, eps_cand)
        logalpha = lpo_cand - lpo_curr
        keep     = np.log(uniform(logalpha.shape)) < logalpha
        eps_curr[keep] = eps_cand[keep]
        self.epsil_mh_try += 1
        self.epsil_mh_keep[keep] += 1
        return eps_curr

    def initialize_sampler(
            self, 
            ns : int,
            ):
        # Instantiate Sampler
        self.samples = Samples(ns, self.N, self.S, self.D)
        # Set initial values
        delta = choice(int(0.1 * self.J), size = self.N)
        theta = normal(loc = 0, scale = 0.5, size = (self.J, self.D))
        delta, theta = self.clean_delta_theta(delta, theta)
        self.samples.delta[0] = delta
        self.samples.theta[0] = theta
        if not self.fixed_effects:
            self.samples.epsilon[:] = 0.
        else:
            self.samples.epsilon[0] = 0.
        self.samples.r[0] = self.sample_r(
            self.samples.delta[0], 
            self.samples.theta[0], 
            self.samples.epsilon[0],
            )
        self.samples.mu[0]    = normal(size = self.D)
        self.samples.Sigma[0] = np.eye(self.D)
        # Placeholders, Iterator
        self.curr_iter = 0
        self.rate_placeholder_1 = np.ones((self.J, self.D))
        self.rate_placeholder_2 = np.ones((self.N, self.D))
        # AM tune-checking
        self.theta_mh_keep = np.zeros(self.J, int)
        self.theta_mh_try  = np.zeros(self.J, int)
        self.epsil_mh_keep = np.zeros(self.S, int)
        self.epsil_mh_try  = np.zeros(self.S, int)
        return
    
    def log_posterior_epsilon(
            self, 
            delta   : np.ndarray, 
            r       : np.ndarray, 
            theta   : np.ndarray, 
            epsilon : np.ndarray,
            ):
        ase = self.compute_shape_theta(delta, theta, epsilon)
        Y   = r[:,None] * self.data.Yp
        lp  = np.zeros(self.S)
        lp  += np.einsum('ns,ns->s', ase - 1, np.log(Y))
        lp  -= np.einsum('ns->s', gammaln(ase))
        dif = (epsilon - self.priors.epsilon.mu) / self.priors.epsilon.sigma
        lp  -= 0.5 * dif * dif
        return lp

    def log_posterior_theta(
            self, 
            delta   : np.ndarray, 
            r       : np.ndarray, 
            theta   : np.ndarray, 
            epsilon : np.ndarray, 
            mu      : np.ndarray, 
            Sigma   : np.ndarray,
            ):
        # Setup
        ase = self.compute_shape_theta(delta, theta, epsilon) # (N,S)
        Y   = r[:,None] * self.data.Yp
        # Calculation
        lp  = np.zeros(theta.shape[0])
        for i in range(delta.max() + 1):
            # lp[i] += (ase[delta == i] * np.log(self.data.Yp[delta == i])).sum()
            lp[i] += (ase[delta == i] * np.log(Y[delta == i])).sum()
            lp[i] -= gammaln(ase[delta == i]).sum()
            # lp[i] += gammaln(ase[delta == i].sum())
            # lp[i] -= ase[delta == i].sum() * np.log(
            #    self.data.Yp[delta == i].sum(axis = -1)
            #     ).sum()
        # lp -= 0.5 * np.diag(
        #     multi_dot(theta - mu[None], Sigma, (theta - mu[None]).T),
        #     )
        Li  = cho_solve(cho_factor(Sigma), np.eye(self.D))
        # lp -= 0.5 * np.einsum(
        #     'jd,de,je->j', 
        #     theta - mu[None], Sigma, theta - mu[None],
        #     )
        tmp = (theta - mu[None]) @ Li
        lp -= 0.5 * (tmp * tmp).sum(axis = -1)
        lp[np.isnan(lp)] = -np.inf
        return lp

    def log_likelihood_delta(
            self, 
            # r       : np.ndarray, 
            theta   : np.ndarray, 
            epsilon : np.ndarray,
            ):
        # setup
        self.compute_shape_delta(theta, epsilon) # N,J,S
        # Y   = r[:, None] * self.data.Yp
        # calculation
        asum = self.shape_delta @ np.ones(self.S)
        self.lglik_delta[:] = 0.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            self.lglik_delta += np.einsum('njs,ns->nj', self.shape_delta, np.log(self.data.Yp))
            self.lglik_delta -= gammaln(self.shape_delta).sum(axis = -1)
            self.lglik_delta += gammaln(asum)
            self.lglik_delta -= asum * np.log(self.data.Yp.sum(axis = -1))[:,None]
        np.nan_to_num(self.lglik_delta, copy = False, nan = -np.inf)
        return

    def record_log_density(self):
        Y = self.curr_r[:,None] * self.data.Yp 
        ld = 0.
        ld += self.log_posterior_theta(
            self.curr_delta, self.curr_r, self.curr_theta, 
            self.curr_epsilon, self.curr_mu, self.curr_Sigma,
            ).sum()
        ld += multivariate_normal.logpdf(
            self.curr_mu, self.priors.mu_Sigma.mu, 
            self.curr_Sigma / self.priors.mu_Sigma.kappa,
            ).sum()
        ld += invwishart.logpdf(
            self.curr_Sigma, self.priors.mu_Sigma.nu, 
            self.priors.mu_Sigma.psi,
            )
        self.samples.ld[self.curr_iter] = ld
        return

    def iter_sample(self):
        # current cluster assignments; number of new candidate clusters
        delta   = self.curr_delta.copy();  m = self.J - (delta.max() + 1)
        theta   = self.curr_theta
        epsilon = self.curr_epsilon
        mu      = self.curr_mu
        Sigma   = self.curr_Sigma
        r       = self.curr_r

        self.curr_iter += 1
        ci = self.curr_iter
        # Augment theta with new draws
        theta = np.concatenate((theta, self.sample_theta_new(mu, Sigma, m)), 0)
        # Sample Delta, 
        delta = self.sample_delta(delta, theta, epsilon)
        # clear out dropped/unused clusters and re-index
        delta, theta = self.clean_delta_theta(delta, theta)
        self.samples.delta[ci] = delta
        # sample new parameters
        self.samples.r[ci]     = self.sample_r(self.curr_delta, theta, epsilon)
        self.samples.theta[ci] = self.sample_theta(
            self.curr_delta, self.curr_r, theta, epsilon, mu, Sigma,
            )
        self.samples.mu[ci], self.samples.Sigma[ci] = self.sample_mu_Sigma(
            self.curr_theta,
            )
        self.samples.epsilon[ci] = self.sample_epsilon(
            self.curr_delta, self.curr_r, self.curr_theta, epsilon,
            )
        # cleanup
        self.record_log_density()
        if ci > 500:
            self.cov_theta.update(self.curr_theta[self.curr_delta])
            self.cov_epsil.update(self.curr_epsilon)
        return
    
    def write_to_disk(self, path, nBurn, nThin = 1):
        if type(path) is str:
            folder = os.path.split(path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            if os.path.exists(path):
                os.remove(path)
        
        thetas   = np.vstack([
            np.hstack((np.ones((theta.shape[0], 1)) * i, theta))
            for i, theta in enumerate(self.samples.theta[nBurn :: nThin])
            ])
        epsilons = self.samples.epsilon[nBurn :: nThin]
        deltas   = self.samples.delta[nBurn :: nThin]
        rs       = self.samples.r[nBurn :: nThin]
        mus      = self.samples.mu[nBurn :: nThin]
        Sigmas   = self.samples.Sigma[nBurn :: nThin]

        out = {
            'thetas'   : thetas,
            'epsilons' : epsilons,
            'rs'       : rs,
            'deltas'   : deltas,
            'mus'      : mus,
            'Sigmas'   : Sigmas,
            'nCol'     : self.D,
            'nDat'     : self.N,
            'nLoc'     : self.S,
            'Xobs'     : self.data.X.obs,
            'Xloc'     : self.data.X.loc,
            'Xint'     : self.data.X.int,
            'V'        : self.data.V,
            'logd'     : self.samples.ld,
            'time'     : self.time_elapsed_numeric,
            'conc'     : self.concentration,
            'disc'     : self.discount,
            'bounds'   : self.bounds,
            'mhdiag'   : self.mh_accept_rate()
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
        # self.data.Yp = (self.data.V.T / (self.data.V**self.p).sum(axis = 1)**(1/self.p)).T
        self.data.Yp = (
            self.data.V /
            ((self.data.V**self.p).sum(axis = -1)**(1 / self.p))[:,None]
            )
        return

    def set_shapes(self):
        self.N = self.data.nDat
        self.S = self.data.nCol
        self.d = Dimensions(
            self.data.X.obs.shape[-1],
            self.data.X.loc.shape[-1],
            self.data.X.int.shape[-1],
            )
        self.D = sum(self.d)

        self.lglik_delta = np.zeros((self.N, self.J))
        self.shape_delta = np.zeros((self.N, self.J, self.S))
        self.bounds = Bounds(
            (0, self.d.beta),
            (self.d.beta, self.d.beta + self.d.gamma), 
            (self.d.beta + self.d.gamma, self.d.beta + self.d.gamma + self.d.zeta),
            )
        return

    def mh_accept_rate(self):
        theta = (self.theta_mh_keep / (self.theta_mh_try + 1e-9))
        epsil = (self.epsil_mh_keep / (self.epsil_mh_try + 1e-9))
        return (theta, epsil)

    def __init__(
            self,
            data : RegressionData,
            prior_mu_Sigma = (0, 1, 10, 10),
            prior_epsilon = (0, 1), 
            p             = 10,
            concentration = 0.03,
            discount      = 0.03,
            max_clust     = 100,
            fixed_effects = True,
            **kwargs
            ):
        self.data = data
        self.J = max_clust
        self.set_shapes()
        # Parsing the inputs
        self.concentration = concentration
        self.discount = discount
        self.fixed_effects = fixed_effects
        self.p = p
        # Setting the priors
        _prior_mu_Sigma = NIWPrior(
            np.ones(self.D) * prior_mu_Sigma[0],
            prior_mu_Sigma[1],
            prior_mu_Sigma[2],
            np.eye(self.D) * prior_mu_Sigma[3],
            )
        self.priors = Prior(
            _prior_mu_Sigma, 
            NormalPrior(*prior_epsilon),
            )
        # Rest of setup
        self.set_projection()
        self.cov_theta = PerObsOnlineCovariance(self.N, self.D, self.J, 1e-6)
        self.cov_epsil = OnlineCovariance(self.S, 1e-6)
        return

class Result(ChainBase):
    def generate_conditional_posterior_predictive_gammas(self):
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        gammas = []
        for i in range(self.nSamp):
            shape = self.compute_shape_theta(
                self.samples.delta[i], 
                self.samples.theta[i], 
                self.samples.epsilon[i],
                )
            gammas.append(gamma(shape))
        return np.stack(gammas)

    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            dmax = self.samples.delta[s].max()
            njs = np.bincount(self.samples.delta[s], minlength = int(dmax + 1 + m))
            ljs = (
                + njs - (njs > 0) * self.discount 
                + (njs == 0) * (
                    self.concentration 
                    + (njs > 0).sum() * self.discount
                    ) / m
                )
            new_zetas = gamma(
                shape = self.samples.alpha[s],
                scale = 1. / self.samples.beta[s],
                size = (m, self.nCol),
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

        epsilons = out['epsilons']
        thetas = out['thetas']
        rs     = out['rs']
        deltas = out['deltas']
        conc   = out['conc']
        disc   = out['disc']
        mus    = out['mus']
        Sigmas = out['Sigmas']
        conc   = out['conc']
        disc   = out['disc']
        Xobs   = out['Xobs']
        Xloc   = out['Xloc']
        Xint   = out['Xint']
        V      = out['V']
        logd   = out['logd']
        
        self.D = out['nCol']
        self.N = out['nDat']
        self.S = out['nLoc']
        self.nSamp = epsilons.shape[0]
        self.time_elapsed_numeric = out['time']
        self.concentration = conc
        self.discount      = disc
        self.bounds = out['bounds']
        
        self.data = RealData(out['V'], real_type = 'sphere')
        self.data.X = Regressors(Xobs, Xloc, Xint)
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass

        self.samples       = Samples(self.nSamp, self.N, self.D, self.S)
        self.samples.delta = deltas
        self.samples.theta = thetas
        self.samples.mu    = mus
        self.samples.Sigma = Sigmas
        self.samples.epsilon = epsilons
        self.samples.theta = [
            thetas[np.where(thetas.T[0] == i)[0],1:] 
            for i in range(self.nSamp)
            ]
        self.samples.r     = rs
        self.samples.ld    = logd
        return

    def __init__(self, path):
        self.load_data(path)
        return

Summary = namedtuple('Summary','mean sd')

def summarize(X : np.ndarray):
    m  = X.mean(axis = 0)
    s  = X.std(axis = 0)
    return Summary(m, s)

def scale(X : np.ndarray, p : Summary):
    Xm = np.asmatrix(X)
    return ((Xm - p.mean[None]) / p.sd[None])

if __name__ == '__main__':
    X    = pd.read_csv('./simulated/reg/X.csv').values
    loc  = pd.read_csv('./simulated/reg/Xloc.csv').values
    Y    = pd.read_csv('./simulated/reg/Y.csv').values

    Xobs = np.zeros((X.shape[0], 0))
    Xloc = np.zeros((Y.shape[1], 0))
    Xint = np.empty((X.shape[0], Y.shape[1], X.shape[1] * Y.shape[1]))
    Xint[:] = np.kron(X, loc).reshape(Xint.shape) # verified
    
    data = RegressionData(
        raw_real = Y, real_type = 'sphere', 
        observation = Xobs, location = Xloc, interaction = Xint,
        )
    
    model = Chain(data)
    model.sample(30000, True)
    model.write_to_disk('./simulated/reg/result.pkl', 1, 1)
    res   = Result('./simulated/reg/result.pkl')
    postalphas = res.generate_conditional_posterior_predictive_gammas()
    with open('./simulated/reg/postalphas.pkl', 'wb') as file:
        pickle.dump(postalphas, file)
    pd.DataFrame(res.samples.delta).to_csv('./simulated/reg/postdeltas.csv', index = False)
    
    if False:
        slosh  = pd.read_csv(
            './datasets/slosh/filtered_data.csv.gz', 
            compression = 'gzip',
            )
        sloshx = pd.read_csv('./datasets/slosh/slosh_params.csv')

    if False: # sloshltd    
        # sloshltd  = ~slosh.MTFCC.isin(['C3061','C3081'])
        sloshltd = slosh.MTFCC.isin(['K2451'])
        sloshltd_ids = slosh[sloshltd].iloc[:,:8]                             # location parms
        sloshltd_obs = slosh[sloshltd].iloc[:,8:].values.astype(np.float64).T # storm runs

        sloshx.theta.loc[sloshx.theta < 100] += 360
        sloshx_par    = summarize(sloshx.values)
        locatx_par    = summarize(sloshltd_ids[['x','y']].values)
        sloshx_par.mean[-1] = locatx_par.mean[-1] # latitude values will be 
        sloshx_par.sd[-1]   = locatx_par.sd[-1]   # on same scale for both datasets
        sloshx_std    = scale(sloshx.values, sloshx_par)
        locatx_std    = scale(sloshltd_ids[['x','y']].values, locatx_par)
        
        x_observation = sloshx_std
        x_location    = locatx_std
        x_interaction = (sloshx_std[:,-1][None] * locatx_std[:,-1][:,None])[:,:,None]

        data = RegressionData(
            raw_real    = sloshltd_obs, 
            real_type   = 'threshold',
            decluster   = False, 
            quantile    = 0.90,
            observation = x_observation,
            location    = x_location,
            interaction = x_interaction,
            )
        model = Chain(data, p = 10)
        model.sample(20000, verbose = True)
        model.write_to_disk('./test/results.pkl', 1, 1)
        res = Result('./test/results.pkl')
        postalphas = res.generate_conditional_posterior_predictive_gammas()
        with open('./test/conpostpredgammas.pkl', 'wb') as file:
            pickle.dump(postalphas, file)
        model.mh_accept_rate()
        
        # inputs = pd.read_csv('~/git/surge/data/inputs.csv')
        # finputs = inputs.iloc[model.data.I]

        # deltas = model.generate_conditional_posterior_deltas()
        # import posterior as post
        # smat   = post.similarity_matrix(deltas)
        # graph  = post.minimum_spanning_trees(smat)
        # g      = pd.DataFrame(graph)
        # g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
        # # write to disk

        # d = {
        #         'ids'    : sloshltd_ids,
        #         'obs'    : sloshltd_obs,
        #         # 'alphas' : postalphas,
        #         'inputs' : finputs,
        #         'deltas' : deltas,
        #         'smat'   : smat,
        #         'graph'  : g,
        #         }
        #     with open('./datasets/slosh/sloshltd.pkl', 'wb') as file:
        #         pkl.dump(d, file)
        #     g.to_csv('./datasets/slosh/sloshltd_mst.csv', index = False)
        #     finputs.to_csv('./datasets/slosh/sloshltd_in.csv', index = False)
        #     pd.DataFrame(deltas).to_csv('./datasets/slosh/sloshltd_delta.csv', index = False)
            
        #     deltastar = post.emergent_clusters_pre(model)
        #     pd.DataFrame({'obs' : np.arange(model.N), 'cid' : deltastar}).to_csv(
        #         './datasets/slosh/sloshltd_cluster_pre.csv', index = False,
        #         )
        #     deltastar_ = post.emergent_clusters_post(model)
        #     pd.DataFrame({'obs' : np.arange(model.N), 'cid' : deltastar_}).to_csv(
        #         './datasets/slosh/sloshltd_clusters_post.csv', index = False, 
        #         )

    if False: # Full model
        data = Data_From_Raw(
            slosh_obs.values.T.astype(np.float64), 
            decluster = False, 
            quantile = 0.99,
            )
        model = vb.VarPYPG(data)
        model.fit_advi()

        inputs = pd.read_csv('~/git/surge/data/inputs.csv')
        finputs = inputs.iloc[model.data.I]

        deltas = model.generate_conditional_posterior_deltas()
        import posterior as post
        smat   = post.similarity_matrix(deltas)
        graph  = post.minimum_spanning_trees(smat)
        g      = pd.DataFrame(graph)
        g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
        # write to disk

        d = {
            'ids'    : slosh_ids,
            'obs'    : slosh_obs,
            'inputs' : finputs,
            'deltas' : deltas,
            'smat'   : smat,
            'graph'  : g,
            }
        with open('./datasets/slosh/slosh.pkl', 'wb') as file:
            pkl.dump(d, file)
        g.to_csv('./datasets/slosh/slosh_mst.csv', index = False)
        finputs.to_csv('./datasets/slosh/slosh_in.csv', index = False)
        pd.DataFrame(deltas).to_csv('./datasets/slosh/slosh_delta.csv', index = False)
        
        deltastar = post.emergent_clusters_pre(model)
        pd.DataFrame({'obs' : np.arange(model.N), 'cid' : deltastar}).to_csv(
            './datasets/slosh/slosh_cluster_pre.csv', index = False,
            )
        deltastar_ = post.emergent_clusters_post(model)
        pd.DataFrame({'obs' : np.arange(model.N), 'cid' : deltastar_}).to_csv(
            './datasets/slosh/slosh_clusters_post.csv', index = False, 
            )

    if False: # filtered to 0.9 threshold
        slosh9_obs = pd.read_csv(
            './datasets/slosh/slosh_thr.90.csv.gz', 
            compression = 'gzip',
            )
        data = Data_From_Raw(
            slosh9_obs.values.T.astype(np.float64), 
            decluster = False, 
            quantile = 0.90,
            )
        model = vb.VarPYPG(data)
        model.fit_advi()

        inputs = pd.read_csv('~/git/surge/data/inputs.csv')
        finputs = inputs.iloc[model.data.I]

        deltas = model.generate_conditional_posterior_deltas()
        import posterior as post
        smat   = post.similarity_matrix(deltas)
        graph  = post.minimum_spanning_trees(smat)
        g      = pd.DataFrame(graph)
        g = g.rename(columns = {0 : 'node1', 1 : 'node2', 2 : 'weight'})
        # write to disk

        d = {
            'ids'    : slosh_ids,
            'obs'    : slosh9_obs,
            'inputs' : finputs,
            'deltas' : deltas,
            'smat'   : smat,
            'graph'  : g,
            }
        with open('./datasets/slosh/slosht90.pkl', 'wb') as file:
            pkl.dump(d, file)
        g.to_csv('./datasets/slosh/slosht90_mst.csv', index = False)
        finputs.to_csv('./datasets/slosh/slosht90_in.csv', index = False)
        pd.DataFrame(deltas).to_csv('./datasets/slosh/slosht90_delta.csv', index = False)
        
        deltastar = post.emergent_clusters_pre(model)
        pd.DataFrame({'obs' : np.arange(model.N), 'cid' : deltastar}).to_csv(
            './datasets/slosh/slosht90_cluster_pre.csv', index = False,
            )
        deltastar_ = post.emergent_clusters_post(model)
        pd.DataFrame({'obs' : np.arange(model.N), 'cid' : deltastar_}).to_csv(
            './datasets/slosh/slosht90_clusters_post.csv', index = False, 
            )

# EOF
