"""
Implementation of SPYPPRG-R

Regression extension of SPYPPRG.

"""
from numpy.random import choice, gamma, beta, uniform
from collections import namedtuple
from itertools import repeat
import numpy as np
np.seterr(divide='raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log
from scipy.special import gammaln
from scipy.stats import invwishart
from numpy.linalg import cholesky
from io import BytesIO

from cUtility import pityor_cluster_sampler, generate_indices
from samplers import DirichletProcessSampler
from data import euclidean_to_angular, euclidean_to_hypercube, Data_From_Sphere
from projgamma import NormalPrior, InvWishartPrior,                             \
    pt_logd_projgamma_my_mt_inplace_unstable, logd_gamma,                       \
    logd_prodgamma_my_mt, logd_prodgamma_paired, logd_prodgamma_my_st
from cov import OnlineCovariance

def softplus(X : np.ndarray):
    """ 
    softplus function -- less agressive than log-transformation for 
        unbounding X to the real number line.
    """
    return np.log(1. + np.exp(X))

NIWPrior = namedtuple('NIWPrior','mu kappa nu lambda')
Prior = namedtuple('Prior', 'muSig')
Dimensions = namedtuple('Dimensions','beta gamma zeta')

class Samples(object):
    theta = None
    epsilon = None
    Sigma = None
    mu    = None
    delta = None
    r     = None
    ld    = None
    
    def __init__(self, nSamp, nDat, nCol):
        self.theta = [None] * (nSamp + 1)
        self.epsilon = np.empty((nSamp + 1, nCol))
        self.Sigma = np.empty((nSamp + 1, nCol, nCol))
        self.mu    = np.empty((nSamp + 1, nCol))
        self.delta = np.empty((nSamp + 1, nDat), dtype = int)
        self.r     = np.empty((nSamp + 1, nDat))
        self.ld    = np.empty((nSamp + 1))
        return

class Chain(DirichletProcessSampler):
    concentration = None
    discount      = None

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

    def compute_shape(
            self, 
            delta   : np.ndarray,
            theta   : np.ndarray,
            epsilon : np.ndarray,
            ):
        # Parsing the theta vector
        beta  = theta[:,self.bounds.beta[0]:self.bounds.beta[1]]
        gamma = theta[:,self.bounds.gamma[0]:self.bounds.gamma[1]]
        zeta  = theta[:,self.bounds.zeta[0]:self.bounds.zeta[1]]
        # Computing shape parameters
        Ase = np.zeros((self.N, self.S))
        Ase += np.einsum('nd,nd->n', self.data.theta, beta[delta])[:,None] # [n,d1]->[n]
        Ase += np.einsum('sd,sd->s', self.data.chi, gamma[delta])[None]    # [s,d2]->[s]
        Ase += np.einsum('nsd,sd->ns', self.data.phi, zeta[delta])     # [n,s,d3]->[n,s]
        Ase += epsilon[None]
        return Ase

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
        Ase = self.compute_shape(delta, theta, epsilon)
        As = self.linkfn(Ase).sum(axis = -1) # sum over last dimension
        Bs = self.data.Yp.sum(axis = -1)     # Sum over last dimension
        return gamma(shape = As, scale = 1 / Bs)
    
    def sample_theta(
            self,
            delta   : np.ndarray, # cluster ID
            theta   : np.ndarray, # regression coef's
            epsilon : np.ndarray, # location specific effect
            mu      : np.ndarray, # mean of centering distribution
            Sigma   : np.ndarray, # cov of centering distribution
            ):
        Ase_curr = self.compute_shape(delta, theta, epsilon)
        Ase_cand = None # current stopping point
        return

    def sample_theta_new(
            self, 
            mu    : np.ndarray, 
            Sigma : np.ndarray, 
            m     : int,
            ):
        out = np.zeros((m, self.D))
        out += (cholesky(Sigma) @ np.random.normal(size = (self.D, self.m))).T
        out += mu[None]
        return out

    def sample_mu_Sigma(
            self,
            theta : np.ndarray, 
            Sigma : np.ndarray,
            ):
        # Parsing prior parameters
        mu_0, ka_0, nu_0, la_0 = self.priors.muSig
        tbar = theta.mean(axis = 0)
        S1 = (theta - tbar[None]).T @ (theta - tbar[None])
        S2 = (tbar - mu_0)[:,None] @ (tbar - mu_0)[None]
        n = theta.shape[0]
        # Posterior Parameters
        mu_n = (ka_0 / (ka_0 + n)) * mu_0 + (n / (ka_0 + n)) * tbar
        ka_n = ka_0 + n
        nu_n = nu_0 + n
        la_n = la_0 + S1 + (k_0 * n)/(k_0 + n) * S2
        Sigma = invwishart.rvs(df = nu_n, scale = la_n)
        C = cholesky(Sigma / ka_n)
        mu = mu_n + C @ np.random.normal(size = self.D)
        return mu, Sigma

    def initialize_sampler(
            self, 
            ns : int,
            ):
        self.samples = Samples(ns, self.N, self.D)
        self.samples.delta[0] = choice(self.J - 30, size = self.N)
        self.samples.theta[0] = np.random.normal(
            loc = 0, scale = 1, size = (self.J, self.D),
            )
        self.samples.epsilon[0] = np.random.normal(
            loc = 0, scale = 1, size = (self.S),
            )
        self.samples.r[0] = self.sample_r(
            self.samples.delta[0], self.samples.theta[0], self.samples.epsilon[0],
            )
        self.samples.mu[0]    = np.random.normal(size = self.D)
        self.samples.Sigma[0] = np.eye(self.D)
        self.curr_iter = 0
        self.rate_placeholder_1 = np.ones((self.J, self.D))
        self.rate_placeholder_2 = np.ones((self.N, self.D))
        return
    
    def record_log_density(self):
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
        # augment theta with new draws
        theta = np.concatenate(
            theta,
            self.sample_theta_new(mu, Sigma, m)
            )

        # Sample Delta
        log_likelihood = logd_prodgamma_my_mt(
            r[:,None] * self.data.Yp, 
            self.compute_shape(delta, theta, epsilon),
            self.rate_placeholder_1,
            )
        unifs = uniform(size = self.nDat)
        delta = pityor_cluster_sampler(
            delta, log_likelihood, unifs, self.concentration, self.discount,
            )
        # clear out dropped/unused clusters and re-index
        delta, theta = self.clean_delta_theta(delta,theta)
        self.samples.delta[ci] = delta
        # sample new parameters
        self.samples.r[ci]     = self.sample_r(self.curr_delta, theta)
        self.samples.theta[ci] = self.sample_theta(
            self.curr_delta, theta, epsilon, mu, Sigma,
            )
        self.samples.mu[ci], self.samples.Sigma[ci] = self.sample_mu_Sigma(
            self.curr_beta, self.curr_gamma, self.curr_zeta,
            )
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
        alphas = self.samples.alpha[nBurn :: nThin]
        betas  = self.samples.beta[nBurn :: nThin]
        deltas = self.samples.delta[nBurn :: nThin]
        rs     = self.samples.r[nBurn :: nThin]

        out = {
            'zetas'  : zetas,
            'alphas' : alphas,
            'betas'  : betas,
            'rs'     : rs,
            'deltas' : deltas,
            'nCol'   : self.nCol,
            'nDat'   : self.nDat,
            'V'      : self.data.V,
            'logd'   : self.samples.ld,
            'time'   : self.time_elapsed_numeric,
            'conc'   : self.concentration,
            'disc'   : self.discount,
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
            self.data.x1.shape[-1],
            self.data.x2.shape[-1],
            self.data.x3.shape[-1],
            )
        self.D = sum(self.d)
        return

    def __init__(
            self,
            data,
            prior_mu_Sigma = (0, 1, 10, 10),
            p             = 10,
            concentration = 0.2,
            discount      = 0.2,
            max_clust_count = 200,
            link_fn       = softplus,
            **kwargs
            ):
        assert all(type(x) is np.ndarray for x in data.X)
        # assert type(data.theta) is np.ndarray   # storm effects
        # assert type(data.chi)   is np.ndarray   # location effects
        # assert type(data.phi)   is np.ndarray   # storm-location interaction effects
        self.data = data
        self.set_shapes()
        # Parsing the inputs
        self.J = max_clust_count
        self.concentration = concentration
        self.discount = discount
        self.p = p
        # Setting the priors
        self.priors = Prior(NIWPrior(*prior_mu_Sigma))
        # Rest of setup
        self.set_projection()
        self.linkfn = link_fn
        self.cov = OnlineCovariance(self.D)
        return

class Result(object):
    def generate_conditional_posterior_predictive_gammas(self):
        """ rho | zeta, delta + W ~ Gamma(rho | zeta[delta] + W) """
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for delta, zeta 
            in zip(self.samples.delta, self.samples.zeta)
            ]),0,1) # (n,s,d)
        return gamma(shape = zetas)

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
        
        deltas = out['deltas']
        zetas  = out['zetas']
        alphas = out['alphas']
        betas  = out['betas']
        rs     = out['rs']
        conc   = out['conc']
        disc   = out['disc']

        self.concentration = conc
        self.discount      = disc
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
        self.samples.alpha = alphas
        self.samples.beta  = betas
        self.samples.zeta  = [
            zetas[np.where(zetas.T[0] == i)[0], 1:] for i in range(self.nSamp)
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
    model = Chain(data, p = 10)
    model.sample(10000, verbose = True)
    model.write_to_disk('./test/results.pkl', 5000, 2)
    res = Result('./test/results.pkl')

# EOF
