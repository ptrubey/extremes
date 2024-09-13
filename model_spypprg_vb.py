from numpy.random import choice, gamma, beta, uniform, normal, lognormal
from collections import namedtuple, deque
from itertools import repeat
import numpy as np
np.seterr(divide = 'raise', over = 'raise', under = 'ignore', invalid = 'raise')
import pandas as pd
import os
import pickle
from math import log
from scipy.special import gammaln, digamma
from io import BytesIO
from cUtility import pityor_cluster_sampler, generate_indices

import samplers as samp
from data import euclidean_to_hypercube, Data_From_Sphere, DataBase
from projgamma import GammaPrior, logd_projgamma_my_mt_inplace_unstable


Prior = namedtuple('Prior','alpha beta')
class Samples(object):
    r     : deque      # radius (projected gamma)
    chi   : deque      # stick-breaking weights (unnormalized)
    delta : deque   # cluster identifiers
    beta  : deque   # rate hyperparameter
    
    def __init__(
            self, 
            nkeep : int, 
            N : int,  # nDat
            S : int,  # nCol
            J : int,  # nClust
            ):
        self.r     = deque([], maxlen = nkeep)
        self.chi   = deque([], maxlen = nkeep)
        self.delta = deque([], maxlen = nkeep)
        self.beta  = deque([], maxlen = nkeep)
        self.r.append(lognormal(mean = 3, sigma = 1, size = N))
        self.chi.append(1 / np.arange(2, J + 1)[::-1]) # uniform probability
        self.delta.append(choice(J, N))
        self.beta.append(gamma(shape = 2, scale = 1 / 2, size = S))
        return
    pass

def stickbreak(nu):
    """
        Stickbreaking cluster probability
        nu : (S x (J - 1))
    """
    lognu = np.log(nu)
    log1mnu = np.log(1 - nu)

    S = nu.shape[0]; J = nu.shape[1] + 1
    out = np.zeros((S,J))
    out[:,:-1] += lognu
    out[:, 1:] += np.cumsum(log1mnu, axis = -1)
    return np.exp(out)

def gradient_resgammagamma_ln(
        theta   : np.ndarray,  # np.stack((mu, tau))
        lYs     : np.ndarray,  # sum of log(Y)
        n       : np.ndarray,  # number of observations
        a       : np.ndarray,  # hierarchical shape 
        b       : np.ndarray,  # hierarchical rate
        ns = 10,
        ):
    epsilon = normal(size = (ns, *theta.shape[1:]))
    ete = np.exp(theta[1]) * epsilon
    alpha = np.exp(theta[0] + ete)

    dtheta = np.zeros((ns, *theta.shape))
    dtheta += alpha[:,None] * lYs
    dtheta -= n[None,None,:,None] * digamma(alpha[:,None]) * alpha[:,None]
    dtheta += (a - 1)
    dtheta -= b * alpha[:,None]
    
    dtheta[:,1] *= ete
    
    dtheta[:,0] -= -1
    dtheta[:,1] -= -1 - np.exp(2 * theta[1])
    return dtheta.mean(axis = 0)

def gradient_gammagamma_ln(
        theta   : np.ndarray,  # np.stack((mu, tau))
        lYs     : np.ndarray,  # sum of log(Y)
        Ys      : np.ndarray,  # sum of Y
        n       : np.ndarray,  # number of observations
        a       : np.ndarray,  # hierarchical (shape) shape 
        b       : np.ndarray,  # hierarchical (shape) rate
        c       : np.ndarray,  # hierarchical (rate) shape
        d       : np.ndarray,  # hierarchical (rate) rate
        ns = 10,
        ):
    epsilon = normal(size = (ns, *theta.shape[1:]))
    ete = np.exp(theta[1]) * epsilon
    alpha = np.exp(theta[0] + ete)

    dtheta = np.zeros((ns, *theta.shape))
    dtheta += alpha[:,None] * lYs
    dtheta -= n * digamma(alpha[:,None]) * alpha[:,None]
    dtheta += (a - 1)
    dtheta -= b * alpha[:,None]
    dtheta += digamma(n * alpha[:,None] + c)
    dtheta -= (n * alpha[:,None] + c) * np.log(Ys + d)
    
    dtheta[:,1] *= ete
    
    dtheta[:,0] -= -1
    dtheta[:,1] -= -1 - np.exp(2 * theta[1])
    return dtheta.mean(axis = 0)

class Adam(object):
    # Adam Parameters
    eps    : float = 1e-8
    rate   : float
    decay1 : float
    decay2 : float
    iter   : int
    niter  : int

    # Adam Updateables 
    momentum : np.ndarray # momentum
    sumofsqs : np.ndarray # sum of squares of past gradients
    
    # Loss function
    dloss = None   # function of theta

    def update(self):
        self.iter += 1
        dloss = self.dloss(self.theta)
        self.momentum[:] = (
            + self.decay1 * self.momentum
            + (1 - self.decay1) * dloss
            )
        self.sumofsqs[:] = (
            + self.decay2 * self.sumofsqs
            + (1 - self.decay2) * dloss * dloss
            )
        self.theta -= (
            (self.momentum / (1 - self.decay1**self.iter)) * self.rate / 
            (np.sqrt(self.sumofsqs / (1 - self.decay2**self.iter )) + self.eps)
            )
        return
    
    def specify_dloss(self, func):
        self.dloss = func

    def initialization(
            self, 
            rate   : float, # Adam Learning Rate
            decay1 : float, # Adam Decay 1
            decay2 : float, # Adam Decay 2
            niter  : int,   # Number of Adam Iterations per sample
            ):
        self.decay1 = decay1
        self.decay2 = decay2
        self.rate = rate
        self.iter = 0
        self.niter = niter
        self.momentum = np.zeros(self.theta.shape)
        self.sumofsqs = np.zeros(self.theta.shape)
        return

    def optimize(self):
        for _ in range(self.niter):
            self.update()
        return
    
    def __init__(
            self, 
            theta, 
            rate   = 1e-3, 
            decay1 = 0.9, 
            decay2 = 0.999, 
            niter  = 10,
            ):
        self.theta = theta
        self.initialization(rate, decay1, decay2, niter)
        return

class VariationalParameters(object):
    zeta_mutau   : np.ndarray
    zeta_adam    : Adam
    alpha_mutau  : np.ndarray
    alpha_adam   : Adam

    def __init__(self, S : int, J : int, **kwargs):
        self.zeta_mutau = normal(size = (2, J, S))
        self.alpha_mutau = normal(size = (2, S))

        self.zeta_adam = Adam(self.zeta_mutau, **kwargs)
        self.alpha_adam = Adam(self.alpha_mutau, **kwargs)
        return
    pass

class Chain(samp.StickBreakingSampler):
    samples         : Samples
    varparm         : VariationalParameters
    priors          : Prior
    concentration   : float
    discount        : float
    N               : int
    J               : int
    data            : DataBase
    curr_iter       : int

    @property
    def curr_r(self):
        return self.samples.r[-1]
    @property
    def curr_chi(self):
        return self.samples.chi[-1]
    @property
    def curr_delta(self):
        return self.samples.delta[-1]
    @property
    def curr_alpha(self):
        return lognormal(
            mean = self.varparm.alpha_mutau[0], 
            sigma = np.exp(self.varparm.alpha_mutau[1])
            )
    @property
    def curr_beta(self):
        return self.samples.beta[-1]
    @property
    def curr_zeta(self):
        return lognormal(
            mean = self.varparm.zeta_mutau[0], 
            sigma = np.exp(self.varparm.zeta_mutau[1]),
            )
    
    def update_zeta(
            self, 
            delta : np.ndarray, 
            r     : np.ndarray, 
            alpha : np.ndarray, 
            beta  : np.ndarray,
            ):
        dmat = delta[:,None] == np.arange(self.J)
        Y = r[:,None] * self.data.Yp
        n = dmat.sum(axis = 0)
        lYs = (np.log(Y).T @ dmat).T
        
        func = lambda theta: - gradient_resgammagamma_ln(
            theta, lYs, n, alpha, beta,
            )

        self.varparm.zeta_adam.specify_dloss(func)
        self.varparm.zeta_adam.optimize()
        replace = np.where(n == 0)[0]
        # self.varparm.zeta_mutau[0,replace] = 0.
        # self.varparm.zeta_mutau[1,replace] = 3.
        return
    
    def update_alpha(
            self, 
            zeta : np.ndarray, 
            delta : np.ndarray,
            ):
        active = np.where(np.bincount(delta, minlength = self.J) > 0)[0]
        n = active.shape[0]
        lZs = np.log(zeta)[active].sum(axis = 0)
        Zs  = zeta[active].sum(axis = 0)

        func = lambda theta: - gradient_gammagamma_ln(
            theta, lZs, Zs, n,
            *self.priors.alpha,
            *self.priors.beta,
            ns = self.var_samp
            )
                
        self.varparm.alpha_adam.specify_dloss(func)
        self.varparm.alpha_adam.optimize()
        return

    def update_beta(
            self, 
            zeta : np.ndarray, 
            alpha : np.ndarray, 
            delta : np.ndarray,
            ):
        active = np.where(np.bincount(delta, minlength = self.J) > 0)[0]
        n = active.shape[0]
        zs = zeta[active].sum(axis = 0)
        As = n * alpha + self.priors.beta.a
        Bs = zs + self.priors.beta.b
        self.samples.beta.append(gamma(shape = As, scale = 1 / Bs))
        return

    def update_r(self, zeta : np.ndarray, delta : np.ndarray):
        As = zeta[delta].sum(axis = -1)  # np.einsum('il->i', zeta[delta])
        Bs = self.data.Yp.sum(axis = -1) # np.einsum('il->i', self.data.Yp)
        self.samples.r.append(gamma(shape = As, scale = 1 / Bs))
        return
    
    def update_chi(self, delta : np.ndarray):
        chi = samp.py_sample_chi_bgsb_fixed(
            delta, self.discount, self.concentration, self.J,
            )
        self.samples.chi.append(chi)
        return
    
    def update_delta(self, zeta : np.ndarray, chi : np.ndarray):
        llk = np.zeros((self.N, self.J))
        logd_projgamma_my_mt_inplace_unstable(
            llk, self.data.Yp, zeta, np.ones(zeta.shape),
            )
        delta = samp.py_sample_cluster_bgsb_fixed(chi, llk)
        self.samples.delta.append(delta)
        return

    def iter_sample(self):
        chi   = self.curr_chi
        alpha = self.curr_alpha
        beta  = self.curr_beta
        zeta  = self.curr_zeta
        self.curr_iter += 1

        self.update_delta(zeta, chi)
        self.update_chi(self.curr_delta)
        self.update_r(zeta, self.curr_delta)
        self.update_zeta(self.curr_delta, self.curr_r, alpha, beta)

        zeta = self.curr_zeta
        self.update_alpha(zeta, self.curr_delta)
        self.update_beta(zeta, self.curr_alpha, self.curr_delta)
        return

    def initialize_sampler(self, nSamp : int):
        self.samples = Samples(self.gibbs_samp, self.N, self.S, self.J)
        self.varparm = VariationalParameters(
            self.S, self.J,
            niter = self.var_iter,
            )
        self.curr_iter = 0
        pass

    def set_projection(self):
        self.data.Yp = (
            self.data.V.T / 
            (self.data.V ** self.p).sum(axis = 1)**(1/self.p)
            ).T
        return
    
    def write_to_disk(self, path):
        if type(path) is str:
            folder = os.path.split(path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            if os.path.exists(path):
                os.remove(path)
        out = {
            'zeta_mutau'  : self.varparm.zeta_mutau,
            'alpha_mutau' : self.varparm.alpha_mutau,
            'betas'       : np.stack(self.samples.beta),
            'rs'          : np.stack(self.samples.r),
            'deltas'      : np.stack(self.samples.delta),
            'chis'        : np.stack(self.samples.chi),
            'nCol'        : self.S,
            'nDat'        : self.N,
            'time'        : self.time_elapsed_numeric,
            'conc'        : self.concentration,
            'disc'        : self.discount,
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
    
    def __init__(
            self, 
            data, 
            variational_samples = 10, 
            variational_iterations_per = 10,
            gibbs_samples = 1000,
            max_clusters = 200,
            p = 10,
            prior_alpha = (0.5, 0.5),
            prior_beta = (2., 2.),
            concentration = 0.1, 
            discount = 0.1,
            ):
        self.data = data
        self.N = self.data.nDat
        self.S = self.data.nCol
        self.J = max_clusters
        self.p = p
        self.concentration = concentration
        self.discount = discount
        self.var_samp = variational_samples
        self.var_iter = variational_iterations_per
        self.gibbs_samp = gibbs_samples
        self.priors = Prior(GammaPrior(*prior_alpha), GammaPrior(*prior_beta))
        self.set_projection()
        return

class ResultSamples(Samples):
    r     : np.ndarray
    chi   : np.ndarray
    delta : np.ndarray
    beta  : np.ndarray
    alpha : np.ndarray
    zeta  : np.ndarray

    def __init__(self, dict):
        self.r     = dict['rs']
        self.chi   = dict['chis']
        self.delta = dict['deltas']
        self.beta  = dict['betas']
        self.alpha = lognormal(
            mean = dict['alpha_mutau'][0], 
            sigma = np.exp(dict['alpha_mutau'][1]),
            size = (self.r.shape[0], *dict['alpha_mutau'][0].shape),
            )
        self.zeta  = lognormal(
            mean = dict['zeta_mutau'][0],
            sigma = np.exp(dict['zeta_mutau'][1]),
            size = (self.r.shape[0], *dict['zeta_mutau'][0].shape),
            )
        return

class Result(object):
    samples : ResultSamples
    discount : float
    concentration : float
    time_elapsed_numeric : float
    N : int
    S : int
    J : int

    def generate_conditional_posterior_predictive_zetas(self):
        zetas = np.swapaxes(np.array([
            zeta[delta]
            for zeta, delta
            in zip(self.samples.zeta, self.samples.delta)
            ]), 0, 1)
        return zetas
    
    def generate_conditional_posterior_predictive_gammas(self):
        zetas = self.generate_conditional_posterior_predictive_zetas()
        return gamma(shape = zetas)
    
    def generate_posterior_predictive_zetas(self, n_per_sample = 10):
        zetas = []
        probs = stickbreak(self.samples.chi)
        Sprob = np.cumsum(probs, axis = -1)
        unis  = uniform(size = (self.nSamp, n_per_sample))
        for s in range(self.nSamp):
            delta = np.searchsorted(unis[s], probs[s])
            zetas.append(self.samples.zeta[delta])
        return np.stack(zetas)
    
    def generate_posterior_predictive_gammas(self, n_per_sample = 10):
        zetas = self.generate_posterior_predictive_zetas(n_per_sample)
        return gamma(shape = zetas)

    def generate_posterior_predictive_hypercube(self, n_per_sample = 10):
        gammas = self.generate_conditional_posterior_predictive_gammas(n_per_sample)
        return euclidean_to_hypercube(gammas)

    def load_data(self, path):
        if type(path) is BytesIO:
            out = pickle.loads(path.getvalue())
        else:
            with open(path, 'rb') as file:
                out = pickle.load(file)
        self.samples = ResultSamples(out)
        self.concentration = out['conc']
        self.discount = out['disc']
        self.N = out['nDat']
        self.S = out['nCol']
        self.nSamp = self.samples.chi.shape[0]
        self.time_elapsed_numeric = out['time']
        return

    def __init__(self, path):
        self.load_data(path)
        return
    
if __name__ == '__main__':
    pass
    # from data import Data_From_Raw
    # raw = pd.read_csv('./datasets/ivt_updated_nov_mar.csv')
    # data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    # model = Chain(data, p = 10, gibbs_samples = 1000,)
    # model.sample(5000, verbose = True)
    # model.write_to_disk('./test/results.pkl')
    # res = Result('./test/results.pkl')
    # cond_zetas  = res.generate_conditional_posterior_predictive_zetas()
    # cond_gammas = res.generate_conditional_posterior_predictive_gammas()
    # zetas       = res.generate_posterior_predictive_zetas()
    # gammas      = res.generate_posterior_predictive_gammas()

# EOF