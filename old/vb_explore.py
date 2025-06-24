from numpy.random import gamma, normal
from collections import namedtuple, deque
import numpy as np
import pandas as pd
from scipy.special import gammaln, digamma
from data import euclidean_to_psphere
from samplers import BaseSampler
from projgamma import GammaPrior
from model_spypprg_vb import gradient_resgammagamma_ln, gradient_gammagamma_ln, Adam


class RGGSamples(object):
    r     : deque
    zeta  : deque
    
    def __init__(self, nkeep : int, N, S):
        self.r    = deque([], maxlen = nkeep)
        self.zeta = deque([], maxlen = nkeep)
        self.r.append(gamma(shape = 4, scale = 1/2, size = N))
        self.zeta.append(gamma(shape = 2., scale = 1/2., size = (1,S)))
        return

class MultivariateResGammaGamma(BaseSampler):
    lYs : np.ndarray
    n   : np.ndarray
    samples : RGGSamples
    adam : Adam
    zeta_mutau : np.ndarray
    a : np.ndarray
    b : np.ndarray
    varsamp : int = 20

    @property
    def curr_zeta(self):
        return np.exp(
            self.zeta_mutau[0] + 
            np.exp(self.zeta_mutau[1]) * normal(size = self.zeta_mutau[1].shape)
            )
    
    def dloss(self):
        return - gradient_resgammagamma_ln(
            self.zeta_mutau, self.lYs, self.n, self.a, self.b, self.varsamp
            )
    
    def update_statistics(self, Y):
        self.lYs[:] = np.log(Y)
        self.n[:]   = np.array(Y.shape[0],)
        return

    def initialize_sampler(self, ns):
        self.samples = RGGSamples(1000, self.N, self.S)
        self.adam = Adam(self.zeta_mutau)
        self.adam.specify_dloss(self.dloss)
        self.curr_iter = 0
        return
    
    def update_zeta(self):
        self.adam.optimize()
        self.samples.zeta.append(self.curr_zeta)
        return
    
    def iter_sample(self):
        self.curr_iter += 1
        self.update_zeta()
        return

    def __init__(self, Y : np.ndarray, a : float, b : float):
        self.N, self.S = Y.shape
        self.lYs = np.log(Y).sum(axis = 0).reshape(1,-1)
        self.n = np.array(self.N,)
        self.zeta_mutau = np.zeros((2, 1, self.S))
        self.a = np.ones(self.S) * a
        self.b = np.ones(self.S) * b
        return

class MultivariateResProjgammaGamma(MultivariateResGammaGamma):
    @property
    def curr_r(self):
        return self.samples.r[-1]

    def dloss(self):
        return - gradient_resgammagamma_ln(
            self.zeta_mutau, self.lYs, self.n, self.a, self.b, self.varsamp
            )

    def update_r(self, zeta : np.ndarray):
        As = zeta.sum(axis = -1)
        Bs = self.Yp.sum(axis = -1)
        r = gamma(As, scale = 1/Bs)
        r[r < 1e-5] = 1e-5
        self.samples.r.append(r)
        return

    def update_statistics(self, r):
        Y = self.curr_r[:,None] * self.Yp
        self.lYs[:] = np.log(Y).sum(axis = 0).reshape(1,-1)
        return

    def update_zeta(self, r):
        self.update_statistics(r)
        self.adam.optimize()
        self.samples.zeta.append(self.curr_zeta)
        return
    
    def iter_sample(self):
        r    = self.curr_r
        zeta = self.curr_zeta

        self.curr_iter += 1
        
        self.update_zeta(r)
        self.update_r(zeta)
        return

    def __init__(self, Yp : np.ndarray, a : int, b : int):
        self.Yp         = Yp
        self.N, self.S  = self.Yp.shape
        self.zeta_mutau = np.zeros((2, 1, self.S))

        self.lYs = np.zeros((1, self.S))
        self.n   = np.array((self.N,))
        self.a   = np.ones(self.S) * a
        self.b   = np.ones(self.S) * b
        return


if __name__ == '__main__': 
    gammavars = np.array((5., 3., 0.5))
    Y  = gamma(shape = gammavars, size = (500, gammavars.shape[0]))
    Yp = euclidean_to_psphere(Y)
    # m = MultivariateResGammaGamma(Y, 1., 1.)
    m = MultivariateResProjgammaGamma(Yp, 1., 1.)
    
    print('Optimization')
    for _ in range(10):
        m.sample(100)
        print(m.zeta_mutau.ravel())
    print('target: {}'.format(np.log(gammavars)))