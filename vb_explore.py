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
        self.zeta.append(gamma(shape = 2., scale = 1/2., size = S))
        return

class MultivariateResGammaGamma(BaseSampler):
    samples    : RGGSamples
    adam       : Adam
    zeta_mutau : np.ndarray
    a          : np.ndarray
    b          : np.ndarray

    @property
    def curr_r(self):
        return self.samples.r[-1]
    @property
    def curr_zeta(self):
        return np.exp(self.zeta_mutau[0] + self.zeta_mutau[1] * normal(size = self.S))

    def initialize_sampler(self, ns):
        self.samples = RGGSamples(1000, self.N, self.S)
        self.adam = Adam(self.zeta_mutau)
        self.curr_iter = 0
        return

    def update_r(self, zeta : np.ndarray):
        As = zeta.sum(axis = -1)
        Bs = self.Yp.sum(axis = -1)
        self.samples.r.append(gamma(As, scale = 1 / Bs))
        return

    def update_zeta(self, r):
        Y = r[:,None] * self.Yp
        lYs = np.log(Y).sum(axis = 0).reshape(1,-1)
        n = np.array((self.N,))
        self.adam.specify_dloss(
            lambda theta: gradient_resgammagamma_ln(
                self.zeta_mutau, lYs, n, self.a, self.b,
                )
            )
        self.adam.optimize()
        return
    
    def iter_sample(self):
        r    = self.curr_r
        zeta = self.curr_zeta

        self.curr_iter += 1
        
        self.update_zeta(r)
        self.update_r(zeta)
        return

    def __init__(self, Yp : np.ndarray, a : int, b : int):
        self.Yp = Yp
        self.N, self.S = self.Yp.shape        
        self.zeta_mutau = np.zeros((2, 1, self.S))
        self.a = np.ones(self.S) * a
        self.b = np.ones(self.S) * b
        return

if __name__ == '__main__': 
    gammavars = np.array((5., 3., 0.5))
    Yp = euclidean_to_psphere(gamma(gammavars[0], size = (500, gammavars.shape[0])))
    m = MultivariateResGammaGamma(Yp, 1., 1.)
    m.sample(1000)
    print(m.zeta_mutau)
    print(np.log(gammavars))
    pass
