""" Comparison of MCMC and VB on a univariate gamma-gamma model. """
import numpy as np
from numpy.random import normal, uniform, gamma
from scipy.special import gammaln, digamma
import matplotlib.pyplot as plt

from samplers import BaseSampler

class Samples(object):
    def __init__(self, ns):
        self.alpha = np.empty(ns + 1)
        return
    pass

class UnivariateGammaGammaMC(BaseSampler):
    @property
    def curr_alpha(self):
        return self.samples.alpha[self.curr_iter]
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns)
        self.samples.alpha[0] = 1.
        self.curr_iter = 0
        return

    def log_posterior_alpha(self, sigma):
        return (
            + (np.exp(sigma) - 1) * self.lYs 
            - self.n * gammaln(np.exp(sigma))
            + self.a * sigma
            - self.b * np.exp(sigma)
            )

    def sample_alpha(self, curr_alpha):
        curr_log_alpha = np.log(curr_alpha)
        cand_log_alpha = normal(curr_log_alpha, scale = 5e-2)
        curr_lp = self.log_posterior_alpha(curr_log_alpha)
        cand_lp = self.log_posterior_alpha(cand_log_alpha)
        if np.log(uniform(1)) < cand_lp - curr_lp:
            return np.exp(cand_log_alpha)
        else:
            return np.exp(curr_log_alpha)
        pass
    
    def iter_sample(self):
        alpha = self.curr_alpha
        self.curr_iter += 1
        self.samples.alpha[self.curr_iter] = self.sample_alpha(alpha)
        return

    def __init__(self, Y, a, b):
        self.Y = Y
        self.lYs = np.log(Y).sum()
        self.n = Y.shape[0]
        self.a = a
        self.b = b
        return
    pass

class Adam(object):
    """ Adam optimizer """
    eps    : float = 1e-8
    rate   : float
    decay1 : float
    decay2 : float
    iter   : int

    theta    : np.ndarray # current value for theta
    momentum : np.ndarray # momentum
    sumofsqs : np.ndarray # sum of squares of past gradients

    def update(self):
        "performs adam optimization step on theta in-place"
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

    def initialization(
            self, 
            rate : float, 
            decay1 : float, 
            decay2 : float
            ):
        self.decay1 = decay1
        self.decay2 = decay2
        self.rate = rate
        self.iter = 0
        self.momentum = np.zeros(self.theta.shape[0])
        self.sumofsqs = np.zeros(self.theta.shape[0])
        return

    def optimize(self, niter):
        for _ in range(niter):
            self.update()
        return

    def __init__(self, theta, dloss, rate = 1e-3, decay1 = 0.9, decay2 = 0.999):
        self.dloss = dloss  # derivative of loss fn with respect to theta
        self.theta = theta  # initial value for theta
        self.initialization(rate, decay1, decay2)
        return

class UnivariateGammaGammaVB(object):
    Y   : np.ndarray
    n   : int
    lYs : np.ndarray
    a   : float
    b   : float
    ns  : int

    def gradient(self, mu, tau):
        epsilon = normal(size = self.ns)
        ete = np.exp(tau) * epsilon
        alpha = np.exp(mu + ete)
        # alpha = np.exp(self.mu + self.sigma * epsilon)
        dmu = (
            + alpha * self.lYs
            - self.n * digamma(alpha) * alpha
            + (self.a - 1)
            - self.b * alpha
            )
        # dtau = (
        #     + alpha * self.lYs * ete
        #     - self.n * digamma(alpha) * alpha * ete
        #     + (self.a - 1) * ete
        #     - b * alpha * ete
        #     ).mean()
        dtau = dmu * ete
        # dsigma = (
        #     + alpha * self.lYs * epsilon
        #     - self.n * digamma(alpha) * alpha * epsilon
        #     + (self.a - 1) * epsilon
        #     - b * alpha * epsilon
        #     ).mean()
        # return np.array((dmu, dsigma))
        return np.array((dmu.mean(), dtau.mean()))

    def __init__(self, Y, a, b, ns = 20):
        self.Y = Y
        self.n = Y.shape[0]
        self.lYs = np.log(Y).sum()
        self.a = a
        self.b = b
        self.ns = ns
        return
    pass

if __name__ == '__main__':
    Y = gamma(shape = 5, scale = 1, size = 10)
    a = 1
    b = 1
    
    mc = UnivariateGammaGammaMC(Y, a, b)
    mc.sample(5000)
    # print(np.log(mc.samples.alpha[-2000:]).mean())
    # print(np.log(mc.samples.alpha[-2000:]).std())
    vb = UnivariateGammaGammaVB(Y, a, b)

    def dloss(theta):
        return -vb.gradient(*theta)
    
    adam = Adam(np.array((0.,0.)), dloss, rate = 1e-1, decay1 = 0.9, decay2 = 0.999)
    adam.optimize(100)
    print(adam.theta)
    
    
    # mc = UnivariateGammaGammaMC(Y, a, b)
    # mc.sample(5000)

    # plt.plot(np.arange(5000), mc.samples.alpha[1:])
    # plt.show()

# EOF