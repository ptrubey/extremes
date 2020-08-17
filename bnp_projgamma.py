from projgamma import *
from scipy.stats import gamma, gmean
from numpy.random import choice
from collections import namedtuple
import numpy as np

BNPPGPrior = namedtuple('BNPPGPrior', 'alpha beta eta')
Theta      = namedtuple('Theta','alpha beta')

class BNPPG(object):
    samples_theta = []
    samples_delta = None

    def sample_beta_j(self, alpha_j, Yj):
        prop_beta = np.empty(self.nCol)
        prop_beta[0] = 1.
        for k in range(1,self.nCol):
            prop_beta[k] = sample_beta_fc(alpha_j, Yj, self.priors.beta)
        return prop_beta

    def sample_alpha_j(self, Yj):
        """ Sampler for the alpha vector associated with cluster j """
        prop_alpha = np.empty(self.nCol)
        prop_alpha[0] = sample_alpha_1_slice(
            gmean(Yj[:,0]), Yj[:,0], self.priors.alpha,
            )
        for k in range(1, self.nCol):
            prop_alpha[k] = sample_alpha_k_slice(
                gMean(Yj[:,k]), Yj[:,k], self.priors.alpha, self.priors.beta,
                )
        return prop_alpha

    def sample_theta_j(self, R, delta, j):
        """ Sampler for theta parameter set associated with cluster j """
        delta_idx = np.which(delta == j)
        Yj = (self.data.Yl[delta_idx].T * self.samples_r[delta_idx]).T
        



    def set_priors(self):
        self.alpha_prior = gamma(self.priors.alpha.a, 1 / self.priors.alpha.b)
        self.beta_prior  = gamma(self.priors.beta.a,  1 / self.priors.beta.b)
        return

    def initialize_sampler(self, ns):
        self.samples_delta = np.empty((ns + 1, self.nDat))
        self.samples_r     = np.empty((ns, self.nDat))
        self.samples_theta = []
        return

    def __init__(
            self,
            data,
            prior_alpha = GammaPrior(1.,1.),
            prior_beta = GammaPrior(1.,1.),
            prior_eta = DirichletPrior(5.),
            ):
        self.data = data
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        self.priors = BNPPGPrior(
            prior_alpha,
            prior_beta,
            prior_eta,
            )
        self.set_priors()
        return

# EOF
