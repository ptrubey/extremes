"""
Model description for Dirichlet-process Mixture of Projected Gammas on unit p-sphere
---
PG is unrestricted (allow betas to vary)
Centering distribution is product of Gammas
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
from io import BytesIO

from cUtility import diriproc_cluster_sampler, generate_indices
from samplers import BaseSampler
from cProjgamma import sample_alpha_1_mh_summary
from data import euclidean_to_angular, euclidean_to_hypercube, Data_From_Sphere
from projgamma import GammaPrior, logd_prodgamma_my_mt, logd_prodgamma_my_st,   \
    logd_prodgamma_paired, logd_gamma

def sample_dirichlet_shape_wrapper(args):
    return sample_alpha_1_mh_summary(*args)

Prior = namedtuple('Prior', 'zeta')

class Samples(object):
    zeta  = None
    r     = None
    ld    = None

    def __init__(self, nSamp, nDat, nCol):
        self.zeta  = np.empty((nSamp + 1, nCol))
        self.r     = np.empty((nSamp + 1, nDat))
        self.ld    = np.empty((nSamp + 1))
        return

class Chain(BaseSampler):
    @property
    def curr_zeta(self):
        return self.samples.zeta[self.curr_iter]
    @property
    def curr_r(self):
        return self.samples.r[self.curr_iter]
    
    def sample_r(self, zeta):
        As = zeta.sum()
        Bs = self.data.Yp.sum(axis = 1)
        return gamma(shape = As, scale = 1 / Bs)
    
    def sample_zeta(self, curr_zeta, r):
        Y = r[:, None] * self.data.Yp
        Ysv = Y.sum(axis = 0)
        lYsv = np.log(Y).sum(axis = 0)
        args = zip(curr_zeta, repeat(self.nDat), Ysv, lYsv, 
                   repeat(self.priors.zeta.a), repeat(self.priors.zeta.b))
        res = map(sample_dirichlet_shape_wrapper, args)
        return np.array(list(res))
    
    def initialize_sampler(self, ns):
        self.samples = Samples(ns, self.nDat, self.nCol)
        self.samples.zeta[0] = gamma(shape = 2., scale = 2., size = self.nCol)
        self.samples.r[0] = self.sample_r(self.samples.zeta[0])
        self.curr_iter = 0
        return

    def record_log_density(self):
        lpl = 0.
        lpp = 0.
        Y = self.curr_r[:,None] * self.data.Yp
        lpl += logd_prodgamma_my_st(Y, self.curr_zeta, np.ones(self.curr_zeta.shape)).sum()
        lpp += logd_gamma(self.curr_zeta, *self.priors.zeta).sum()
        self.samples.ld[self.curr_iter] = lpl + lpp
        return

    def iter_sample(self):
        zeta = self.curr_zeta
        r    = self.curr_r

        self.curr_iter += 1

        self.samples.r[self.curr_iter] = self.sample_r(zeta)
        self.samples.zeta[self.curr_iter] = self.sample_zeta(zeta, self.curr_r)
        
        self.record_log_density()
        return
    
    def write_to_disk(self, path, nBurn, nThin = 1):
        if type(path) is str:
            folder = os.path.split(path)[0]
            if not os.path.exists(folder):
                os.mkdir(folder)
            if os.path.exists(path):
                os.remove(path)
        
        zetas = self.samples.zeta[nBurn::nThin]
        rs    = self.samples.r[nBurn::nThin]

        out = {
            'zetas'  : zetas,
            'rs'     : rs,
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

    def __init__(
            self,
            data,
            prior_zeta  = (1., 1.),
            p           = 1,
            **kwargs
            ):
        self.data = data
        self.p = p
        self.nCol = self.data.nCol
        self.nDat = self.data.nDat
        _prior_zeta = GammaPrior(*prior_zeta)
        self.priors = Prior(_prior_zeta)
        self.set_projection()
        return

class Result(object):
    def generate_posterior_predictive_gammas(self, n_per_sample = 1, m = 10):
        new_gammas = []
        for s in range(self.nSamp):
            new_gammas.append(
                gamma(shape = self.samples.zeta[s], size = (n_per_sample, self.nCol))
                )
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
        
        zetas  = out['zetas']
        rs     = out['rs']

        self.nSamp = zetas.shape[0]
        self.nDat  = rs.shape[1]
        self.nCol  = zetas.shape[1]

        self.data = Data_From_Sphere(out['V'])
        try:
            self.data.fill_outcome(out['Y'])
        except KeyError:
            pass

        self.samples       = Samples(self.nSamp, self.nDat, self.nCol)
        self.samples.zeta  = zetas
        self.samples.r     = rs
        self.samples.ld    = out['logd']
        return

    def __init__(self, path):
        self.load_data(path)
        return

if __name__ == '__main__':
    pass

    from data import Data_From_Raw, euclidean_to_simplex
    from projgamma import GammaPrior
    from pandas import read_csv
    from io import BytesIO
    import os

    raw = read_csv('./datasets/ivt_nov_mar.csv')
    data = Data_From_Raw(raw, decluster = True, quantile = 0.95)
    model = Chain(data, prior_eta = GammaPrior(2, 1), p = 1)
    model.sample(50000)
    out = BytesIO()
    model.write_to_disk(out, 40000, 20)
    res = Result(out)
    pp = res.generate_posterior_predictive_hypercube(10)
    pps = euclidean_to_simplex(pp)
    print(pps.shape)
    print(pps.sum(axis = 1))


# EOF
